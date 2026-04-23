from __future__ import annotations

import multiprocessing as mp
import threading
import time
import uuid
import queue
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from abc import ABC, abstractmethod
from collections import defaultdict
import requests
import socket
import psutil

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn
from submitit.core.core import Job
import submitit

from sgs.models.model_types import ResourcesConfig
from sgs.data.dataset_types import UtilizationReport
from sgs.utils.monitor import ResourceMonitor
from sgs.utils import cleanup_submitit_job
from sgs.models.model_types import ModelConfig


class TaskItem(BaseModel):
    task_id: str
    task: str


class GetTasksResponse(BaseModel):
    tasks: List[TaskItem]


class SubmitResult(BaseModel):
    task_id: str
    worker_id: str
    result: Any


class SubmitResults(BaseModel):
    results: List[SubmitResult]


class StatusResponse(BaseModel):
    pending: int
    in_progress: int
    completed: int
    is_done: bool


class SubmitUtilReport(BaseModel):
    worker_id: str
    util_report: UtilizationReport


# This doesnt need to be secure, just avoid any collisions
AUTH_TOKEN = "lean_project"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TaskServer(ABC):
    """
    A lightweight HTTP task server for distributing a list of string tasks to many workers.
    - Start/stop programmatically from the master.
    - Workers GET a task, POST back a result.
    - In-memory storage; suitable for ephemeral runs.
    """

    def __init__(
        self,
        worker_resources_config: ResourcesConfig,
        monitor: bool = False,
        allow_idling: bool = False,
        # Optionaly used by some server types
        model_config: ModelConfig | None = None,
        # Optioanlly for the verifer server
        verify_timeout: int = 200,
    ):
        self.host = socket.gethostbyname(socket.gethostname())
        self.port = find_free_port()
        self.server_address = f"http://{self.host}:{self.port}"
        self.auth_token = AUTH_TOKEN
        self.num_dead_workers = 0
        self.worker_resources_config = worker_resources_config

        # This affects the is_done logic
        self.allow_idling = allow_idling

        self._tasks: queue.Queue[Tuple[str, str]] = queue.Queue()  # (task_id, task_str)
        self._in_progress: Dict[str, str] = {}  # task_id -> task_str
        self._worker_ids_to_task_ids: Dict[str, List[str]] = defaultdict(list)
        self._task_id_to_worker_ids: Dict[str, Set[str]] = defaultdict(set)
        # ^ some tasks can have multiple workers at the end of processing

        self._results: Dict[str, Any] = {}  # task_id -> result
        self._util_reports: Dict[
            str, UtilizationReport
        ] = {}  # worker_id -> Util report

        self._lock = threading.Lock()
        self._done_cv = threading.Condition(self._lock)

        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._running = False

        self.app = self._create_app()

        self.workers: List[Job] = []  # worker_id -> Job

        self.max_concurrent_workers = 0

        self.model_config = model_config
        self.in_depth_monitoring = monitor

        self.current_decile = 0
        self.num_tasks = 0

        self.verify_timeout = verify_timeout

        self.master_worker_processes: List[mp.Process] = []

    # ------------- Public API -------------
    @abstractmethod
    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        """
        Launch a number of workers based on the resources config.
        This should be implemented by the subclass.

        If num_workers not given, then we will use what is given in the resource config
        """
        ...

    def kill_workers(self) -> None:
        """
        This will clean up all the workers that the server launched. Important incase there are pending
        workers that never have to be used.
        """

        for job in self.workers:
            # Sleeping to make sure we do not overwhelm the Slurm API
            time.sleep(0.5)
            try:
                # avoid raising if scancel fails when Slurm is down
                job.cancel(check=False)
            except Exception as e:
                print(
                    f"Warning: failed to cancel job {getattr(job, 'job_id', None)}: {e}"
                )

        # Cleanup logs best-effort
        for job in self.workers:
            try:
                cleanup_submitit_job(job)
            except Exception:
                pass

        for process in self.master_worker_processes:
            print(f"Terminating master worker {process.name} (PID: {process.pid})")

            # Kill all children of this process first (VLLM subprocesses)

            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)

                # First try graceful termination
                for child in children:
                    try:
                        print(f"  Terminating child process PID: {child.pid}")
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                # Give children time to terminate
                gone, alive = psutil.wait_procs(children, timeout=5)

                # Force kill any remaining children
                for child in alive:
                    try:
                        print(f"  Force killing child process PID: {child.pid}")
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

                # Now terminate the parent process
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    print(
                        f"Force killing master worker {process.name} (PID: {process.pid})"
                    )
                    process.kill()
                    process.join()
            except Exception as e:
                print(f"Error in terminating master worker {process.name} (PID: {process.pid}): {e}")

            

    def start(self) -> None:
        if self._running:
            return
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="error"
        )
        self._uvicorn_server = uvicorn.Server(config=config)

        def _run():
            # This is a blocking call until should_exit is True
            self._uvicorn_server.run()

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        # Give the server a moment to bind; for robust code, poll /status, but this is usually sufficient.
        time.sleep(0.3)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._uvicorn_server = None
        self._server_thread = None
        self._running = False
        self.kill_workers()

    def add_tasks(self, tasks: List[str]) -> List[str]:
        """Add string tasks. Returns the list of generated task_ids."""
        ids: List[str] = []
        with self._lock:
            for t in tasks:
                task_id = str(uuid.uuid4())
                self._tasks.put((task_id, t))
                ids.append(task_id)

            self.num_tasks += len(tasks)
        with self._done_cv:
            self._done_cv.notify_all()
        return ids

    def results(self) -> Dict[str, Any]:
        """Copy of results dict: task_id -> result."""
        with self._lock:
            return dict(self._results)

    def util_reports(self) -> List[UtilizationReport]:
        """Copy of util_reports list."""
        with self._lock:
            return list(self._util_reports.values())

    def wait_until_done(
        self, poll_interval: float = 0.2, timeout: Optional[float] = None
    ) -> bool:
        """Block until all tasks are completed. Returns True if done, False if timeout."""
        start = time.time()

        grace_period_start: float | None = None
        grace_period_duration = 10

        with self._done_cv:
            while True:
                # Use this as opportunity to update the max concurrent workers, and log
                self.max_concurrent_workers = max(
                    self.max_concurrent_workers,
                    len(self._worker_ids_to_task_ids.keys()),
                )

                completed = len(self._results)
                percent_complete = completed / max(1, self.num_tasks)
                decile = int(percent_complete * 10)
                if decile > self.current_decile:
                    self.current_decile = decile  # type: ignore
                    print(f"Processing {percent_complete*100:.1f}% complete")

                if self._is_done_locked():
                    if grace_period_start is None:
                        grace_period_start = time.time()
                        print(
                            f"All tasks completed, starting grace period of {grace_period_duration}."
                        )

                    elif time.time() - grace_period_start > grace_period_duration:
                        return True
                else:
                    grace_period_start = None

                if timeout is not None and (time.time() - start) >= timeout:
                    return False

                self._done_cv.wait(timeout=poll_interval)

    def pending_count(self) -> int:
        return self._tasks.qsize()

    def in_progress_count(self) -> int:
        with self._lock:
            return len(self._in_progress)

    def completed_count(self) -> int:
        with self._lock:
            return len(self._results)

    def __enter__(self) -> "TaskServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------- Internal -------------

    def _is_done_locked(self) -> bool:
        # Must be called under self._lock
        is_done = self._tasks.empty() and not self._in_progress

        if self.allow_idling:
            # Also not done
            return False

        return is_done

    def _auth_dependency(self, token: Optional[str] = None):
        def _check():
            if self.auth_token is None:
                return
            if token is None:
                raise HTTPException(status_code=401, detail="Missing token")
            if token != self.auth_token:
                raise HTTPException(status_code=403, detail="Invalid token")

        return _check

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="TaskServer")

        def _extract_token(authorization: Optional[str]) -> Optional[str]:
            # Accept "Bearer <token>" or raw token in header "Authorization"
            if not authorization:
                return None
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
            return authorization

        @app.get("/status", response_model=StatusResponse)
        def status(authorization: Optional[str] = Header(None)):
            token = _extract_token(authorization)
            self._auth_dependency(token)()
            with self._lock:
                return StatusResponse(
                    pending=self._tasks.qsize(),
                    in_progress=len(self._in_progress),
                    completed=len(self._results),
                    is_done=self._is_done_locked(),
                )

        @app.get("/get_task", response_model=GetTasksResponse)
        def get_task(
            worker_id: str,
            num_tasks: int = 1,
            authorization: Optional[str] = Header(None),
        ):
            token = _extract_token(authorization)
            self._auth_dependency(token)()
            if num_tasks <= 0:
                return GetTasksResponse(tasks=[])
            items: List[TaskItem] = []
            with self._lock:
                while len(items) < num_tasks:
                    try:
                        task_id, task = self._tasks.get_nowait()
                        if task_id in self._results:
                            # task already completed; drop it and continue
                            continue
                        if task_id not in self._in_progress:
                            self._in_progress[task_id] = task
                        items.append(TaskItem(task_id=task_id, task=task))
                        self._worker_ids_to_task_ids[worker_id].append(task_id)
                        self._task_id_to_worker_ids[task_id].add(worker_id)
                    except queue.Empty:
                        # No pending tasks: assign a duplicate of an in-progress task (min-workers first)
                        if not self._in_progress:
                            # We are done
                            break
                        owned = set(self._worker_ids_to_task_ids.get(worker_id, []))
                        # candidates are tasks this worker doesn't already hold
                        candidates = [
                            tid for tid in self._in_progress.keys() if tid not in owned
                        ]
                        if not candidates:
                            break
                        # pick task with fewest current workers
                        tid = min(
                            candidates,
                            key=lambda t: len(
                                self._task_id_to_worker_ids.get(t, set())
                            ),
                        )
                        task = self._in_progress[tid]
                        items.append(TaskItem(task_id=tid, task=task))
                        self._worker_ids_to_task_ids[worker_id].append(tid)
                        self._task_id_to_worker_ids[tid].add(worker_id)

            with self._done_cv:
                self._done_cv.notify_all()
            return GetTasksResponse(tasks=items)

        @app.get("/report_dead_worker")
        def report_dead_worker(
            worker_id: str,
            reason: Optional[str] = None,
            authorization: Optional[str] = Header(None),
        ):
            token = _extract_token(authorization)
            self._auth_dependency(token)()
            with self._lock:
                if reason != "finished":
                    print(f"Worker {worker_id} has died.")
                    print(f"Reason: {reason}")

                task_ids = self._worker_ids_to_task_ids.pop(worker_id, [])
                for tid in task_ids:
                    s = self._task_id_to_worker_ids.get(tid)
                    if s is not None:
                        s.discard(worker_id)
                        if not s:
                            # There is now no workers working on this task
                            self._task_id_to_worker_ids.pop(tid, None)
                            # if task still tracked as in-progress and not completed, requeue
                            task = self._in_progress.pop(tid, None)
                            if task is not None and tid not in self._results:
                                self._tasks.put((tid, task))

                # If we are not done then we will launch a new worker
                self.num_dead_workers += 1

                if not self._is_done_locked():
                    # We are not done so launch another job to replace this dead one
                    print(
                        "Worker died, but there is still work to do, so launching new worker."
                    )
                    self.launch_workers(num_workers=1)

                else:
                    if reason != "finished":
                        print("No work left, so NOT launching a new worker.")

            with self._done_cv:
                self._done_cv.notify_all()
            return {"status": "ok"}

        @app.post("/submit_result")
        def submit_result(
            payload: SubmitResults, authorization: Optional[str] = Header(None)
        ):
            token = _extract_token(authorization)
            self._auth_dependency(token)()
            item_statuses: List[Dict[str, str]] = []
            with self._lock:
                for res in payload.results:
                    if res.task_id not in self._in_progress:
                        # Either duplicate or unknown
                        if res.task_id in self._results:
                            item_statuses.append(
                                {"task_id": res.task_id, "status": "duplicate"}
                            )
                            continue
                        item_statuses.append(
                            {"task_id": res.task_id, "status": "unknown"}
                        )
                        continue

                    # Normal completion
                    _ = self._in_progress.pop(res.task_id, None)
                    self._results[res.task_id] = res.result
                    item_statuses.append({"task_id": res.task_id, "status": "ok"})

                    # cleanup all workers holding this task
                    holders = self._task_id_to_worker_ids.pop(res.task_id, set())
                    for wid in holders:
                        lst = self._worker_ids_to_task_ids.get(wid)
                        if lst is not None:
                            while res.task_id in lst:
                                lst.remove(res.task_id)
                            if not lst:
                                del self._worker_ids_to_task_ids[wid]

            with self._done_cv:
                self._done_cv.notify_all()

            return {"status": "ok", "items": item_statuses}

        @app.post("/submit_util_report")
        def submit_util_report(
            payload: SubmitUtilReport, authorization: Optional[str] = Header(None)
        ):
            token = _extract_token(authorization)
            self._auth_dependency(token)()

            worker_id = payload.worker_id
            util_report = payload.util_report

            with self._lock:
                # Add the util report to the results
                self._util_reports[worker_id] = util_report

            return {"status": "ok"}

        return app


class SubmititWorker(submitit.helpers.Checkpointable, ABC):
    def __init__(self, server_address: str, monitor: bool = False) -> None:
        self.server_address = server_address.rstrip("/")
        self.worker_id = str(uuid.uuid4())
        self.auth_token = AUTH_TOKEN
        self.do_in_depth_monitoring = monitor
        self.monitor: ResourceMonitor | None = None
        self.num_examples_processed: int = 0

        self.im_dead = False
        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None: ...

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def get_tasks(self, num_tasks: int = 1) -> GetTasksResponse:
        url = f"{self.server_address}/get_task"
        params: Dict[str, Union[str, int]] = {
            "worker_id": self.worker_id,
            "num_tasks": int(num_tasks),
        }
        r = requests.get(url, headers=self._headers(), params=params, timeout=120)
        r.raise_for_status()

        # Load into GetTasksResponse
        response: GetTasksResponse = GetTasksResponse.model_validate(r.json())
        return response

    def get_done(self) -> bool:
        url = f"{self.server_address}/status"
        r = requests.get(url, headers=self._headers(), timeout=120)
        r.raise_for_status()
        status: StatusResponse = StatusResponse.model_validate(r.json())
        return status.is_done

    def submit_results(self, results_model: SubmitResults) -> Dict[str, Any]:
        payload = results_model.model_dump()
        url = f"{self.server_address}/submit_result"
        r = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def submit_util_report(self, report: UtilizationReport) -> Dict[str, Any]:
        url = f"{self.server_address}/submit_util_report"

        submit_util = SubmitUtilReport(worker_id=self.worker_id, util_report=report)
        payload = submit_util.model_dump()
        r = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def report_dead(self, reason: Optional[str] = None) -> None:
        if self.im_dead:
            return
        self.im_dead = True

        url = f"{self.server_address}/report_dead_worker"
        params = {"worker_id": self.worker_id}
        if reason is not None:
            params["reason"] = reason
        try:
            requests.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=10,
            )

        except requests.RequestException:
            pass

    def checkpoint(self, *args, **kwargs) -> None:
        self.report_dead(reason="Preempted")
        # Return None by default (no automatic resubmission). Subclasses can override
        # to return a submitit.helpers.DelayedSubmission for requeueing.

        if self.monitor is not None:
            if self.monitor.has_started:
                util_report: UtilizationReport = self.monitor.stop_and_report(
                    num_examples=self.num_examples_processed
                )

                # Submit util report
                try:
                    self.submit_util_report(util_report)
                except Exception:
                    pass

        # We return None because we DONT want to requeue this job, the
        # server is incharge of submitting a new worker and removing
        # this from the job list
        return None
