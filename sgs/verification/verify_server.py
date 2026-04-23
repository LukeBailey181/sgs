from typing import Optional
import multiprocessing as mp

from sgs.utils import get_submitit_executor
from sgs.utils.server import TaskServer
from sgs.verification.verify_local import VerifyWorkerToServerTask


class VerifyServer(TaskServer):
    def __init__(self, lean_version: str = "4.15", **kwargs):
        super().__init__(**kwargs)
        self.lean_version = lean_version

    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        if self._tasks.qsize() == 0 and len(self._in_progress) == 0:
            raise ValueError(
                "There is no work to be done and you are trying to launch jobs."
            )

        resources_config = self.worker_resources_config
        if not resources_config.submitit:
            return

        executor = get_submitit_executor(resources_config)

        if num_workers is None:
            num_workers = resources_config.num_jobs

        for _ in range(num_workers):
            worker = VerifyWorkerToServerTask(
                self.server_address, monitor=self.in_depth_monitoring
            )
            job = executor.submit(
                worker,
                num_workers=resources_config.cpus_per_task,
                timeout=self.verify_timeout,
                lean_version=self.lean_version,
            )
            self.workers.append(job)

    # This is a fast and dirty solution
    def launch_master_worker(self, num_workers: int) -> None:
        """
        Launches a worker as a subprocess on the master machine.
        """

        worker = VerifyWorkerToServerTask(
            self.server_address, monitor=self.in_depth_monitoring
        )

        job = mp.Process(
            target=worker.__call__,
            args=(
                num_workers,
                self.verify_timeout,
            ),
            kwargs={"lean_version": self.lean_version},
            name="verify_master_worker",
        )

        job.start()

        self.master_worker_processes.append(job)
