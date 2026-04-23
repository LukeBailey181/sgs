"""
verify_local.py

Functions to run lean code verification LOCALLY
"""

from typing import List, Tuple, Optional
from dataclasses import asdict
import time
import sys
import random

from sgs.verification.prover.lean.verifier import Lean4ServerScheduler
from sgs.verification.types import VerificationOutput
from sgs.data.dataset_types import UtilizationReport, JobType
from sgs.utils.monitor import ResourceMonitor
from sgs.utils.server import SubmititWorker, SubmitResults, SubmitResult, TaskItem
from sgs.utils import export

LOCAL_VERIFIER_TIMEOUT = 200

"""
Functions in here are designed to be run on a worker machine, and return their results to the master.

verify_local takes in a list of proofs and verifies them, and returns. If this is run using submitit, then
the result is send back to the master.

verify_worker_to_server connects to a server that should be running on the master machine, and communicated with it
to verify proofs.
"""


def verify_local(
    proofs: List[str],
    num_workers: int,
    timeout: int,
    memory_limit: int = -1,
    monitor: bool = False,
    lean_version: str = "4.15",
) -> Tuple[List[VerificationOutput], UtilizationReport]:
    monitor_runner = ResourceMonitor(
        interval_sec=5.0,
        track_children=False,
        per_process_top_n=0,
        track_cgroup=True,
        profile_gpu=False,
        timings_only=not monitor,
    )
    monitor_runner.start()

    verifier_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=num_workers, timeout=timeout, memory_limit=memory_limit,
        lean_version=lean_version,
    )

    # Submit all proofs for verification
    request_id_list = verifier_scheduler.submit_all_request(proofs)

    # Get all verification outputs
    verification_outputs = verifier_scheduler.get_all_request_outputs(request_id_list)

    verdicts = [
        output.get("complete", False) if isinstance(output, dict) else False
        for output in verification_outputs
    ]
    return_data = [
        VerificationOutput(
            verdict=verdict,
            output=output,
            system_error="system_errors" in output
            and output[
                "system_errors"
            ],  # The server is setup to return "error" key if sys error, else lean compiler has "errors" key.
        )
        for verdict, output in zip(verdicts, verification_outputs)
    ]

    if verifier_scheduler is not None:
        verifier_scheduler.close()

    # Stop monitor and build report
    report: UtilizationReport = monitor_runner.stop_and_report(num_examples=len(proofs))
    report.job_type = JobType.VERIFICATION.value

    return return_data, report


class VerifyWorkerToServerTask(SubmititWorker):
    def graceful_exit(
        self,
        monitor: ResourceMonitor | None,
        verifier_scheduler: Lean4ServerScheduler | None,
        reason: Optional[str] = None,
    ) -> None:

        try:

            if verifier_scheduler is not None:
                verifier_scheduler.close()

            self.report_dead(reason=reason)

            if monitor is not None:
                if monitor.has_started:
                    util_report: UtilizationReport = monitor.stop_and_report(
                        num_examples=self.num_examples_processed
                    )
                    util_report.job_type = JobType.VERIFICATION.value
                    self.submit_util_report(util_report)
                    # Erase the monitor
                    monitor = None
        
        except Exception as e:
            print(f"Error in graceful exit: {e}")

        sys.exit(1)

    def __call__(
        self,
        num_workers: int,
        timeout: int,
        memory_limit: int = -1,
        lean_version: str = "4.15",
    ) -> None:
        self.exit_reason = "Unkown"
        verifier_scheduler: Lean4ServerScheduler | None = None

        try:
            export()

            self.monitor = ResourceMonitor(
                interval_sec=10,
                track_children=False,
                track_cgroup=False,
                profile_gpu=False,
                timings_only=not self.do_in_depth_monitoring,
            )
            self.monitor.start()

            verifier_scheduler = Lean4ServerScheduler(
                max_concurrent_requests=num_workers,
                timeout=timeout,
                memory_limit=memory_limit,
                lean_version=lean_version,
            )

            buffer_size = num_workers * 2

            tasks: List[TaskItem] = self.get_tasks(num_tasks=buffer_size).tasks

            currently_processing: List[Tuple[TaskItem, int]] = []  # (task, request_id)

            # Now lets submit all these tasks
            request_id_list = verifier_scheduler.submit_all_request(
                [task.task for task in tasks]
            )
            assert len(request_id_list) == len(tasks)
            currently_processing = list(zip(tasks, request_id_list))

            # We keep going while there are still things in the buffer or we are still processing
            status_check_interval = 2
            last_status_check = time.time()
            report_util_interval = 20
            last_report_util = time.time()

            print("=" * 100)
            print("BEGINNING VERIFICATION LOOP WITH THE SERVER")
            print("=" * 100)

            while True:
                # We wait a jittered amount of time to avoid overloading the server
                time.sleep(random.uniform(0.5, 1.5))

                # First we check if any of the currently processing tasks are done
                now = time.time()
                if now - last_status_check > status_check_interval:
                    last_status_check = now

                if now - last_report_util > report_util_interval:
                    util_report: UtilizationReport = self.monitor.report(
                        num_examples=self.num_examples_processed
                    )
                    self.submit_util_report(report=util_report)
                    last_report_util = now

                completed_tasks: List[SubmitResult] = []
                not_completed_tasks: List[Tuple[TaskItem, int]] = []
                for task, request_id in currently_processing:
                    output = verifier_scheduler.get_request_status(request_id)
                    if output is not None:
                        # Result is the actual verifier result
                        verification_output = VerificationOutput(
                            verdict=output.get("complete", False),
                            output=output,
                            system_error="system_errors" in output
                            and output[
                                "system_errors"
                            ],  # The server is setup to return "error" key if sys error, else lean compiler has "errors" key.
                        )

                        # Construct the submit result
                        result = SubmitResult(
                            task_id=task.task_id,
                            worker_id=self.worker_id,
                            result=asdict(verification_output),
                        )
                        completed_tasks.append(result)
                    else:
                        not_completed_tasks.append((task, request_id))

                currently_processing = not_completed_tasks

                # Now we fill up the buffer
                tasks = []
                we_got_no_new_tasks = False
                if len(currently_processing) < buffer_size:
                    tasks = self.get_tasks(
                        num_tasks=buffer_size - len(currently_processing)
                    ).tasks

                    if len(tasks) > 0:
                        request_id_list = verifier_scheduler.submit_all_request(
                            [task.task for task in tasks]
                        )
                        assert len(request_id_list) == len(tasks)
                        currently_processing.extend(
                            [
                                (task, request_id)
                                for task, request_id in zip(tasks, request_id_list)
                            ]
                        )

                    else:
                        we_got_no_new_tasks = True

                # Now we submit the completed tasks
                if completed_tasks:
                    # Probably need to do some error handling here?
                    self.submit_results(SubmitResults(results=completed_tasks))
                    self.num_examples_processed += len(completed_tasks)

                if we_got_no_new_tasks:
                    print(
                        "We got no tasks from the server, so we will wait 1 second, and try again"
                    )
                    time.sleep(1)

            self.exit_reason = "finished"

        except Exception as e:
            print(f"Error in worker: {e}")
            self.exit_reason = f"Error in worker: {e}"
            raise

        finally:
            self.graceful_exit(
                self.monitor, verifier_scheduler, reason=self.exit_reason
            )
