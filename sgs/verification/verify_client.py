"""
verify_client.py

Functions to submit lean code for verification to a verifier server.
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
import time

from submitit.core.utils import FailedJobError, UncompletedJobError
from submitit.core.core import Job

from sgs.verification.types import VerificationOutput
from sgs.verification.verify_local import verify_local
from sgs.models.model_types import ResourcesConfig
from sgs.utils import (
    chunk_list,
    SubmititCleanupExecutor,
)
from sgs.utils.monitor import UtilizationReport
from sgs.utils import get_job_start_and_end_times, export
from sgs.verification.verify_server import VerifyServer

logger = logging.getLogger(__name__)

# This is a magin number for now
# We have increased this from 60 to 200 to allow for more time for the verifier to run


def verify_lean_code(
    verifier_address: str,
    lean_code: List[str],
    timeout: int,
    resources_config: Optional[ResourcesConfig] = None,
    monitor: bool = False,
    master_num_workers: int = 0,
    lean_version: str = "4.15",
) -> Tuple[List[VerificationOutput], List[UtilizationReport]]:
    """Send Lean code to the verifier and check if it passes.

    Args:
        verifier_address: The address of the verifier server.
        lean_code: A list of strings, each representing a FULL lean file, header + theorem + proof.
        resources_config: A ResourcesConfig object, which contains the resources to use for the verification.
        timeout: The timeout for the verification.
        wandb_run: A wandb run object, which is used to log the utilization report.
        wandb_prefix: A prefix for the wandb run.


    Returns:
        A tuple containing two lists:
        - The first list contains the verdicts for each proof.
        - The second list contains the verification outputs for each proof.
    """
    export()

    if verifier_address == "server":
        # First we setup the server address

        # We will run the verification server locally
        assert (
            resources_config is not None
        ), "Resources config is required for server verification"
        with VerifyServer(
            worker_resources_config=resources_config,
            monitor=monitor,
            verify_timeout=timeout,
            lean_version=lean_version,
        ) as server:
            # Add tasks -> Launch workers
            task_ids: List[str] = server.add_tasks(lean_code)
            server.launch_workers()

            if master_num_workers > 0:
                print(f"Launching {master_num_workers} master workers")
                server.launch_master_worker(master_num_workers)  # type: ignore

            # Now we wait for the server to finish
            server.wait_until_done()

            # Now we get the results. Dict of task_id -> result
            results: Dict[str, Any] = server.results()

            # Now we convert the results to the verification outputs
            task_id_to_verification_output: Dict[str, VerificationOutput] = {}

            for task_id, r in results.items():
                assert (
                    task_id not in task_id_to_verification_output
                ), "Task id already exists"

                task_id_to_verification_output[task_id] = VerificationOutput(
                    verdict=r.get("verdict"),
                    output=r.get("output"),
                    system_error=r.get("system_error"),
                )

            # We need to put the results in the correct order
            verification_outputs: List[VerificationOutput] = [
                task_id_to_verification_output[task_id] for task_id in task_ids
            ]

            util_reports: List[UtilizationReport] = server.util_reports()
            max_concurrent_workers = server.max_concurrent_workers

        print(f"Max concurrent workers during verification: {max_concurrent_workers}")
        return verification_outputs, util_reports

    elif verifier_address == "local":
        # Run local verification

        # Then we will also set the number of workers on verify_local and ram depending on
        # what is in the resource config

        assert (
            resources_config is not None
        ), "Resources config is required for local verification"

        if resources_config.submitit:
            with SubmititCleanupExecutor(resources_config=resources_config) as executor:
                # We have a target set of jobs defined by examples_per_job
                # We chunk the data by this
                # We then maintain a pool of resource_config.num_jobs jobs
                # Everytime a job completes, we submit a new job with the next chunk

                if resources_config.examples_per_job == -1:
                    # We will submit everything as a single job
                    job = executor.submit(
                        verify_local,
                        proofs=lean_code,
                        num_workers=resources_config.cpus_per_task,
                        memory_limit=-1,
                        timeout=timeout,
                        monitor=monitor,
                        lean_version=lean_version,
                    )

                    output, report = job.result()

                    job_state_data: List[Tuple[str, float, Optional[float]]] = (
                        get_job_start_and_end_times(job.job_id)
                    )
                    if len(job_state_data) == 1:
                        # We got the correct data
                        _, start_time, _ = job_state_data[0]
                        if start_time is not None:
                            report.outer_job_start_time = start_time

                        # We use the time now as the true outer end time of this job
                        report.outer_job_end_time = time.time()

                    job_reports: List[UtilizationReport] = [report]

                else:
                    # We do some checking of the number of jobs
                    anticipated_proofs_in_pool = (
                        resources_config.examples_per_job * resources_config.num_jobs
                    )
                    if anticipated_proofs_in_pool > len(lean_code):
                        # In this case we just chunk smartly.
                        max_jobs: int = max(
                            1, len(lean_code) // resources_config.cpus_per_task
                        )
                        num_concurrent_jobs: int = max(
                            1, min(max_jobs, resources_config.num_jobs)
                        )

                        lean_code_chunks: List[List[str]] = chunk_list(
                            lean_code, n_chunks=num_concurrent_jobs
                        )

                        num_proofs_per_job = len(lean_code_chunks[0])

                    else:
                        # In this case we need to maintain a pool of jobs
                        # We do this by chunking the data into chunks of size examples_per_job
                        # We then maintain a pool of resource_config.num_jobs jobs
                        # Everytime a job completes, we submit a new job with the next chunk

                        total_jobs = len(lean_code) // resources_config.examples_per_job

                        lean_code_chunks = chunk_list(lean_code, n_chunks=total_jobs)
                        num_concurrent_jobs = resources_config.num_jobs

                        num_proofs_per_job = resources_config.examples_per_job

                    # Now we calculate the number of workers for each job and ram per worker
                    num_workers = resources_config.cpus_per_task

                    if resources_config.mem == "0":
                        # Mem 0 makes slurm assign all the memory on the node to the job
                        pass
                    else:
                        max_ram_per_worker = (
                            int(resources_config.mem.split("G")[0]) // num_workers
                        )
                        if max_ram_per_worker == 0:
                            raise ValueError("Not enough memory to run the jobs")

                    logger.info("-" * 100)
                    logger.info(f"Verification {len(lean_code)} proofs")
                    logger.info(f"Maintaining pool of {num_concurrent_jobs} jobs")
                    logger.info(
                        f"Overall there are {len(lean_code_chunks)} chunks of lean code to process"
                    )
                    logger.info(f"There will be {num_proofs_per_job} proofs per job")
                    logger.info(
                        f"Thus there is {num_proofs_per_job/num_concurrent_jobs} proofs per subprocess"
                    )
                    logger.info("-" * 100)

                    lean_code_chunks_with_id: List[Tuple[int, List[str]]] = [
                        (i, lean_code_chunk)
                        for i, lean_code_chunk in enumerate(lean_code_chunks)
                    ]
                    retries: List[int] = [3] * len(lean_code_chunks)

                    chunk_verifications: List[List[VerificationOutput]] = [
                        [] for _ in range(len(lean_code_chunks))
                    ]

                    # Each element is the chunk id and the job object
                    job_pool: List[Tuple[int, Job]] = []
                    job_reports = []

                    original_num_chunks = len(lean_code_chunks_with_id)
                    current_decile = 0
                    while len(lean_code_chunks_with_id) > 0 or len(job_pool) > 0:
                        # We have code to submit or jobs to wait for

                        # Give update every 10% completion
                        num_chunks_processed = sum(
                            [1 for x in chunk_verifications if len(x) > 0]
                        )
                        percent_complete = num_chunks_processed / original_num_chunks
                        decile = percent_complete // 0.1
                        if decile > current_decile:
                            current_decile = decile  # type: ignore
                            logger.info(
                                f"Verification {percent_complete*100:.1f}% complete"
                            )

                        # There are chunks to process!
                        # First we see if we should add a new job to the pool
                        if (
                            len(job_pool) < num_concurrent_jobs
                            and len(lean_code_chunks_with_id) > 0
                        ):
                            # We can add a new job to the pool
                            # We take the first chunk from the list
                            chunk_id, chunk = lean_code_chunks_with_id.pop(0)

                            job = executor.submit(
                                verify_local,
                                proofs=chunk,
                                num_workers=num_workers,
                                memory_limit=-1,  # Setting this to -1 for now, which does unlimited memory
                                timeout=timeout,
                                monitor=monitor,
                            )
                            job_pool.append((chunk_id, job))

                        completed_jobs = []
                        remaining_jobs = []
                        for chunk_id, job in job_pool:
                            if job.done():
                                completed_jobs.append((chunk_id, job))
                            else:
                                remaining_jobs.append((chunk_id, job))

                        # Replace the job pool with the remaining jobs
                        job_pool = remaining_jobs

                        # Now we check if any of the jobs are done:
                        for chunk_id, job in completed_jobs:
                            # Process the jobs

                            # We add the results to the chunk verifications
                            try:
                                result: List[VerificationOutput]
                                result, report = job.result()
                            except (FailedJobError, UncompletedJobError) as e:
                                logger.info(f"Job failed with error: {e}")
                                # Create failed verification outputs for each proof in the chunk
                                result = [
                                    VerificationOutput(
                                        verdict=False,
                                        output={
                                            "error": str(e),
                                            "system_errors": str(e),
                                        },
                                        system_error=True,
                                    )
                                    for _ in lean_code_chunks[chunk_id]
                                ]
                                report = None

                            job_state_data = get_job_start_and_end_times(job.job_id)
                            if len(job_state_data) == 1 and report is not None:
                                _, start_time, _ = job_state_data[0]
                                if start_time is not None:
                                    report.outer_job_start_time = start_time

                                # We use the time now as the true outer end time of this job
                                report.outer_job_end_time = time.time()

                            # check proporition of system errors
                            prop_system_errors = sum(
                                [1 for x in result if x.system_error]
                            ) / len(result)
                            num_system_errors = sum(
                                [1 for x in result if x.system_error]
                            )

                            if report is not None:
                                report.num_errors = num_system_errors
                                job_reports.append(report)

                            if prop_system_errors > 0.9:
                                # We need to retry
                                logger.info(
                                    f"Chunk {chunk_id} has {prop_system_errors*100:.1f}% system errors"
                                )
                                if retries[chunk_id] > 0:
                                    logger.info(
                                        f"Retrying chunk {chunk_id} {retries[chunk_id]} times"
                                    )
                                    retries[chunk_id] -= 1
                                    lean_code_chunks_with_id.append(
                                        (chunk_id, lean_code_chunks[chunk_id])
                                    )

                                else:
                                    logger.info(
                                        f"We are out of retries, so adding chunk {chunk_id} to the verifications"
                                    )
                                    # We are out of retries, we have to add this chunk
                                    chunk_verifications[chunk_id] = result
                            else:
                                # We did not have too many system errors, so we can add this chunk
                                chunk_verifications[chunk_id] = result

                    """
                    # Now we need to wait for all the remaining jobs to complete
                    for i, (chunk_id, job) in enumerate(job_pool):
                        try:
                            result: List[VerificationOutput] = job.result()
                            chunk_verifications[chunk_id] = result
                        except (FailedJobError, UncompletedJobError) as e:
                            logger.info(f"Job failed with error: {e}")
                            # Create failed verification outputs for each proof in the chunk
                            result = [
                                VerificationOutput(
                                    verdict=False, output={"error": str(e)}, system_error=True
                                )
                                for _ in lean_code_chunks[chunk_id]
                            ]
                            chunk_verifications[chunk_id] = result
                    """

                    # Now we flatten the verified chunks
                    output: List[VerificationOutput] = []  # type: ignore
                    for verified_chunk in chunk_verifications:
                        output.extend(verified_chunk)

        else:
            # We need to do some local things
            result, report = verify_local(
                proofs=lean_code,
                num_workers=resources_config.cpus_per_task,
                memory_limit=-1,
                timeout=timeout,
                monitor=monitor,
            )

            output = result
            job_reports = [report]

        # We are done with local processing
        return output, job_reports

    else:
        raise ValueError(
            f"Unknown verifier_address: {verifier_address!r}. Use 'server' or 'local'."
        )
