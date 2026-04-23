from typing import List, Dict, Any, Tuple, Optional
import time
import copy
import logging

import wandb
import matplotlib.pyplot as plt
from enum import Enum

from sgs.models.model_types import (
    ResourcesConfig,
    QueryResult,
    ModelConfig,
)
from submitit.core.utils import FailedJobError, UncompletedJobError
from sgs.utils import get_job_start_and_end_times
from sgs.models.query_local import (
    query_model_batch_local,
    QueryBatchLocalCheckpointableTask,
)
from sgs.utils import chunk_list, SubmititCleanupExecutor
from sgs.utils.monitor import UtilizationReport
from sgs.models.query_server import QueryServer

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    CHUNKED = "chunked"
    SERVER = "server"


def log_token_counts(
    wandb_run: wandb.sdk.wandb_run.Run | None,
    model_responses: List[QueryResult],
    log_prefix: str,
    iteration: Optional[int] = None,
    num_generations: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns data that can be logged to wandb.

    If wandb_run given then this data is logged inside this function.
    """
    if len(model_responses) == 0:
        return []

    to_log = []

    # Now lets do some logging of the prompt and response token counts
    input_token_counts: List[int] = [
        response.input_token_count for response in model_responses
    ]
    output_token_counts: List[int] = [
        response.output_token_count for response in model_responses
    ]
    # Create histogram of input token counts
    fig = plt.figure(figsize=(10, 6))
    plt.hist(input_token_counts, bins=30)
    plt.title("Input Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")

    if wandb_run is not None:
        wandb_run.log({f"{log_prefix}/input_token_histogram": wandb.Image(plt)})
    else:
        to_log.append({f"{log_prefix}/input_token_histogram": copy.deepcopy(fig)})

    plt.close()

    # Create histogram of output token counts
    fig = plt.figure(figsize=(10, 6))
    plt.hist(output_token_counts, bins=30)
    plt.title("Output Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")

    if wandb_run is not None:
        wandb_run.log({f"{log_prefix}/output_token_histogram": wandb.Image(plt)})
    else:
        to_log.append({f"{log_prefix}/output_token_histogram": copy.deepcopy(fig)})

    plt.close()

    # Get the average input and output token counts
    avg_input_token_count = sum(input_token_counts) / len(input_token_counts)
    avg_output_token_count = sum(output_token_counts) / len(output_token_counts)

    log_dict = {
        f"{log_prefix}/avg_input_token_count": avg_input_token_count,
        f"{log_prefix}/avg_output_token_count": avg_output_token_count,
        "iteration": iteration,
        "num_generations": num_generations,
    }
    if iteration is not None:
        log_dict["iteration"] = iteration
    if num_generations is not None:
        log_dict["num_generations"] = num_generations

    if wandb_run is not None:
        wandb_run.log(log_dict)
    else:
        to_log.append(log_dict)

    return to_log


def query_model_batch(
    prompts: List[str],
    model_config: ModelConfig,
    resources_config: ResourcesConfig,
    mode: QueryMode = QueryMode.SERVER,
) -> Tuple[List[QueryResult], List[UtilizationReport]]:
    if len(prompts) == 0:
        return [], []

    if resources_config.submitit:
        # We are using submitit to run the request on slurm
        if mode == QueryMode.SERVER:
            # Got to do some work here
            with QueryServer(
                worker_resources_config=resources_config,
                monitor=True,
                model_config=model_config,
            ) as server:
                # Add tasks -> Launch workers
                task_ids: List[str] = server.add_tasks(prompts)
                server.launch_workers()

                # Now we wait for the server to finish
                server.wait_until_done()

                # Now we get the results, Dict of task_id -> result
                results: Dict[str, Any] = server.results()

                # Now we convert the results to the verification outputs
                task_id_to_query_result: Dict[str, QueryResult] = {}

                for task_id, r in results.items():
                    assert (
                        task_id not in task_id_to_query_result
                    ), "Task id already exists"

                    task_id_to_query_result[task_id] = QueryResult(
                        response_text=r.get("response_text"),
                        input_token_count=r.get("input_token_count"),
                        output_token_count=r.get("output_token_count"),
                        is_error=r.get("is_error"),
                        cost=r.get("cost", 0),
                        average_entropy=r.get("average_entropy"),
                        log_probs=r.get("log_probs"),
                        output_tokens=r.get("output_tokens"),
                    )

                query_results: List[QueryResult] = [
                    task_id_to_query_result[task_id] for task_id in task_ids
                ]

                util_reports: List[UtilizationReport] = server.util_reports()
                max_concurrent_workers = server.max_concurrent_workers

            print(
                f"Max concurrent workers during verification: {max_concurrent_workers}"
            )
            return query_results, util_reports

        elif mode == QueryMode.CHUNKED:
            with SubmititCleanupExecutor(resources_config=resources_config) as executor:
                # Split prompts into resources_config.num_jobs chunks
                prompt_chunks = chunk_list(prompts, n_chunks=resources_config.num_jobs)

                # Run the jobs
                jobs = []
                print(f"Submitting {resources_config.num_jobs} jobs")
                for prompt_chunk in prompt_chunks:
                    # job = executor.submit(query_model_batch_local, prompt_chunk, model_config)
                    task = QueryBatchLocalCheckpointableTask()
                    job = executor.submit(task, prompt_chunk, model_config)
                    jobs.append(job)

                chunk_outputs: List[List[QueryResult]] = [[] for _ in range(len(jobs))]
                restarts = [3] * len(jobs)
                completed_jobs = [False] * len(jobs)
                num_completed = 0
                util_reports = []
                # We go into a loop monitoring the jobs:
                while True:
                    # Poll every half second
                    time.sleep(0.5)

                    if num_completed == len(jobs):
                        break

                    # We have jobs that are not done
                    for i, job in enumerate(jobs):
                        if completed_jobs[i]:
                            continue

                        if job.done():
                            try:
                                result: List[QueryResult]
                                reports: List[UtilizationReport]

                                result, reports = job.result()

                                job_state_data: List[
                                    Tuple[str, float, Optional[float]]
                                ] = get_job_start_and_end_times(job.job_id)

                                if len(job_state_data) == len(reports):
                                    # We were able to get the job state data for all attempts
                                    for job_data, report in zip(
                                        job_state_data, reports
                                    ):
                                        _, start_time, end_time = job_data
                                        if start_time is not None:
                                            report.outer_job_start_time = start_time
                                        if end_time is not None:
                                            report.outer_job_end_time = end_time

                                    # We actually overwrite the last job with the current time
                                    reports[-1].outer_job_end_time = time.time()

                            except (FailedJobError, UncompletedJobError) as e:
                                if restarts[i] > 0:
                                    restarts[i] -= 1

                                    # Sleep for 10s to give the job a chance to restart
                                    time.sleep(10)

                                    # Prepare a new job on this prompt chunk
                                    task = QueryBatchLocalCheckpointableTask()
                                    job = executor.submit(
                                        task, prompt_chunks[i], model_config
                                    )
                                    jobs[i] = job
                                else:
                                    raise e
                            else:
                                # We were able to gather the result from the job
                                chunk_outputs[i].extend(result)
                                util_reports.extend(reports)
                                num_completed += 1
                                completed_jobs[i] = True

                output = []
                for chunk_output in chunk_outputs:
                    output.extend(chunk_output)

            return output, util_reports

        else:
            raise ValueError("Query mode not supported")

    else:
        # We are running things locally
        # No util report for this for now
        return (query_model_batch_local(prompts, model_config), [])
