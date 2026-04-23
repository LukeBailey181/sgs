from typing import List, Tuple, Optional
import copy
import time
import sys
import random
import os

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import submitit
from vllm import LLM, SamplingParams
from dataclasses import asdict
import traceback
import multiprocessing as mp  # add near the top with other imports

from sgs.models.model_types import QueryResult, ModelConfig
from sgs.utils import chunk_list, export
from sgs.data.dataset_types import UtilizationReport, JobType
from sgs.utils.monitor import ResourceMonitor
from sgs.utils.server import SubmititWorker, SubmitResult, TaskItem, SubmitResults


def vllm_result_to_entropy(result) -> Tuple[float, List[float]]:
    entropies = []

    log_probs_of_sampled_tokens: List[float] = []

    assert result.outputs[0].logprobs is not None, "Logprobs must be present"

    generated_tokens: List[int] = result.outputs[0].token_ids

    assert len(generated_tokens) == len(
        result.outputs[0].logprobs
    ), "Number of generated tokens and logprobs must match"

    for token_logprobs, generated_token in zip(
        result.outputs[0].logprobs, generated_tokens
    ):
        if token_logprobs:
            # Extract logprobs for top-k tokens at this position
            logprobs_list = [lp.logprob for lp in token_logprobs.values()]

            # Convert log probs to probabilities
            probs = np.exp(logprobs_list)

            # Normalize (should already sum to ~1 for top-k)
            probs = probs / probs.sum()

            # Calculate entropy: H = -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        # Now get the sampled token logprob
        generated_token_logprob = token_logprobs[generated_token].logprob
        log_probs_of_sampled_tokens.append(generated_token_logprob)

    avg_entropy = float(np.mean(entropies)) if entropies else -1

    return avg_entropy, log_probs_of_sampled_tokens


def process_chunk_on_gpu(chunk_data):
    """Process a chunk of prompts on a specific GPU."""
    gpu_id, chunk_prompts, model_config = chunk_data
    if not chunk_prompts:  # Skip empty chunks
        return []

    try:
        seed = gpu_id + random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set CUDA_VISIBLE_DEVICES to only show the specific GPU for this subprocess
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set the GPU for this process
        torch.cuda.set_device(
            0
        )  # Use GPU 0 since CUDA_VISIBLE_DEVICES makes it the only visible GPU

        # Load model on this specific GPU
        vllm_model = LLM(
            model=model_config.model_name,
            dtype=model_config.dtype,
            gpu_memory_utilization=0.9,  # Use most of the GPU memory
            seed=seed,
        )

        sampling_params = SamplingParams(
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            # skip_special_tokens=False,
            # include_stop_str_in_output=True,
            logprobs=10,
        )

        if model_config.chat:
            if model_config.use_system_prompt:
                conversations = [
                    [
                        {"role": "system", "content": model_config.system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    for prompt in chunk_prompts
                ]
            else:
                conversations = [
                    [{"role": "user", "content": prompt}] for prompt in chunk_prompts
                ]
            results = vllm_model.chat(conversations, sampling_params=sampling_params)
        else:
            results = vllm_model.generate(
                chunk_prompts, sampling_params=sampling_params
            )

        # Extract token counts from the results
        query_results = []
        for result in results:
            input_token_count = len(result.prompt_token_ids)
            output_token_count = len(result.outputs[0].token_ids)

            avg_entropy, log_probs_of_sampled_tokens = vllm_result_to_entropy(result)

            query_results.append(
                QueryResult(
                    response_text=result.outputs[0].text,
                    input_token_count=input_token_count,
                    output_token_count=output_token_count,
                    is_error=False,
                    average_entropy=avg_entropy,
                    log_probs=log_probs_of_sampled_tokens,
                    output_tokens=result.outputs[0].token_ids,
                )
            )

        return query_results

    except Exception as e:
        print(f"Error in GPU {gpu_id} subprocess: {e}")
        raise
    finally:
        # Explicit cleanup (though OS will handle it anyway)
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        torch.cuda.empty_cache()


def query_model_batch_local(
    prompts: List[str],
    model_config: ModelConfig,
) -> List[QueryResult]:
    num_gpus = torch.cuda.device_count()  # Should be 8
    if num_gpus < 8:
        print(f"Warning: Only {num_gpus} GPUs available, using {num_gpus} instead of 8")
        num_gpus = num_gpus

    # Split prompts into chunks for each GPU using the robust chunk_list function
    prompt_chunks = chunk_list(prompts, num_gpus)

    # Process chunks in parallel across GPUs using subprocesses
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        chunk_data = [(i, chunk, model_config) for i, chunk in enumerate(prompt_chunks)]
        results = list(executor.map(process_chunk_on_gpu, chunk_data))

    # Flatten results
    all_results = []
    for result in results:
        if result is None:
            raise ValueError("Error in GPU subprocess when querying model")
        all_results.extend(result)

    return all_results


class QueryBatchLocalCheckpointableTask(submitit.helpers.Checkpointable):
    def __init__(self, num_checkpoints: int = 10):
        self.reset_state(num_checkpoints)
        self.monitor_runner: ResourceMonitor | None = None

    def reset_state(self, num_checkpoints: int):
        self.prompts_to_process: List[List[str]] = []
        self.processed_outputs: List[List[QueryResult]] = []
        self.num_prompts: int = 0
        self.num_checkpoints = num_checkpoints
        self.original_num_chunks: int | None = None
        self.setup_done: bool = False
        # This is a list of util reports as if we have interrupts we save multiple reports
        self.util_reports: List[UtilizationReport] = []

        self.num_examples_processed_in_current_job: int = 0

    def __call__(
        self, prompts: List[str], model_config: ModelConfig
    ) -> Tuple[List[QueryResult], List[UtilizationReport]]:
        export()

        self.monitor_runner = ResourceMonitor(
            interval_sec=5.0,
            track_children=False,
            per_process_top_n=0,
            track_cgroup=True,
            profile_gpu=False,
            timings_only=True,
        )
        self.monitor_runner.start()

        if not self.setup_done:
            # This is the first time running, so set prompts to process to prompts
            self.reset_state(self.num_checkpoints)

            prompts_chunks: List[List[str]] = chunk_list(prompts, self.num_checkpoints)
            self.prompts_to_process = prompts_chunks
            self.num_prompts = len(prompts)
            self.original_num_chunks = len(prompts_chunks)

            self.setup_done = True

        vllm_model = LLM(
            model=model_config.model_name,
            dtype=model_config.dtype,
        )

        sampling_params = SamplingParams(
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            # include_stop_str_in_output=True,
            # skip_special_tokens=False,
            logprobs=10,  # For entropy calculation
        )

        # Process each chunk one by one, saving progress after each chunk

        # Copy prompt chunks as we edit this
        prompts_to_process_static = copy.deepcopy(self.prompts_to_process)

        for prompts_chunk in prompts_to_process_static:
            query_results: List[QueryResult] = []

            if model_config.chat:
                if model_config.use_system_prompt:
                    conversations: List = [
                        [
                            {
                                "role": "system",
                                "content": model_config.system_prompt,
                            },
                            {"role": "user", "content": prompt},
                        ]
                        for prompt in prompts
                    ]
                else:
                    conversations = [
                        [{"role": "user", "content": prompt}] for prompt in prompts
                    ]

                    results = vllm_model.chat(
                        conversations, sampling_params=sampling_params
                    )
            else:
                # Run the model
                results = vllm_model.generate(
                    prompts_chunk, sampling_params=sampling_params
                )

            # Extract token counts from the results
            for result in results:
                input_token_count = len(result.prompt_token_ids)
                output_token_count = len(result.outputs[0].token_ids)

                avg_entropy, log_probs_of_sampled_tokens = vllm_result_to_entropy(
                    result
                )

                query_results.append(
                    QueryResult(
                        response_text=result.outputs[0].text,
                        input_token_count=input_token_count,
                        output_token_count=output_token_count,
                        is_error=False,
                        average_entropy=avg_entropy,
                        log_probs=log_probs_of_sampled_tokens,
                        output_tokens=result.outputs[0].token_ids,
                    )
                )

            # Now save the progress
            # Create new state atomically
            # Pop the first chunk as we just pocessed it
            new_prompts = self.prompts_to_process[1:]
            new_outputs = self.processed_outputs + [query_results]

            # Technically this is not atomic so we can be interrupted
            # between two lines, but we handle that in checkpointing
            self.prompts_to_process = new_prompts
            self.processed_outputs = new_outputs
            self.num_examples_processed_in_current_job += len(prompts_chunk)

        # When we are done return the processed outputs
        output: List[QueryResult] = []
        for processed_outputs in self.processed_outputs:
            output.extend(processed_outputs)

        report: UtilizationReport = self.monitor_runner.stop_and_report(
            num_examples=self.num_examples_processed_in_current_job
        )
        report.job_type = JobType.GENERATION.value
        self.util_reports.append(report)

        return output, self.util_reports

    def checkpoint(self, *args, **kwargs):
        if self.monitor_runner is not None:
            report: UtilizationReport = self.monitor_runner.stop_and_report(
                num_examples=self.num_examples_processed_in_current_job
            )
            report.job_type = JobType.GENERATION.value
            self.util_reports.append(report)

            self.num_examples_processed_in_current_job = 0

        if not self.setup_done:
            # Nothing got processed yet, so reset and try again
            print("Preemted during setup, resetting")
            self.reset_state(self.num_checkpoints)
        elif (
            len(self.processed_outputs) + len(self.prompts_to_process)
            != self.original_num_chunks
        ):
            print("Race condition detected, starting job over")
            # There was an interruption as we were changing state, lets just start over
            self.reset_state(self.num_checkpoints)
        else:
            percent_complete = len(self.processed_outputs) / self.original_num_chunks  # type: ignore
            print(f"Checkpointing {percent_complete*100:.2f}% complete")

        # Set to none so this can be pickled
        self.monitor_runner = None

        return super().checkpoint(*args, **kwargs)


class QueryWorkerToServerTask(SubmititWorker):
    def graceful_exit(
        self,
        monitor: ResourceMonitor | None,
        reason: Optional[str] = None,
    ) -> None:
        self.report_dead(reason=reason)

        if monitor is not None:
            if monitor.has_started:
                util_report: UtilizationReport = monitor.stop_and_report(
                    num_examples=self.num_examples_processed
                )
                util_report.job_type = JobType.GENERATION.value
                self.submit_util_report(util_report)
                # Erase the monitor
                monitor = None

        sys.exit(1)

    def __call__(
        self,
        model_config: ModelConfig,
        buffer_size: int = 128,
        gpu_id: Optional[int] = None,
    ) -> None:
        self.exit_reason = "Unkown"

        if gpu_id is not None:
            # Set CUDA_VISIBLE_DEVICES to only show the specific GPU for this subprocess
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # Set the GPU for this process
            torch.cuda.set_device(
                0
            )  # Use GPU 0 since CUDA_VISIBLE_DEVICES makes it the only visible GPU

        try:
            self.monitor = ResourceMonitor(
                interval_sec=10,
                track_children=False,
                track_cgroup=False,
                profile_gpu=False,
                timings_only=not self.do_in_depth_monitoring,
            )
            self.monitor.start()

            vllm_model = LLM(
                model=model_config.model_name,
                dtype=model_config.dtype,
                max_model_len=model_config.max_tokens,
            )

            sampling_params = SamplingParams(
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                top_k=model_config.top_k,
                # skip_special_tokens=False,
                # include_stop_str_in_output=True,
                logprobs=10,
            )

            # We keep going while there are still things in the buffer or we are still processing
            status_check_interval = 2
            last_status_check = time.time()
            report_util_interval = 20
            last_report_util = time.time()

            print("=" * 100)
            print("BEGINNING GENERATION LOOP WITH THE SERVER")
            print("=" * 100)

            # We keep going while there are still things in the buffer or we are still processing
            while True:
                # First we do some reporting to the server
                time.sleep(random.uniform(0.5, 1.5))
                now = time.time()
                if now - last_status_check > status_check_interval:
                    last_status_check = now

                    if self.get_done():
                        # We break out into the finally block, and gracefully exit
                        break

                if now - last_report_util > report_util_interval:
                    util_report: UtilizationReport = self.monitor.report(
                        num_examples=self.num_examples_processed
                    )
                    self.submit_util_report(report=util_report)
                    last_report_util = now

                # Get tasks
                tasks: List[TaskItem] = self.get_tasks(num_tasks=buffer_size).tasks

                if len(tasks) == 0:
                    print(
                        "We got no tasks from the server, so we will wait 1 second, and try again"
                    )
                    time.sleep(1)
                    continue

                # Process them all
                prompts: List[str] = [task.task for task in tasks]
                if model_config.chat:
                    if model_config.use_system_prompt:
                        conversations: List = [
                            [
                                {
                                    "role": "system",
                                    "content": model_config.system_prompt,
                                },
                                {"role": "user", "content": prompt},
                            ]
                            for prompt in prompts
                        ]
                    else:
                        conversations = [
                            [{"role": "user", "content": prompt}] for prompt in prompts
                        ]

                    results = vllm_model.chat(
                        conversations, sampling_params=sampling_params
                    )
                else:
                    # Run the model
                    results = vllm_model.generate(
                        prompts, sampling_params=sampling_params
                    )

                # Extract token counts from the results
                query_results: List[QueryResult] = []
                for result in results:
                    input_token_count = len(result.prompt_token_ids)
                    output_token_count = len(result.outputs[0].token_ids)

                    avg_entropy, log_probs_of_sampled_tokens = vllm_result_to_entropy(
                        result
                    )

                    query_results.append(
                        QueryResult(
                            response_text=result.outputs[0].text,
                            input_token_count=input_token_count,
                            output_token_count=output_token_count,
                            is_error=False,
                            average_entropy=avg_entropy,
                            log_probs=log_probs_of_sampled_tokens,
                            output_tokens=result.outputs[0].token_ids,
                        )
                    )

                completed_tasks: List[SubmitResult] = []
                for task, query_result in zip(tasks, query_results):
                    result = SubmitResult(
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        result=asdict(query_result),
                    )
                    completed_tasks.append(result)

                self.num_examples_processed += len(completed_tasks)

                # Now we submit the completed tasks
                if completed_tasks:
                    # Probably need to do some error handling here?
                    self.submit_results(SubmitResults(results=completed_tasks))
                    self.num_examples_processed += len(completed_tasks)

            self.exit_reason = "finished"

        except Exception as e:
            print(f"error in worker: {e}")
            full_error = traceback.format_exc()
            print(f"Full error: {full_error}")
            self.exit_reason = f"Error in worker: {e}"
            raise

        finally:
            self.graceful_exit(self.monitor, reason=self.exit_reason)
