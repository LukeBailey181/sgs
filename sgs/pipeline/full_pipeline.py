"""
Here we define full pipelines that can be run.

This is object oriented.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import os
import time
from typing import List, Dict, Any
import json
import multiprocessing as mp
import sys

import wandb
from termcolor import colored
from matplotlib.figure import Figure
import submitit

from sgs.data.dataset_types import IterationMetadata
from sgs.models import Guide
from sgs.pipeline.config import PipelineConfig
from sgs.pipeline.step1_data_gen import data_gen
from sgs.pipeline.step2_train import train_prover_and_conjecturer
from sgs.training.evaluate import evaluate_prover
from sgs.utils import (
    export,
    get_submitit_executor,
    example_get_master_running_config,
    cleanup_submitit_job,
)
from sgs.models.guide.llm_judge_guide import (  # noqa: F401
    DeepseekProverV2LemmaGuideLocal,
)

logger = logging.getLogger(__name__)


class PipelineRunner(ABC):
    def __init__(
        self,
        # Data
        pipeline_config: PipelineConfig,
    ):
        export()

        self.config: PipelineConfig = pipeline_config
        run_name: str = f"Pipeline:{self.__class__.__name__}_Guide:{self.config.guide_class}_CMode:{self.config.conjecturer_model_config.setup}"

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            logger.info(f"Running in SLURM job: {slurm_job_id}")
        else:
            logger.info("No SLURM job ID found")

        wandb_tags = list(self.config.wandb_tags) if self.config.wandb_tags else []
        if slurm_job_id:
            wandb_tags.append(f"slurm_job:{slurm_job_id}")

        wandb_checkpoint_path = Path(self.config.checkpoint_dir) / "wandb.json"
        if self.config.wandb_project is not None:
            if self.config.wandb_resume_id and wandb_checkpoint_path.exists():
                with open(wandb_checkpoint_path, "r") as f:
                    wandb_run_id = json.load(f)["wandb_run_id"]

                # Resume existing run
                self.wandb_run = wandb.init(
                    name=run_name,
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    config=self.config.to_dict(),
                    tags=wandb_tags,
                    resume="must",  # Will error if ID does not exist
                    id=wandb_run_id,  # The run ID to resume
                )
                logger.info(f"Resumed wandb run: {self.wandb_run.id}")

                # Use the entity wandb actually resolved (handles entity=None -> user default).
                api = wandb.Api()
                run_path = f"{self.wandb_run.entity}/{self.config.wandb_project}/{wandb_run_id}"
                historical_run = api.run(run_path)
                history = historical_run.history(keys=["num_generations"])

                if not history.empty:
                    last_value = history["num_generations"].iloc[-1]
                    self.total_num_generations = int(last_value)
                    print(
                        f"Recovered total num generations: {self.total_num_generations}"
                    )
                else:
                    if self.config.num_generations_at_start is not None:
                        self.total_num_generations = (
                            self.config.num_generations_at_start
                        )
                    else:
                        raise ValueError(
                            "Could not retrieve last generations from wandb"
                        )

            else:
                self.wandb_run = wandb.init(
                    name=run_name,
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    config=self.config.to_dict(),
                    tags=wandb_tags,
                )

                with open(wandb_checkpoint_path, "w") as f:
                    json.dump({"wandb_run_id": self.wandb_run.id}, f)

                # If this has been provided
                if self.config.num_generations_at_start is not None:
                    self.total_num_generations = self.config.num_generations_at_start
                else:
                    self.total_num_generations = 0
        else:
            self.wandb_run = None

        # Set this up as the path to the data that we start at?
        self.conjecturer_dataset_load_path = self.config.conjecturer_dataset_path
        self.prover_dataset_load_path = self.config.prover_dataset_path

        self.starting_prover_model_path = self.config.prover_model_config.model_name
        self.starting_conjecturer_model_path = (
            self.config.conjecturer_model_config.model_name
        )

        # Parallel job handling
        # Non sequential eval jobs are handled in parallel
        # We keep at most 1 eval job running simultaneously with other jobs
        self.eval_job_queue: List = []

        self.total_cost: float = 0.0

    def check_parallel_jobs(self, block=False):
        """
        Polls parallel jobs and cleans up after finished jobs.

        For now we only fun eval jobs in parallel so this is all we need to do.

        Args:
            block: If True, we will block and wait for the job to finish.
        """

        new_eval_job_queue: List = []

        for job in self.eval_job_queue:
            # Check if the job is done:
            if job.done() or block:
                # Now we can handle logging
                result: List[Dict[str, Any]] = job.result()
                if self.wandb_run is not None:
                    for log_item in result:
                        for key, value in log_item.items():
                            if isinstance(value, Figure):
                                self.wandb_run.log({key: wandb.Image(value)})
                            else:
                                self.wandb_run.log({key: value})

                # The job is done, so we clean up those pesky files
                cleanup_submitit_job(job)
            else:
                new_eval_job_queue.append(job)

        self.eval_job_queue = new_eval_job_queue

    def run(self):
        logger.info(colored("*" * 100, "green", attrs=["bold"]))
        logger.info(
            colored(
                f"Running pipeline with {self.config.end_iteration - self.config.start_iteration} iterations",
                "green",
                attrs=["bold"],
            )
        )
        logger.info(
            colored(
                "We will start with an evaluation of the prover model",
                "green",
                attrs=["bold"],
            )
        )
        logger.info(colored("*" * 100, "green", attrs=["bold"]))

        save_proofs_path = (
            Path(self.config.checkpoint_dir)
            / "eval_proofs"
            / f"eval_proofs_initial_model.{self.config.save_data_extension}"
        )
        if (
            self.config.start_iteration == 0
            and not Path(save_proofs_path).exists()
            and self.config.eval_init_model
            and self.config.eval_datasets
        ):
            if self.config.eval_in_seperate_job:
                eval_executor = get_submitit_executor(example_get_master_running_config())
                eval_job = eval_executor.submit(
                    evaluate_prover,
                    prover_config=self.config.prover_model_config,
                    gen_resources_config=self.config.eval_gen_resources_config,
                    eval_datasets=self.config.eval_datasets,
                    verifier_address=self.config.verification_address,
                    best_of_n=self.config.eval_best_of_n,
                    verification_resources_config=self.config.eval_verification_resources_config,
                    wandb_run=None,  # Set to None as submitting wandb runs doesn't work
                    save_proofs_path=str(save_proofs_path),
                    iteration_metadata=IterationMetadata(
                        num_generated_conjectures=0,
                        num_target_statements=0,
                        proofs_per_statement=0,
                        num_generated_proofs=0,
                        num_generations=0,
                        num_generated_tokens=0,
                        num_input_tokens=0,
                    ),
                )
                self.eval_job_queue.append(eval_job)

            else:
                result: List[Dict[str, Any]] = evaluate_prover(
                    prover_config=self.config.prover_model_config,
                    gen_resources_config=self.config.eval_gen_resources_config,
                    eval_datasets=self.config.eval_datasets,
                    verifier_address=self.config.verification_address,
                    best_of_n=self.config.eval_best_of_n,
                    verification_resources_config=self.config.eval_verification_resources_config,
                    wandb_run=None,  # We set this to none as we want logging to be handled in this master process
                    save_proofs_path=save_proofs_path,
                    iteration_metadata=IterationMetadata(
                        num_generated_conjectures=0,
                        num_target_statements=0,
                        proofs_per_statement=0,
                        num_generated_proofs=0,
                        num_generations=0,
                        num_generated_tokens=0,
                        num_input_tokens=0,
                    ),
                )

                if self.wandb_run is not None:
                    for log_item in result:
                        for key, value in log_item.items():
                            if isinstance(value, Figure):
                                self.wandb_run.log({key: wandb.Image(value)})
                            else:
                                self.wandb_run.log({key: value})

        start_iteration = self.config.start_iteration

        for i in range(start_iteration, self.config.end_iteration):
            # Check if we have exceed our budget
            if self.total_cost > self.config.budget:
                logger.info(colored("*" * 100, "red", attrs=["bold"]))
                logger.info(
                    colored(
                        f"We have exceeded our budget of {self.config.budget}",
                        "red",
                        attrs=["bold"],
                    )
                )
                logger.info(
                    colored("We will stop the pipeline here.", "red", attrs=["bold"])
                )
                logger.info(colored("*" * 100, "red", attrs=["bold"]))
                return

            self.run_iteration(i)

        # Now we are done
        if self.wandb_run is not None:
            self.wandb_run.finish()

    @abstractmethod
    def run_iteration(self, i: int):
        # We run a full round of STP
        # 1. Data gen, conjectures, proofs of conjectures and statements
        # 2. Train guide, review, train prover, train conjecturer
        pass


class PipelineRunnerStandard(PipelineRunner):
    def get_conjecturer_dataset_name(self, iteration: int) -> str:
        return f"conjecturer_dataset_{iteration}.{self.config.save_data_extension}"

    def get_prover_dataset_name(self, iteration: int) -> str:
        return f"prover_dataset_{iteration}.{self.config.save_data_extension}"

    def get_model_name(self, iteration: int) -> str:
        return f"model_prover_conjecturer_iteration_{iteration}"

    def get_prover_model_name(
        self,
        iteration: int,
        checkpoint_dir: str,
    ) -> str:
        if self.config.parameter_sharing:
            ending = f"model_prover_conjecturer_iteration_{iteration}"
        else:
            ending = f"model_prover_iteration_{iteration}"

        return str(Path(checkpoint_dir) / ending)

    def get_conjecturer_model_name(self, iteration: int, checkpoint_dir: str) -> str:
        if self.config.parameter_sharing:
            ending = f"model_prover_conjecturer_iteration_{iteration}"
        else:
            ending = f"model_conjecturer_iteration_{iteration}"

        return str(Path(checkpoint_dir) / ending)

    def check_if_datasets_exist(self, checkpoint_dir: Path, iteration: int) -> bool:
        # List all files in checkpoint dir
        files = os.listdir(checkpoint_dir)

        conjecturer_dataset_name = self.get_conjecturer_dataset_name(iteration)
        prover_dataset_name = self.get_prover_dataset_name(iteration)

        if conjecturer_dataset_name in files and prover_dataset_name in files:
            return True
        if conjecturer_dataset_name in files and prover_dataset_name not in files:
            raise ValueError(
                f"We checkpointed the conjectuer dataset but not the prover dataset for iteration {iteration}?"
            )
        if conjecturer_dataset_name not in files and prover_dataset_name in files:
            raise ValueError(
                f"We checkpointed the prover dataset but not the conjectuer dataset for iteration {iteration}?"
            )

        return False

    def check_if_models_exists(
        self,
        prover_model_save_path: Path,
        conjecturer_model_save_path: Path,
        iteration: int,
    ) -> bool:
        # HF models won't be stored on disk so we keep track of them here

        hf_models_list = [
            "deepseek-ai/DeepSeek-Prover-V2-7B",
            "LukeBailey181Pub/dspv2_guide",
        ]

        def local_model_is_complete(path: Path) -> bool:
            # A valid HF model dir saves config.json; treat bare dirs (e.g. from
            # a crashed run) as missing so we retrain rather than crash on load.
            return path.exists() and (path / "config.json").exists()

        prover_exists = (
            local_model_is_complete(Path(prover_model_save_path))
            or str(prover_model_save_path) in hf_models_list
        )
        conjecturer_exists = (
            local_model_is_complete(Path(conjecturer_model_save_path))
            or str(conjecturer_model_save_path) in hf_models_list
        )
        return prover_exists and conjecturer_exists

    def run_iteration(self, iteration: int):
        # Run data generation
        # Fail early on instantiating the guide
        if self.config.guide_config is not None:
            guide: Guide | None = eval(self.config.guide_class)(  # type: ignore
                self.config.guide_config
            )
        else:
            guide = None

        start_iteration_time = time.time()
        start_data_gen_time = time.time()
        logger.info(colored("*" * 100, "cyan", attrs=["bold"]))
        logger.info(
            colored(
                f"STEP A: DATA GENERATION for iteration {iteration}",
                "cyan",
                attrs=["bold"],
            )
        )
        logger.info(colored("*" * 100, "cyan", attrs=["bold"]))

        # We will periodically check the parallel jobs
        self.check_parallel_jobs()

        # We set up the save paths for the data
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"iteration_{iteration}"
        eval_dir = Path(self.config.checkpoint_dir) / "eval_proofs"

        eval_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        conjecturer_dataset_save_path = (
            checkpoint_dir / self.get_conjecturer_dataset_name(iteration)
        )
        prover_dataset_save_path = checkpoint_dir / self.get_prover_dataset_name(
            iteration
        )

        # first we check if the datasets already exist
        iteration_metadata: IterationMetadata | None = None
        if not self.check_if_datasets_exist(checkpoint_dir, iteration):
            # We do not have a checkpoint of the dataset, so we need to generate it
            iteration_metadata = data_gen(
                pipeline_proving_and_verification=self.config.pipeline_proving_and_verification,
                conjecturer_config=self.config.conjecturer_model_config,
                prover_config=self.config.prover_model_config,
                conjecturer_dataset_path=self.conjecturer_dataset_load_path,
                prover_dataset_path=self.prover_dataset_load_path,
                iteration=iteration,
                conjectures_per_statement=self.config.conjectures_per_statement,
                gen_resources_config=self.config.gen_resources_config,
                proofs_per_sample=self.config.proofs_per_sample,
                subsample_target_statements=self.config.subsample_target_statements,
                current_num_generations=self.total_num_generations,
                batch_target_statements=self.config.batch_target_statements,
                batching_check_file_path=str(
                    Path(self.config.checkpoint_dir) / "batching_check.json"
                ),
                # We save to the same path as we load the data from
                conjecturer_dataset_save_path=str(conjecturer_dataset_save_path),
                prover_dataset_save_path=str(prover_dataset_save_path),
                # Verifier
                verifier_address=self.config.verification_address,
                verifier_timeout=self.config.verifier_timeout,
                verification_resources_config=self.config.verification_resources_config,
                num_master_verification_workers=self.config.num_master_verification_workers,
                # Wandb
                statement_selection_mode=self.config.statement_selection_mode,
                wandb_run=self.wandb_run,
            )

            self.total_num_generations = iteration_metadata.total_num_generations  # type: ignore
            assert (
                self.total_num_generations is not None
            ), "Total num generations should be set"

        else:
            # Log that we already have the datasets
            logger.info(
                colored(
                    f"We already have the datasets for iteration {iteration} checkpointed, so we move on.",
                    "cyan",
                    attrs=["bold"],
                )
            )

        # We will periodically check the parallel jobs
        self.check_parallel_jobs()

        # Update the load paths
        self.conjecturer_dataset_load_path = str(conjecturer_dataset_save_path)
        self.prover_dataset_load_path = str(prover_dataset_save_path)

        # Model save dir is the parent directory of wherever the
        prover_model_save_path: Path = Path(
            self.get_prover_model_name(iteration, checkpoint_dir)
        )
        conjecturer_model_save_path: Path = Path(
            self.get_conjecturer_model_name(iteration, checkpoint_dir)
        )

        end_data_gen_time = time.time()

        logger.info(
            colored(
                f"STEP A: Data gen time: {end_data_gen_time - start_data_gen_time} seconds",
                "cyan",
                attrs=["bold"],
            )
        )
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "timing/step_A_data_gen_time(mins)": (
                        end_data_gen_time - start_data_gen_time
                    )
                    / 60,
                    "iteration": iteration,
                }
            )

        start_train_time = time.time()

        logger.info(colored("*" * 100, "cyan", attrs=["bold"]))
        logger.info(
            colored(
                f"STEP B: TRAINING for iteration {iteration}", "cyan", attrs=["bold"]
            )
        )
        logger.info(colored("*" * 100, "cyan", attrs=["bold"]))

        freeze_conjecturer = self.config.freeze_conjecturer
        if not self.config.parameter_sharing:
            if (
                iteration < self.config.num_conjecturer_warmup_rounds
                or self.config.freeze_prover
            ):
                # If you are in a conjecturer warmup round, you freeze the prover
                freeze_prover = True

                # If we are freezing the prover, then the prover_save_path is the same as current path
                prover_model_save_path = Path(
                    self.config.prover_model_config.model_name
                )
            else:
                freeze_prover = False

            if freeze_conjecturer:
                # for updating purposes
                conjecturer_model_save_path = Path(
                    self.config.conjecturer_model_config.model_name
                )

        else:
            # You can't freeze the models if they are sharing params!
            freeze_prover = False
            assert (
                not freeze_conjecturer
            ), "You can't freeze the models if they are sharing params!"

        if self.config.retrain_prover_from_scratch:
            # We need to set the prover config model name to the original
            self.config.prover_model_config.model_name = self.starting_prover_model_path
        if self.config.retrain_conjecturer_from_scratch:
            # We need to set the conjecturer config model name to the original
            self.config.conjecturer_model_config.model_name = (
                self.starting_conjecturer_model_path
            )

        # Now you basically have the paths that you will save the conjecturer and prover models to
        # prover_model_save_path and conjecturer_model_save_path
        # If both exist, then you can just load them and move on

        if not self.check_if_models_exists(
            prover_model_save_path=Path(prover_model_save_path),
            conjecturer_model_save_path=Path(conjecturer_model_save_path),
            iteration=iteration,
        ):
            # We do not have a checkpoint of the models(s), so we train them
            cost = train_prover_and_conjecturer(
                iteration=iteration,
                checkpoint_dir=self.config.checkpoint_dir,
                # Prover information
                prover_dataset_path=self.prover_dataset_load_path,
                prover_config=self.config.prover_model_config,
                prover_iterations_in_buffer=self.config.prover_iterations_in_buffer,
                conjecture_multiplier=self.config.conjecture_multiplier,
                num_prover_train_examples=self.config.num_train_examples,
                # Conjecturer information
                conjecturer_dataset_path=self.conjecturer_dataset_load_path,
                conjecturer_config=self.config.conjecturer_model_config,
                conjecturer_iterations_in_buffer=self.config.conjecturer_iterations_in_buffer,
                # Shared information
                prover_model_save_path=str(prover_model_save_path),
                conjecturer_model_save_path=str(conjecturer_model_save_path),
                training_config=self.config.training_config,
                resources_config=self.config.training_resources_config,
                wandb_run=self.wandb_run,
                stp_round=iteration,
                parameter_sharing=self.config.parameter_sharing,
                freeze_prover=freeze_prover,
                freeze_conjecturer=freeze_conjecturer,
                statement_selection_mode=self.config.statement_selection_mode,
                # guide specific
                guide=guide,
                guide_resources_config=self.config.guide_resources_config,
            )

            self.total_cost += cost
        else:
            logger.info(
                colored(
                    f"We already have the model for iteration {iteration} checkpointed, so we move on.",
                    "cyan",
                    attrs=["bold"],
                )
            )

        self.config.prover_model_config.model_name = str(prover_model_save_path)
        self.config.conjecturer_model_config.model_name = str(
            conjecturer_model_save_path
        )

        end_train_time = time.time()

        # We will periodically check the parallel jobs
        self.check_parallel_jobs()

        logger.info(
            colored(
                f"STEP B: Train time: {end_train_time - start_train_time} seconds",
                "cyan",
                attrs=["bold"],
            )
        )
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "timing/step_B_train_time(mins)": (
                        end_train_time - start_train_time
                    )
                    / 60,
                    "iteration": iteration,
                }
            )

        start_eval_time = time.time()

        logger.info(colored("*" * 100, "cyan", attrs=["bold"]))
        logger.info(
            colored(
                f"STEP C: EVALUATION for iteration {iteration}", "cyan", attrs=["bold"]
            )
        )
        logger.info(colored("*" * 100, "cyan", attrs=["bold"]))

        save_proofs_path: str = str(
            eval_dir
            / f"eval_proofs_end_of_iteration_{iteration}.{self.config.save_data_extension}"
        )

        if not self.config.eval_datasets:
            logger.info(
                colored(
                    "No eval datasets configured, skipping evaluation",
                    "cyan",
                    attrs=["bold"],
                )
            )
        elif iteration % self.config.eval_every != 0 or iteration == 0:
            logger.info(
                colored(
                    f"We are not evaluating this iteration as we only eval every {self.config.eval_every} iteration",
                    "cyan",
                    attrs=["bold"],
                )
            )
        elif not Path(save_proofs_path).exists():
            if self.config.eval_in_seperate_job:
                self.check_parallel_jobs(block=True)

                assert (
                    len(self.eval_job_queue) == 0
                ), "We should have no eval jobs running"
                eval_executor = get_submitit_executor(example_get_master_running_config())
                eval_job = eval_executor.submit(
                    evaluate_prover,
                    prover_config=self.config.prover_model_config,
                    gen_resources_config=self.config.eval_gen_resources_config,
                    eval_datasets=self.config.eval_datasets,
                    verifier_address=self.config.verification_address,
                    best_of_n=self.config.eval_best_of_n,
                    verification_resources_config=self.config.eval_verification_resources_config,
                    wandb_run=None,  # We set this to none as we want logging to be handled in this master process
                    save_proofs_path=save_proofs_path,
                    iteration_metadata=iteration_metadata,
                )
                self.eval_job_queue.append(eval_job)

            else:
                # We are not running the evaluation in a separate job, so we need to do it here
                result: List[Dict[str, Any]] = evaluate_prover(
                    prover_config=self.config.prover_model_config,
                    gen_resources_config=self.config.eval_gen_resources_config,
                    eval_datasets=self.config.eval_datasets,
                    verifier_address=self.config.verification_address,
                    best_of_n=self.config.eval_best_of_n,
                    verification_resources_config=self.config.eval_verification_resources_config,
                    wandb_run=None,  # We set this to none as we want logging to be handled in this master process
                    save_proofs_path=save_proofs_path,
                    iteration_metadata=iteration_metadata,
                )

                if self.wandb_run is not None:
                    for log_item in result:
                        for key, value in log_item.items():
                            if isinstance(value, Figure):
                                self.wandb_run.log({key: wandb.Image(value)})
                            else:
                                self.wandb_run.log({key: value})

        else:
            logger.info(
                colored(
                    f"We already have the eval proofs for iteration {iteration} checkpointed, so we move on.",
                    "cyan",
                    attrs=["bold"],
                )
            )

        end_eval_time = time.time()

        logger.info(
            colored(
                f"STEP C: Eval time: {end_eval_time - start_eval_time} seconds",
                "cyan",
                attrs=["bold"],
            )
        )
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "timing/step_C_eval_time(mins)": (end_eval_time - start_eval_time)
                    / 60,
                    "iteration": iteration,
                }
            )

        end_iteration_time = time.time()

        logger.info(
            colored(
                f"Iteration {iteration} time: {end_iteration_time - start_iteration_time} seconds",
                "red",
                attrs=["bold"],
            )
        )
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "timing/iteration_time(mins)": (
                        end_iteration_time - start_iteration_time
                    )
                    / 60,
                    "iteration": iteration,
                }
            )


def run_pipeline(pipeline_config: PipelineConfig):
    pipeline: PipelineRunner = eval(pipeline_config.pipeline_class)(pipeline_config)
    pipeline.run()


# This will automatically reschedule jobs that are preemted
class RunPipelineCheckpointable(submitit.helpers.Checkpointable):
    def __call__(self, pipeline_config: PipelineConfig) -> None:
        pipeline: PipelineRunner = eval(pipeline_config.pipeline_class)(pipeline_config)
        pipeline.run()

    def checkpoint(self, *args, **kwargs):
        """
        Override checkpoint to only allow the main process to checkpoint.
        Child processes spawned by ProcessPoolExecutor should not try to checkpoint.
        """
        current_process = mp.current_process()

        # Only checkpoint if we're in the main process
        if current_process.name != "MainProcess":
            print(
                f"Checkpoint signal received in child process '{current_process.name}' "
                f"(pid={current_process.pid}). Skipping checkpoint - only main process should handle this."
            )
            # Return None to indicate we should not requeue from this child process
            sys.exit(0)

        # We're in the main process, proceed with normal checkpointing
        print(
            "Checkpoint signal received in main process. Proceeding with checkpoint and requeue."
        )
        return super().checkpoint(*args, **kwargs)
