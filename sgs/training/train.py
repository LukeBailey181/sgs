from typing import List, Tuple, Optional, Dict
import logging
import time
import os
import tempfile
import pickle
import subprocess
import shutil

import torch
import wandb
from submitit.core.utils import FailedJobError, UncompletedJobError

from sgs.models.model_types import ModelConfig, ModelType, ResourcesConfig
from sgs.training.training_types import TrainingConfig, TrainingSampleDatum
from sgs.training.train_local import finetune_local
from sgs.utils import SubmititCleanupExecutor

logger = logging.getLogger(__name__)


def finetune_model(
    group_size: int,
    trainer_cls: str,
    model_save_path: str,
    train_data: List[TrainingSampleDatum],
    training_config: TrainingConfig,
    resources_config: ResourcesConfig,
    model_config: ModelConfig,
    iteration: Optional[int] = None,
    val_data: Optional[List[Tuple[str, str]]] = None,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    wandb_log_prefix: str = "",
) -> None:
    """ 
    Finetune a model on the train data. This function basically serves
    as a switcher between different trainign paths and submit it or not.

    Args:
        model_save_path: Path to save the finetuned model
        train_data: List of tuples of (prompt, target, weight, log_probs)
        training_config: Training configuration
        resources_config: Resources configuration
        model_config: Model configuration
        val_data: Optional list of tuples of (prompt, target) for validation
    """
    result = []

    if len(train_data) == 0:
        print(f"No training data provided")
        print(f"Copying input model to the output path")
        if os.path.exists(model_config.model_name):
            shutil.copytree(model_config.model_name, model_save_path)
        else:
            print(f"Input model path does not exist: {model_config.model_name}")
            print(f"So we will just create an empty dir at the target")
            os.makedirs(model_save_path, exist_ok=True)

        return result

    if model_config.model_type == ModelType.LOCAL:
        if resources_config.submitit:
            num_retries = 3

            with SubmititCleanupExecutor(resources_config=resources_config) as executor:
                while num_retries > 0:
                    try:
                        job = executor.submit(
                            finetune_local,
                            group_size=group_size,
                            trainer_cls=trainer_cls,
                            model_save_path=model_save_path,
                            train_data=train_data,
                            training_config=training_config,
                            resources_config=resources_config,
                            model_config=model_config,
                            val_data=val_data,
                        )

                        # Wait for the job to finish
                        if resources_config.jobs_per_node > 1:
                            result: List[Dict[str, float]] = job.results()[0]
                        else:
                            result = job.result()  # type: ignore

                        # Training worked so break out of the loop
                        break

                    except (FailedJobError, UncompletedJobError) as e:
                        num_retries -= 1

                        logger.warning(
                            f"Training failed with error: {e}. "
                            f"Retries left: {num_retries}"
                        )

                        if num_retries == 0:
                            raise e
                        else:
                            # Sleep for 30 seconds before retrying
                            time.sleep(30)

        else:
            # Run locally via torchrun across all visible GPUs.
            num_gpus = torch.cuda.device_count()
            if num_gpus < 1:
                raise ValueError(
                    "Local training requires at least one visible CUDA device."
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                args_path = os.path.join(tmpdir, "finetune_args.pkl")
                with open(args_path, "wb") as f:
                    # These are the arguments for finetune_local
                    pickle.dump(
                        {
                            "group_size": group_size,
                            "model_save_path": model_save_path,
                            "train_data": train_data,
                            "training_config": training_config,
                            "resources_config": resources_config,
                            "model_config": model_config,
                            "val_data": val_data,
                            "trainer_cls": trainer_cls,
                        },
                        f,
                    )
                cmd = [
                    "torchrun",
                    "--standalone",
                    f"--nproc_per_node={num_gpus}",
                    "-m",
                    "sgs.training.train_local",
                    f"--args_pickle_path={args_path}",
                ]
                logger.info(
                    f"Launching local distributed training on {num_gpus} GPU(s): "
                    f"{' '.join(cmd)}"
                )
                _ = subprocess.run(cmd, check=True)

            # Collect metrics from trainer_state.json
            state_path = os.path.join(model_save_path, "log_history.pkl")
            result = []
            if os.path.exists(state_path):
                with open(state_path, "rb") as f:
                    result = pickle.load(f)

    # Now check if there is a checkpoint path in the load path and if so delete it 
    model_load_path = model_config.model_name
    if os.path.exists(model_load_path) and not training_config.reset_trainer:
        checkpoint_dirs = [x for x in os.listdir(model_load_path) if "checkpoint" in x]
        for checkpoint_dir in checkpoint_dirs:
            # check it is a dir and delete it
            if os.path.isdir(os.path.join(model_load_path, checkpoint_dir)):
                print(f"Deleting old checkpoint directory: {os.path.join(model_load_path, checkpoint_dir)}")
                shutil.rmtree(os.path.join(model_load_path, checkpoint_dir))


    if wandb_run is not None:
        for x in result:
            log_dict = {f"{wandb_log_prefix}/{k}": v for k, v in x.items()}

            if iteration is not None:
                log_dict["iteration"] = iteration

            wandb_run.log(log_dict)