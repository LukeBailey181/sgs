from typing import List
import os

from sgs.utils import (
    get_standard_training_config,
    get_submitit_executor,
)
from sgs.utils.experiment_utils import (
    get_deepseek_prover_v2_prover_config,
    get_deepseek_prover_v2_conjecturer_config,
    example_get_master_running_config,
    get_local_running_config,
)
from sgs.data import DatasetType
from sgs.pipeline.config import PipelineConfig
from sgs.models.model_types import ConjecturerSetup
from sgs.pipeline.full_pipeline import run_pipeline, RunPipelineCheckpointable
from sgs.pipeline.config import StatementSelectionMode

def run_experiment(
    checkpoint_dir: str,
    lr: float,
    wandb_tags: List[str],
    run_local: bool = False,
):
    conjectures_per_statement = {
        DatasetType.D_3K: 1,
    }

    prover_config = get_deepseek_prover_v2_prover_config()

    # Turn all the penalties off
    prover_config.penalize_double_lean = False
    prover_config.penalize_try = True
    prover_config.dapo_length_penalty = True
    prover_config.stp_length_penalty = False
    prover_config.penalize_long_proof_str_over_1000 = False

    conjecturer_config = get_deepseek_prover_v2_conjecturer_config(
        ConjecturerSetup.TARGET_STATEMENT_ONLY_UNSOLVED
    )

    guide_config = None
    guide_class = None

    training_config = get_standard_training_config()
    training_config.learning_rate = lr
    training_config.prover_learning_rate = lr
    training_config.conjecturer_learning_rate = lr
    training_config.batch_size = 32
    training_config.reset_trainer = True

    training_resource_config = get_local_running_config()
    gen_resource_config = get_local_running_config()
    guide_resource_config = get_local_running_config()
    eval_gen_resource_config = get_local_running_config()
    verification_resource_config = get_local_running_config()
    eval_verification_resource_config = get_local_running_config()

    pipeline_config = PipelineConfig(
        save_data_extension="pkl",
        eval_init_model=False,
        num_master_verification_workers=110,
        pipeline_proving_and_verification=True,
        eval_in_seperate_job=False,
        eval_gen_resources_config=eval_gen_resource_config,
        # Data
        prover_dataset_path="data/D_3k_prover_dataset.json",
        conjecturer_dataset_path="data/conjecturer_dataset.json",
        proofs_per_sample=8,
        # Models
        checkpoint_dir=checkpoint_dir,
        # Training
        training_config=training_config,
        retrain_prover_from_scratch=False,
        retrain_conjecturer_from_scratch=False,
        # Prover
        prover_iterations_in_buffer=1,
        num_train_examples=None,
        conjecture_multiplier=None,
        prover_model_config=prover_config,
        # Conjecturer
        conjectures_per_statement=conjectures_per_statement,
        conjecturer_model_config=conjecturer_config,
        conjecturer_iterations_in_buffer=1,
        # Guide
        guide_class=guide_class,  # We eval() this to instantiate the guide
        guide_config=guide_config,
        # Verification
        verification_address="server",
        eval_every=200,
        # Pipeline level
        pipeline_class="PipelineRunnerStandard",  # We eval() this to instantiate the pipeline
        end_iteration=500,
        start_iteration=0,
        parameter_sharing=False,
        num_conjecturer_warmup_rounds=0,
        # Resource configs
        training_resources_config=training_resource_config,
        gen_resources_config=gen_resource_config,
        guide_resources_config=guide_resource_config,
        verification_resources_config=verification_resource_config,
        eval_verification_resources_config=eval_verification_resource_config,
        # Evaluation
        eval_datasets=None,
        eval_best_of_n=8,
        # Wandb
        wandb_project="sgs",
        wandb_tags=wandb_tags,
        freeze_prover=False,
        statement_selection_mode=StatementSelectionMode.HARD,
    )

    # Check we can instantiate the guide
    if pipeline_config.guide_class is not None:
        eval(pipeline_config.guide_class)(pipeline_config.guide_config)

    if run_local:
        run_pipeline(pipeline_config=pipeline_config)
    else:
        executor = get_submitit_executor(example_get_master_running_config())
        executor.submit(RunPipelineCheckpointable(), pipeline_config=pipeline_config)

    return

if __name__ == "__main__":
    lrs = [3e-6]
    run_local = True
    for lr in lrs:
        checkpoint_dir = "./checkpoints/sgs_no_guide"

        # make the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        run_experiment(
            checkpoint_dir=checkpoint_dir,
            lr=lr,
            wandb_tags=[
                "no_guide",
            ],
            run_local=run_local,
        )
