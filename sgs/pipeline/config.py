from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from sgs.models import (
    ProverConfig,
    ConjecturerConfig,
    ResourcesConfig,
    GuideConfig,
)
from sgs.training import TrainingConfig
from sgs.data import DatasetType
from sgs.utils import get_standard_training_config


# This config defines all the attributes of a pipeline run

class StatementSelectionMode(Enum):
    HARD = "hard"  # Select only statements and conjectures with solve rate less than 0.5
    UNSOLVED = "unsolved"  # Select only problems that were previously unsolved. This is what standard EI would do
    STP_IMPROVED_EI = "stp_improved_ei"  # In STP they do at most 16/64 proofs per problem. This is what we should do for RL 
    ALL_NONE_0_1 = "all_none_0_1"  # All conjectures and statements, which is what we should do for RL 
    ALL = "all"  # All conjectures and statements, which is what we should do for RL 
    LESS_16_PROOFS = "less_16_proofs"  # Select statements with less than 16 proofs. Select, all conjectures that have proofs

@dataclass
class PipelineConfig:
    # Budget
    budget: float = 100.0  # Our default budget is 100 USD
    # ^ We only spend this if we are using any API models

    statement_selection_mode: StatementSelectionMode = StatementSelectionMode.HARD
    save_data_extension: str = "pkl"    # Or "json"

    restart_from_most_recent_checkpoint: bool = True
    eval_init_model: bool = False
    num_master_verification_workers: int = 0

    pipeline_proving_and_verification: bool = False

    # Data
    prover_dataset_path: str = "data/prover_dataset.json"
    conjecturer_dataset_path: str = "data/conjecturer_dataset.json"

    # Models
    checkpoint_dir: str = "./checkpoints/temp"

    # Training
    training_config: TrainingConfig = field(
        default_factory=get_standard_training_config
    )
    parameter_sharing: bool = True
    num_conjecturer_warmup_rounds: int = 0

    # Prover
    prover_iterations_in_buffer: int = 1
    num_train_examples: Optional[int] = None
    conjecture_multiplier: Optional[int] = None
    prover_model_config: Optional[ProverConfig] = None
    proofs_per_sample: int = 16
    subsample_target_statements: Optional[int] = None
    batch_target_statements: Optional[int] = None
    retrain_prover_from_scratch: bool = False
    freeze_prover: bool = False

    # Conjecturer
    conjecturer_iterations_in_buffer: int = 1  # We normally do this online
    conjectures_per_statement: int | Dict[DatasetType, int] = 1
    conjecturer_model_config: Optional[ConjecturerConfig] = None
    retrain_conjecturer_from_scratch: bool = False
    freeze_conjecturer: bool = False

    # Guide
    guide_class: Optional[str] = None  # We eval() this to instantiate the guide
    guide_config: Optional[GuideConfig] = None
    update_guide: bool = True

    # Verification
    verification_address: str = "server"
    verifier_timeout: int = 500

    # Pipeline level
    pipeline_class: str = (
        "PipelineRunnerStandard"  # We eval() this to instantiate the pipeline
    )
    end_iteration: int = 5
    start_iteration: int = 0

    # Resource configs
    training_resources_config: Optional[ResourcesConfig] = None
    gen_resources_config: Optional[ResourcesConfig] = None
    eval_gen_resources_config: Optional[ResourcesConfig] = None
    guide_resources_config: Optional[ResourcesConfig] = None
    verification_resources_config: Optional[ResourcesConfig] = None
    eval_verification_resources_config: Optional[ResourcesConfig] = None

    # Evaluation
    eval_datasets: Optional[List[DatasetType]] = None
    eval_best_of_n: int = 16
    eval_every: int = 1
    eval_in_seperate_job: bool = True

    # Wandb
    wandb_project: Optional[str] = "sgs"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: [])
    wandb_resume_id: Optional[bool] = True
    num_generations_at_start: Optional[int] = None



    def to_dict(self):
        output = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                temp_dict = {}
                for k2, v2 in v.items():
                    if isinstance(k2, DatasetType):
                        temp_dict[k2.value] = v2
                    else:
                        temp_dict[k2] = v2
                output[k] = temp_dict
            else:
                output[k] = v

        return output


    def __post_init__(self):

        if self.training_config.prover_trainer_cls in ["GroupedImportanceSampledWeightedTrainer"]:
            assert self.training_config.prover_group_size == self.proofs_per_sample, "Prover group size must be equal to proofs per sample"

        if self.training_config.conjecturer_trainer_cls in ["GroupedImportanceSampledWeightedTrainer"]:
            raise ValueError("Training conjecturer with group advantage estimation is not supported yet")

        if not self.update_guide:
            raise ValueError("Update guide must be True. We currently do not support keeping the guide fixed.")