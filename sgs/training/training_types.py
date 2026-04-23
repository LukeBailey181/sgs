from dataclasses import dataclass
from typing import List

@dataclass
class TrainingConfig:
    # Batch size / distributed training details
    batch_size: int = 16
    batch_size_per_gpu: int = 1
    zero_stage: int = 2

    # Logging and saving
    eval_steps: int = 100
    log_steps: int = 10
    save_steps: int | None = None

    # Training hyperparameters
    epochs: int = 1
    learning_rate: float = 0.0001

    prover_learning_rate: float = None
    conjecturer_learning_rate: float = None

    warmup_steps: int = 20
    max_seq_length: int = 8000

    # RL
    cispo_clip_lower: float = 0.0
    cispo_clip_higher: float = 4.0

    # Training class
    prover_trainer_cls: str = "WeightedTrainer"
    conjecturer_trainer_cls: str = "WeightedTrainer"

    # Group size, this is only used for some training classes
    prover_group_size: int = 1
    conjecturer_group_size: int = 1

    reset_trainer: bool = True

    def __post_init__(self):

        if self.prover_trainer_cls in ["GroupedImportanceSampledWeightedTrainer"]:
            self.loss_fn_config = {
                "clip_low_threshold": self.cispo_clip_lower,
                "clip_high_threshold": self.cispo_clip_higher,
            }
        else:
            self.loss_fn_config = None

        #if self.prover_learning_rate is None:
        #    self.prover_learning_rate = self.learning_rate
        
        #if self.conjecturer_learning_rate is None:
        #    self.conjecturer_learning_rate = self.learning_rate


@dataclass
class TrainingSampleDatum:
    prompt_str: str
    gen_str: str
    reward: float
    advantage: float         # Only needed when doing advantage estimation using groups
    log_prob_over_gen: List[float]
    gen_tokens: List[int]
    group_id: str
    num_tokens_in_group: int    # Only needed when doing advantage estimation using groups