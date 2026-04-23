from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional, List


class ModelType(Enum):
    LOCAL = "local"


class ConjecturerSetup(Enum):
    SEED_STATEMENT = "seed_statement"
    TARGET_STATEMENT = "target_statement"
    TARGET_STATEMENT_ONLY_UNSOLVED = "target_statement_only_unsolved"


@dataclass
class ResourcesConfig:
    submitit: bool
    log_dir: str
    # Slurm
    account: str = "nlp"
    partition: str = "sphinx"
    gres: str = "gpu:0"
    mem: str = "128G"
    time: str = "12:00:00"
    cpus_per_task: int = 8
    exclude: str = ""
    constraints: Optional[str] = None
    # Parallelism
    num_jobs: int = 1  # If job can be split into parallel jobs on our side
    jobs_per_node: int = 1  # If job can be split into parallel jobs on the cluster side
    # ^ Required for distributed training, set this to number of gpus per node
    examples_per_job: int = 1000  # Sometimes we will have the client maintain a pool of jobs with a fixed number of examples per job
    node_list: Optional[str] = None
    exclusive: bool = False


@dataclass
class ModelConfig:
    model_name: str
    prompt_getter: Callable[..., str]
    output_extractor: Callable[..., str]
    model_type: ModelType
    dtype: str = "bfloat16"
    temperature: float = 1.0
    max_tokens: int = 8000
    chat: bool = False
    system_prompt: str = "You are a helpful assistant."
    top_p: float = 1.0
    top_k: int = 0
    # Batching for server-based generation. Lives on the model config because
    # different models have different CoT length
    gen_batch_size: int = 128
    use_system_prompt: bool = False
    use_torch_compile: bool = True
    # Reward shaping
    penalize_double_lean: bool = False
    penalize_try: bool = True
    # At most one of these length penalties may be active (enforced below).
    dapo_length_penalty: bool = True
    stp_length_penalty: bool = False
    penalize_long_proof_str_over_1000: bool = False

    assert_theorem_in_proof: bool = False

    def __post_init__(self):
        num_length_penalties = sum(
            [
                self.penalize_long_proof_str_over_1000,
                self.dapo_length_penalty,
                self.stp_length_penalty,
            ]
        )
        assert (
            num_length_penalties <= 1
        ), "Only one length penalty can be used at a time"


@dataclass
class ProverConfig(ModelConfig):
    pass


@dataclass
class ConjecturerConfig(ModelConfig):
    setup: ConjecturerSetup = ConjecturerSetup.SEED_STATEMENT
    use_ice: bool = False


@dataclass
class QueryResult:
    response_text: str
    input_token_count: int
    output_token_count: int
    is_error: bool
    cost: float = 0.0  # We default to 0 as we normally run locally
    average_entropy: Optional[float] = None
    # Log probs of the sampled tokens
    log_probs: Optional[List[float]] = None
    output_tokens: Optional[List[int]] = None
    prompt_tokens: Optional[List[int]] = None
