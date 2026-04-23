from sgs.models.model_types import ResourcesConfig
from sgs.training.training_types import TrainingConfig
from sgs.utils.prompts import (
    extract_proof_deepseek_v2_strict,
    extract_conjecture_deepseek_v2,
    get_deepseek_prover_v2_prompt,
    get_deepseek_prover_v2_conjecturer_prompt,
)
from sgs.models.model_types import (
    ProverConfig,
    ConjecturerConfig,
    ModelType,
    ConjecturerSetup,
)


# ------------------------------------------------------------
# Resource configs
# ------------------------------------------------------------

# The `example_*` getters below are TEMPLATES for how a user might configure
# slurm/submitit-managed training, generation, and verification on their own
# cluster. They contain `<YOUR_SLURM_*>` placeholders that must be filled in
# before they will work. Copy + edit them for your environment, or use
# `get_local_running_config()` to run everything  on a single node.


def example_get_verification_resources_config() -> ResourcesConfig:
    """Template for a CPU-heavy submitit pool used to verify Lean proofs.

    Many parallel jobs, each allocated lots of CPUs and no GPU. Replace the
    `<YOUR_SLURM_*>` placeholders with values appropriate to your cluster.
    """
    resources_config = ResourcesConfig(
        submitit=True,
        log_dir="./tests/test_logs",
        # Slurm
        account="<YOUR_SLURM_ACCOUNT>",
        partition="<YOUR_SLURM_PARTITION_WITH_LOTS_OF_CPUS>",
        mem="128G",
        time="1-00:00:00",
        cpus_per_task=33,
        # Parallelism
        num_jobs=128,
        jobs_per_node=1,
        examples_per_job=100,
    )

    return resources_config


def example_get_generation_resources_config() -> ResourcesConfig:
    """Template for a GPU pool used to sample proofs and conjectures.

    Many parallel jobs, each allocated 1 GPU for a batched inference worker.
    Replace the `<YOUR_SLURM_*>` placeholders with values appropriate to your
    cluster.
    """
    resources_config = ResourcesConfig(
        submitit=True,
        log_dir="./tests/test_logs",
        # Slurm
        account="<YOUR_SLURM_ACCOUNT>",
        partition="<YOUR_SLURM_PARTITION_WITH_GPUS>",
        gres="gpu:1",
        mem="64G",
        time="1:00:00",
        cpus_per_task=16,
        # Parallelism
        num_jobs=64,
        jobs_per_node=1,
    )

    return resources_config


def example_get_training_resource_config() -> ResourcesConfig:
    """Template for a multi-GPU submitit training job.

    One job with multiple GPUs on a single node for distributed training.
    Replace the `<YOUR_SLURM_*>` placeholders with values appropriate to your
    cluster.
    """
    resources_config = ResourcesConfig(
        submitit=True,
        log_dir="./tests/test_logs",
        # Slurm
        account="<YOUR_SLURM_ACCOUNT>",
        partition="<YOUR_SLURM_PARTITION_WITH_H200s>",
        gres="gpu:8",
        mem="2048G",
        time="5:00:00",
        cpus_per_task=16,
        # Parallelism
        num_jobs=1,
        jobs_per_node=8,  # number of GPUs on the node
    )

    return resources_config


def example_get_master_running_config() -> ResourcesConfig:
    """Template for the outer orchestrator job that runs the pipeline loop.

    A single long-running job with modest resources that coordinates the
    training / generation / verification sub-jobs. Replace the
    `<YOUR_SLURM_*>` placeholders with values appropriate to your cluster.
    """
    resources_config = ResourcesConfig(
        submitit=True,
        log_dir="./tests/test_logs",
        # Slurm
        account="<YOUR_SLURM_ACCOUNT>",
        partition="<YOUR_SLURM_PARTITION>",
        mem="32G",
        time="3-00:00:00",
        cpus_per_task=4,
        # Parallelism
        num_jobs=1,
        jobs_per_node=1,
    )

    return resources_config


def get_local_running_config() -> ResourcesConfig:
    """Resource config for running everything in-process on the local machine.

    No slurm, no submitit. Generation uses all visible GPUs, training auto-
    detects GPU count via `torch.cuda.device_count()`, and verification runs
    via local master workers (set `num_master_verification_workers > 0` in
    the pipeline config).
    """
    resources_config = ResourcesConfig(
        partition="",
        submitit=False,
        log_dir="./tests/test_logs/",
        num_jobs=0,
    )

    return resources_config


# ------------------------------------------------------------
# Model configs
# ------------------------------------------------------------


def get_deepseek_prover_v2_prover_config() -> ProverConfig:
    prover_config = ProverConfig(
        prompt_getter=get_deepseek_prover_v2_prompt,
        output_extractor=extract_proof_deepseek_v2_strict,
        max_tokens=8192,
        model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
        dtype="bfloat16",
        chat=True,
        model_type=ModelType.LOCAL,
        # Reward shaping
        penalize_double_lean=True,
        penalize_long_proof_str_over_1000=True,
        stp_length_penalty=False,
        penalize_try=True,
        dapo_length_penalty=False,
    )
    return prover_config


def get_deepseek_prover_v2_conjecturer_config(
    mode: ConjecturerSetup,
) -> ConjecturerConfig:
    conjecturer_config = ConjecturerConfig(
        model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
        model_type=ModelType.LOCAL,
        prompt_getter=get_deepseek_prover_v2_conjecturer_prompt,
        output_extractor=extract_conjecture_deepseek_v2,
        max_tokens=8192,
        dtype="bfloat16",
        setup=mode,
        chat=True,
    )
    return conjecturer_config


# ------------------------------------------------------------
# Training configs
# ------------------------------------------------------------


def get_standard_training_config() -> TrainingConfig:
    training_config = TrainingConfig(
        batch_size=512,
        batch_size_per_gpu=1,
        zero_stage=2,
        eval_steps=100,
        log_steps=1,
        save_steps=None,
        epochs=1,
        learning_rate=3e-6,
        max_seq_length=8000,
        warmup_steps=2,
    )

    return training_config
