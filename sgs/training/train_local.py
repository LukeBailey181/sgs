# For locally finetuning a model

from typing import List, Tuple, Optional, Dict, Any
import argparse
import socket

import submitit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import pickle
import os
import wandb
from transformers.trainer_utils import get_last_checkpoint


from sgs.training.training_types import TrainingConfig, TrainingSampleDatum
from sgs.training.custom_trainers import (
    DataCollatorWithPromptHashes,
    WeightedTrainer,
    SequentialTrainer,
    GroupedImportanceSampledWeightedTrainer,
    ImportanceSampledWeightedTrainer,
)
from sgs.models.model_types import ResourcesConfig, ModelConfig
from sgs.utils.logging_config import get_logger
from sgs.utils import export


logger = get_logger(__name__)


def sanity_check_training_args(
    training_config: TrainingConfig, resources_config: ResourcesConfig
):
    # Sanity check the resource config
    num_gpus = torch.cuda.device_count()
    if (
        training_config.batch_size % (num_gpus * training_config.batch_size_per_gpu)
        != 0
    ):
        raise ValueError(
            f"Batch size {training_config.batch_size} is not divisible by the number of GPUs {num_gpus} and batch size per GPU {training_config.batch_size_per_gpu}"
        )


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)


def prepare_torch_distributed_training():
    # If we are running a submitit job, we need to set up the distributed training environment

    print("exporting PyTorch distributed environment variables")
    dist_env = submitit.helpers.TorchDistributedEnvironment().export(
        set_cuda_visible_devices=False
    )

    print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
    print(f"rank: {dist_env.rank}")
    print(f"world size: {dist_env.world_size}")
    print(f"local rank: {dist_env.local_rank}")
    print(f"local world size: {dist_env.local_world_size}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Using the (default) env:// initialization method
    torch.distributed.init_process_group(backend="nccl")
    assert dist_env.rank == torch.distributed.get_rank()
    assert dist_env.world_size == torch.distributed.get_world_size()

    return dist_env.world_size


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True  # If not in distributed mode, assume this is the main process


def finetune_local(
    group_size: int,
    model_save_path: str,
    train_data: List[TrainingSampleDatum],
    training_config: TrainingConfig,
    trainer_cls: str,
    resources_config: ResourcesConfig,
    model_config: ModelConfig,
    val_data: Optional[List[Tuple[str, str]]] = None,
    report_to: str | List[str] = [],
) -> List[Dict[str, float | wandb.Table]]:
    """
    Finetune a local model on a dataset.

    Args:
        model_save_path: Path to save the finetuned model
        train_data: List of tuples of (prompt, target, optional(data_weight), log_probs)
        training_config: Training configuration
        resources_config: Resources configuration
        model_config: Model configuration
        val_data: Optional list of tuples of (prompt, target) for validation
        rewards: Optional list of rewards for the training data. If provided, then updates are weighted by the rewards.
            Thus if the rewards include a baseline (are advantages) this is the same as reinforce.

    Returns:
        Path to the saved model
    """
    assert len(train_data) > 0, "No training data provided"

    print(f"USING TRAINING CLASS: {trainer_cls}")

    if resources_config.submitit:
        # Log job info for debugging
        job_env = submitit.JobEnvironment()
        export()
        print(f"Starting submitit job {job_env.job_id} on {job_env.hostname}")

        world_size = prepare_torch_distributed_training()
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # Launched via torchrun; initialize the process group here
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    cur = torch.cuda.current_device()
    name = torch.cuda.get_device_name(cur)
    print(
        f"[rank={torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0} "
        f"local_rank={local_rank}] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"current_device={cur} name={name}"
    )

    sanity_check_training_args(training_config, resources_config)

    num_gpus = world_size
    gradient_accumulation_steps = training_config.batch_size // (
        num_gpus * training_config.batch_size_per_gpu
    )
    print(
        f"Using {num_gpus} GPUs with batch size {training_config.batch_size} and batch size per GPU {training_config.batch_size_per_gpu}"
    )
    print(f"Thus, gradient accumulation steps: {gradient_accumulation_steps}")

    print(f"LOADING MODEL ON RANK {local_rank}")

    resume_root = model_config.model_name
    ckpt = None
    if (not training_config.reset_trainer) and os.path.isdir(resume_root):
        ckpt = get_last_checkpoint(resume_root)  # e.g. ".../checkpoint-1234" or None

    model_path = ckpt if ckpt is not None else resume_root
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    torch_dtype = model_config.dtype
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)

    print("-" * 100)
    print(f"MODEL CONFIG DTYPE: {model_config.dtype}")
    print(f"DERIVED TORCH DTYPE: {torch_dtype}")
    print(f"USING TORCH DTYPE: {model.dtype}")
    print("-" * 100)

    print(f"MODEL LOADED ON RANK {local_rank}")

    prompts = [example.prompt_str for example in train_data]
    targets = [example.gen_str for example in train_data]
    weights = [example.reward for example in train_data]
    log_probs = [example.log_prob_over_gen for example in train_data]
    target_tokens = [example.gen_tokens for example in train_data]
    group_ids = [example.group_id for example in train_data]
    num_tokens_in_group = [example.num_tokens_in_group for example in train_data]
    advantages = [example.advantage for example in train_data]

    train_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "target": targets,
            "w": weights,
            "lp": log_probs,
            "target_tokens": target_tokens,
            "group_id": group_ids,
            "num_tokens_in_group": num_tokens_in_group,
            "advantage": advantages,
        }
    )

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": model_config.max_tokens,
            "chat_model": model_config.chat,
        },
        remove_columns=[
            "prompt",
            "target",
            "w",
            "lp",
            "target_tokens",
        ],  # Remove original columns
    )

    # ------------------ Stop for some useful logging ------------------
    n = min(20, len(tokenized_train))
    eos_id = tokenizer.eos_token_id

    table = wandb.Table(columns=["input_ids", "attention_mask", "labels", "weights"])

    for i in range(n):
        ex = tokenized_train[i]
        ids = ex["input_ids"]
        attn = ex["attention_mask"]
        labels = ex["labels"]
        weights = ex["weights"]

        ids_text = tokenizer.decode(ids, skip_special_tokens=False)
        attn_text = tokenizer.decode(
            [t for t, m in zip(ids, attn) if m == 1], skip_special_tokens=False
        )
        labels_text = tokenizer.decode(
            [t if t != -100 else eos_id for t in labels], skip_special_tokens=False
        )

        table.add_data(ids_text, attn_text, labels_text, weights)
    # --------------------------------------------------------------------

    tokenized_val: Dataset | None
    if val_data is not None:
        val_prompts = [example[0] for example in val_data]
        val_targets = [example[1] for example in val_data]
        val_dataset = Dataset.from_dict({"prompt": val_prompts, "target": val_targets})
        tokenized_val = val_dataset.map(
            preprocess_function,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": model_config.max_tokens,
            },
            remove_columns=["prompt", "target"],  # Remove original columns
        )
    else:
        tokenized_val = None

    data_collator = DataCollatorWithPromptHashes(tokenizer=tokenizer)

    # --- 4. Define Training Arguments ---
    ds_config = {
        "zero_optimization": {
            "stage": training_config.zero_stage,
        },
        "fp16": {"enabled": bool(model_config.dtype == "float16")},
        "bf16": {"enabled": bool(model_config.dtype == "bfloat16")},
        "train_batch_size": training_config.batch_size,
        "train_micro_batch_size_per_gpu": training_config.batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }

    print(f"SETTING UP TRAINING ARGUMENTS ON RANK {local_rank}...")

    if not training_config.reset_trainer:
        save_strategy = "epoch"
    else:
        save_strategy = "no"

    training_args = TrainingArguments(
        # torch_empty_cache_steps=1,
        gradient_checkpointing=True,
        output_dir=model_save_path,
        overwrite_output_dir=True,
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size_per_gpu,
        per_device_eval_batch_size=training_config.batch_size_per_gpu,
        eval_steps=training_config.eval_steps,  # Evaluation frequency
        logging_strategy="steps",
        logging_steps=training_config.log_steps,  # Log frequency
        save_strategy=save_strategy,
        # save_steps=training_config.save_steps,  # Checkpoint frequency
        save_total_limit=2,  # Keep only the last 2 checkpoints
        learning_rate=training_config.learning_rate,
        weight_decay=0.00,
        lr_scheduler_type="constant",
        warmup_steps=training_config.warmup_steps,
        fp16=model_config.dtype == "float16",
        bf16=model_config.dtype == "bfloat16",
        # report_to=report_to,
        report_to=report_to,
        load_best_model_at_end=True,  # Load the best model found during training
        metric_for_best_model="loss",  # Use validation loss to determine the best model
        greater_is_better=False,  # Lower loss is better
        deepspeed=ds_config,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        remove_unused_columns=False,  # We need to keep the weights column
        dataloader_drop_last=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
    )
    # Adding custom training arguments
    training_args.cispo_clip_lower = training_config.cispo_clip_lower
    training_args.cispo_clip_higher = training_config.cispo_clip_higher
    training_args.group_size = group_size

    assert (
        training_config.batch_size % group_size == 0
    ), "Batch size must be divisible by group size"
    groups_per_batch = training_config.batch_size // group_size

    training_args.groups_per_batch = groups_per_batch

    # --- 5. Initialize Trainer ---
    print(f"Initializing Trainer on RANK {local_rank}...")
    print(f"Initializing {trainer_cls}")

    trainer = eval(trainer_cls)(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print(f"Starting training on RANK {local_rank}...")

    is_huggingface_hub = not os.path.exists(model_config.model_name)
    if training_config.reset_trainer or is_huggingface_hub:
        trainer.train()
    else:
        ckpt = get_last_checkpoint(model_config.model_name)
        trainer.train(resume_from_checkpoint=ckpt)

    print(f"Training finished on RANK {local_rank}.")

    # --- 8. Save Final Model and Metrics ---
    print(
        f"Saving final model and tokenizer to {model_save_path} on RANK {local_rank}..."
    )

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Get training history from trainer state
    log_history: List[Dict[str, float]] = trainer.state.log_history

    log_history.append({"sample_train_data_table": table})

    # Get all of the avg_batch_entropy values
    avg_batch_entropy_values = [
        x["avg_batch_entropy"] for x in log_history if "avg_batch_entropy" in x
    ]
    if len(avg_batch_entropy_values) > 0:
        avg_batch_entropy = sum(avg_batch_entropy_values) / len(
            avg_batch_entropy_values
        )
        log_history.append(
            {"data_gen/avg_batch_entropy_during_training": avg_batch_entropy}
        )
    else:
        print("WARNING: No avg_batch_entropy values found in log history")

    # Now we will write the log history to a json file in the model save path
    # so it can be read later

    if _is_main_process():
        with open(os.path.join(model_save_path, "log_history.pkl"), "wb") as f:
            pickle.dump(log_history, f)

    return log_history


def preprocess_function(examples, tokenizer, chat_model: bool, max_length=1024):
    """
    Tokenizes the prompt and target, concatenates them, masks the prompt
    tokens in the labels, and shifts labels for causal LM.
    """

    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "weights": [],
        "advantage": [],
        "log_probs": [],
        "num_tokens_in_group": [],
        "group_id": [],
    }  # type: ignore

    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        target = examples["target"][i]

        group_id = examples["group_id"][i]
        num_tokens_in_group = examples["num_tokens_in_group"][i]
        advantage = examples["advantage"][i]
        weight = examples["w"][i]

        if chat_model:
            prompt_msgs = [{"role": "user", "content": prompt}]
            full_msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=False
            )

            full_text = tokenizer.apply_chat_template(
                full_msgs, add_generation_prompt=False, tokenize=False
            )
        else:
            prompt_text = prompt
            full_text = prompt + target

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        generated_ids = examples["target_tokens"][i]

        full_ids = prompt_ids + generated_ids

        assert tokenizer.eos_token_id is not None

        if len(full_ids) < max_length:
            if full_ids[-1] != tokenizer.eos_token_id:
                print(f"WARNING: Last token is not eos token for input {full_ids}")
                print("All IDS:")
                print(full_ids)

                raise ValueError(
                    "For inputs less than max length, last token should be eos token"
                )

        # Create attention mask
        attention_mask = [1] * len(full_ids)

        # Create labels: mask prompt tokens (-100)
        labels = [-100] * len(prompt_ids) + generated_ids

        # Note we have log probs for all the label tokens
        log_probs: List[float] = examples["lp"][i]
        log_probs = [-100] * len(prompt_ids) + log_probs

        # Sanity check
        assert len(labels) == len(log_probs)
        for label, lp in zip(labels, log_probs):
            if label != -100:
                assert lp != -100

        model_inputs["input_ids"].append(full_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
        model_inputs["weights"].append(weight)
        model_inputs["log_probs"].append(log_probs)
        model_inputs["group_id"].append(group_id)
        model_inputs["num_tokens_in_group"].append(num_tokens_in_group)
        model_inputs["advantage"].append(advantage)

    return model_inputs


if __name__ == "__main__":
    # We can call this directly then we run training on some pickled data
    # Parse args
    parser = argparse.ArgumentParser(description="Train a model locally")
    parser.add_argument(
        "--args_pickle_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # Load args from pickle
    with open(args.args_pickle_path, "rb") as f:
        args_dict: Dict[str, Any] = pickle.load(f)

    # We are calling this directly, so we make sure we are not doing submitit prep
    args_dict["resources_config"].submitit = False

    # Run training
    result = finetune_local(**args_dict)
