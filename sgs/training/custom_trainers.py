# For locally finetuning a model

from typing import List, Optional, Dict, Any, Union
from einops import rearrange
import hashlib

from transformers import (
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback,
)
import torch
from torch.utils.data import SequentialSampler
import torch.nn as nn
from jaxtyping import Float, Int64
import torch.nn.functional as F
import torch.distributed as dist


class DataCollatorWithPromptHashes(DataCollatorForTokenClassification):
    def torch_call(self, features):
        group_ids = [f.pop("group_id") for f in features]
        log_probs = [f.pop("log_probs") for f in features]

        batch = super().torch_call(features)
        batch["group_id"] = group_ids  # list[str], unchanged

        # Left pad the log probs
        batch_size, seq_len = batch["labels"].shape
        log_prob: List[Float]
        new_log_probs = []
        for log_prob in log_probs:
            amount_to_pad = seq_len - len(log_prob)
            assert amount_to_pad >= 0, "Amount to pad must be non-negative for log probs"
            new_log_prob = [-100] * amount_to_pad + log_prob
            new_log_probs.append(new_log_prob)

        log_probs_tensor = torch.tensor(new_log_probs)
        log_probs_tensor = log_probs_tensor.to(batch["labels"].device)

        batch["log_probs"] = log_probs_tensor

        # Check that where labels and log probs are -100 are equal
        assert ((batch["labels"] == -100) == (batch["log_probs"] == -100)).all()

        return batch


class SequentialTrainer(Trainer):
    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset

        return SequentialSampler(train_dataset)

class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor | int] = None,
    ):
        labels: Int64[torch.Tensor, "batch_size seq_len"] = inputs.pop("labels")
        weights: Float[torch.Tensor, "batch_size"] | None = inputs.pop("weights", None)
        _ = inputs.pop("log_probs")
        _ = inputs.pop("group_id")
        _ = inputs.pop("num_tokens_in_group")
        _ = inputs.pop("advantage")


        assert weights is not None

        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        # Forward pass
        outputs = model(**inputs)

        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = (
            outputs.logits.float()
        )

        # Shift for causal LM
        shift_logits = logits[
            ..., :-1, :
        ].contiguous()  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]

        batch_size, seq_len, vocab_size = shift_logits.shape

        # Flatten for loss computation but keep batch structure
        shift_logits_flat = shift_logits.view(
            -1, vocab_size
        )  # [batch_size * seq_len, vocab_size]
        shift_labels_flat = shift_labels.view(-1)  # [batch_size * seq_len]

        shift_labels_flat = shift_labels_flat.to(shift_logits.device)

        # Compute per-token loss (no reduction)
        per_token_loss = F.cross_entropy(
            shift_logits_flat, shift_labels_flat, reduction="none", ignore_index=-100
        )

        # Reshape back to batch structure
        per_token_loss = per_token_loss.view(
            batch_size, seq_len
        )  # [batch_size, seq_len]

        sequence_losses = per_token_loss.sum(dim=1)  # [batch_size]

        # Apply batch-level weights
        if weights is not None:
            weighted_losses = sequence_losses * weights  # [batch_size]
        else:
            weighted_losses = sequence_losses

        loss = weighted_losses.sum() / num_items_in_batch

        # Taken from base class compute_loss function
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


class ImportanceSampledWeightedTrainer(Trainer):
    def compute_loss(  # Fixed typo
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor | int] = None,
    ):
        labels: Int64[torch.Tensor, "batch_size seq_len"] = inputs.pop("labels")
        weights: Float[torch.Tensor, "batch_size"] | None = inputs.pop("weights", None)
        gen_log_probs: Float[torch.Tensor, "batch_size seq_len"] = inputs.pop(
            "log_probs"
        )
        _ = inputs.pop("group_id")
        _ = inputs.pop("num_tokens_in_group")
        _ = inputs.pop("advantage")

        # Sanity check on log probs
        assert isinstance(gen_log_probs, torch.Tensor)
        batch_size, seq_len = gen_log_probs.shape
        assert gen_log_probs.ndim == 2
        assert gen_log_probs.shape[0] == batch_size
        assert gen_log_probs.shape[1] == seq_len

        assert weights is not None

        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        # Forward pass
        outputs = model(**inputs)

        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = (
            outputs.logits.float()
        )

        # Shift for causal LM
        shift_logits = logits[
            ..., :-1, :
        ].contiguous()  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]

        shift_gen_log_probs = gen_log_probs[
            ..., 1:
        ].contiguous()  # [batch_size, seq_len-1]

        batch_size, seq_len, vocab_size = shift_logits.shape

        # Flatten for loss computation but keep batch structure
        shift_logits_flat = shift_logits.view(
            -1, vocab_size
        )  # [batch_size * seq_len, vocab_size]
        shift_labels_flat = shift_labels.view(-1)  # [batch_size * seq_len]
        shift_gen_log_probs_flat = shift_gen_log_probs.view(
            -1
        )  # [batch_size * seq_len]

        shift_labels_flat = shift_labels_flat.to(shift_logits.device)
        shift_gen_log_probs_flat = shift_gen_log_probs_flat.to(shift_logits.device)

        # Compute per-token loss (no reduction)
        per_token_loss = F.cross_entropy(
            shift_logits_flat, shift_labels_flat, reduction="none", ignore_index=-100
        )

        # Now we have to compute the importance weights, first we get the log prob of the logits
        valid_mask = shift_labels_flat != -100

        # Compute log weights only on valid positions
        log_importance_weights = torch.zeros_like(per_token_loss)
        log_importance_weights[valid_mask] = (
            -per_token_loss.detach()[valid_mask] - shift_gen_log_probs_flat[valid_mask]
        )

        cispo_clip_lower = self.args.cispo_clip_lower
        cispo_clip_higher = self.args.cispo_clip_higher
        assert cispo_clip_lower is not None
        assert cispo_clip_higher is not None

        log_importance_weights[valid_mask] = log_importance_weights[valid_mask].clamp(
            min=torch.log(
                torch.tensor(cispo_clip_lower, device=log_importance_weights.device)
            ),
            max=torch.log(
                torch.tensor(cispo_clip_higher, device=log_importance_weights.device)
            ),
        )

        importance_weights = torch.ones_like(per_token_loss)
        importance_weights[valid_mask] = torch.exp(log_importance_weights[valid_mask])

        # Reshape back to batch structure
        per_token_loss = per_token_loss.view(
            batch_size, seq_len
        )  # [batch_size, seq_len]
        importance_weights = importance_weights.view(
            batch_size, seq_len
        )  # [batch_size, seq_len]

        num_clip_higher = (
            (
                log_importance_weights[valid_mask]
                > torch.log(
                    torch.tensor(
                        cispo_clip_higher, device=log_importance_weights.device
                    )
                )
            )
            .sum()
            .item()
        )
        num_train_tokens = valid_mask.sum().item()
        average_importance_weights = importance_weights.reshape(-1)[valid_mask].mean().item()
        self.log({"average_importance_weights": average_importance_weights})
        self.log({"num_clip_higher": num_clip_higher})
        self.log({"num_train_tokens_in_batch": num_train_tokens})
        self.log({"percent_train_tokens_clipped": num_clip_higher / num_train_tokens})

        # Apply the importance weights to the losses
        per_token_loss = per_token_loss * importance_weights

        sequence_losses = per_token_loss.sum(dim=1)  # [batch_size]

        # Apply batch-level weights
        if weights is not None:
            weighted_losses = sequence_losses * weights  # [batch_size]
        else:
            weighted_losses = sequence_losses

        loss = weighted_losses.sum() / num_items_in_batch

        # Taken from base class compute_loss function
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def _gid_to_i64(gid: str) -> int:
    # stable 64-bit hash
    h = hashlib.sha1(gid.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, byteorder="little", signed=False) % (2**63)

def global_group_check(
    step_group_ids: List[str], 
    group_size: int, 
    device: torch.device
) -> Optional[Dict[str, int]]:
    """Runtime check that each optimizer step contains *complete groups* globally.

    This is written to be safe even when `dataloader_drop_last=False`, i.e. ranks may
    see different numbers of examples in the final optimizer step of an epoch.

    Args:
        step_group_ids: Group IDs seen ON THIS RANK for one optimizer step.
            Length = B_rank for this step (can vary across ranks if drop_last=False).
        group_size: G, the number of examples that should share the same group_id.
        device: CUDA device to place tiny communication tensors on.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if len(step_group_ids) == 0:
        print(f"No group IDs seen on rank {rank}")

    # Local (per-rank) hashed group IDs for this optimizer step.
    # Shape: [n_local]
    ids_i64: Int64[torch.Tensor, "n_local"] = torch.tensor(
        [_gid_to_i64(x) for x in step_group_ids], dtype=torch.long, device=device
    )

    # all_gather variable-length safely by padding
    # Each rank reports how many examples it saw in this optimizer step.
    # Shape: [1]
    n_local: Int64[torch.Tensor, "1"] = torch.tensor(
        [ids_i64.numel()], device=device, dtype=torch.long
    )
    # Will hold each rank's length (still shape [1] per entry).
    ns: List[Int64[torch.Tensor, "1"]] = [torch.zeros_like(n_local) for _ in range(world_size)]
    dist.all_gather(ns, n_local)
    # Stack lengths into one tensor on CPU for convenience.
    # Shape: [world_size]
    ns_stacked: Int64[torch.Tensor, "world_size"] = torch.stack(ns).cpu()
    # Max number of examples seen across all ranks
    max_n: int = int(ns_stacked.max().item())

    assert max_n > 0, f"Max number of examples seen across all ranks is 0"

    # Pad local IDs out to the maximum length across ranks so all_gather is legal.
    # Shape: [max_n]
    padded: Int64[torch.Tensor, "max_n"] = torch.full(
        (max_n,), -1, device=device, dtype=torch.long
    )
    padded[: ids_i64.numel()] = ids_i64

    # Gather padded payloads from all ranks.
    # `gathered[i]` has shape [max_n] and contains rank i's ids followed by -1 padding.
    gathered: List[Int64[torch.Tensor, "max_n"]] = [
        torch.empty_like(padded) for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded)

    if rank == 0:
        # Concatenate then drop padding.
        # Shape before filtering: [world_size * max_n]
        all_ids_padded: Int64[torch.Tensor, "world_size_times_max_n"] = torch.cat(
            gathered, dim=0
        ).cpu()
        # Shape after filtering: [n_global]
        all_ids: Int64[torch.Tensor, "n_global"] = all_ids_padded[all_ids_padded != -1]

        # Total number of (non-padding) examples across all ranks for this optimizer step.
        B_global: int = int(ns_stacked.sum().item())
        assert B_global % group_size == 0, f"Global samples {B_global} not divisible by G={group_size}"

        # For correctness we require that every unique group_id appears exactly G times globally.
        # uniq:   [n_unique]
        # counts: [n_unique]
        uniq: Int64[torch.Tensor, "n_unique"]
        counts: Int64[torch.Tensor, "n_unique"]
        uniq, counts = torch.unique(all_ids, return_counts=True)
        num_groups = uniq.numel()
        assert (counts == group_size).all(), (
            f"Some groups incomplete or duplicated. "
            f"Example counts: {counts[:10].tolist()}"
        )
        print(f"Passed global group check")
        print(f"num_ids_collected_across_ranks: {B_global}")
        print(f"num_groups_in_global_batch: {num_groups}")
        print(f"world_size: {world_size}")
        

        # Now return useful information
        return {
            "group_check/num_ids_collected_across_ranks": B_global,
            "group_check/num_groups_in_global_batch": num_groups,
            "group_check/world_size": world_size,
        }


class _GlobalGroupCheckCallback(TrainerCallback):
    """Runs `global_group_check` once per *optimizer step*.

    The trainer accumulates `group_id`s across microbatches; this callback fires
    after the optimizer step (`on_step_end`) and validates the just-finished step.
    """

    def __init__(self, trainer: "GroupedImportanceSampledWeightedTrainer"):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        step_group_ids = getattr(self.trainer, "_step_group_ids", None)

        
        if not step_group_ids:
            #return control
            step_group_ids = []

        # Choose a device for the tiny all_gather tensors.
        model = kwargs.get("model", None) or getattr(self.trainer, "model", None)
        device = next(model.parameters()).device

        log_dict = global_group_check(step_group_ids=step_group_ids, group_size=args.group_size, device=device)
        if log_dict is not None:
            if self.trainer.state.epoch is not None:
                log_dict["epoch"] = self.trainer.state.epoch
            if self.trainer.args.include_num_input_tokens_seen:
                log_dict["num_input_tokens_seen"] = self.trainer.state.num_input_tokens_seen

            output = {**log_dict, **{"step": self.trainer.state.global_step}}
            self.trainer.state.log_history.append(output)

        self.trainer._step_group_ids = []
        return control

class GroupedImportanceSampledWeightedTrainer(SequentialTrainer):
    """
    Expects groups, and will calculate advantages for the groups
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Group IDs seen on this rank in the current optimizer step.
        self._step_group_ids: List[str] = []
        self.add_callback(_GlobalGroupCheckCallback(self))


    def compute_loss(  # Fixed typo
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor | int] = None,
    ):
        labels: Int64[torch.Tensor, "batch_size seq_len"] = inputs.pop("labels")
        weights: Float[torch.Tensor, "batch_size"] | None = inputs.pop("weights", None)
        gen_log_probs: Float[torch.Tensor, "batch_size seq_len"] = inputs.pop(
            "log_probs"
        )
        # From `DataCollatorWithPromptHashes`: list[str]
        group_ids: List[str] = inputs.pop("group_id")
        num_tokens_in_group: Float[torch.Tensor, "batch_size"] = inputs.pop("num_tokens_in_group")
        advantages: Float[torch.Tensor, "batch_size"] = inputs.pop("advantage")

        group_size = self.args.group_size
        local_batch_size, seq_len = labels.shape

        # Accumulate IDs for the once-per-optimizer-step global grouping check.
        if model.training:
            self._step_group_ids.extend(group_ids)

        # Sanity checks compatible with microbatching + HF gradient accumulation.
        #
        # We intentionally do NOT require `local_batch_size == group_size` anymore.
        # Instead, we rely on `global_group_check` (run once per optimizer step in
        # `_GlobalGroupCheckCallback`) to ensure that, globally, each optimizer step
        # contains complete groups (each group_id appears exactly `group_size` times).
        assert group_size > 0, "group_size must be positive"
        assert len(group_ids) == local_batch_size, (
            "group_ids must align with the local batch size "
            f"(got len(group_ids)={len(group_ids)} vs local_batch_size={local_batch_size})"
        )
        assert all(isinstance(g, str) for g in group_ids), "group_ids must be a list[str]"

        # Now check that within each group, the i Sanity check on log probs
        assert isinstance(gen_log_probs, torch.Tensor)
        batch_size, seq_len = gen_log_probs.shape
        assert gen_log_probs.ndim == 2
        assert gen_log_probs.shape[0] == batch_size
        assert gen_log_probs.shape[1] == seq_len

        assert weights is not None
        assert advantages is not None

        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        # Forward pass
        outputs = model(**inputs)

        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = (
            outputs.logits.float()
        )

        # Shift for causal LM
        shift_logits = logits[
            ..., :-1, :
        ].contiguous()  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]

        shift_gen_log_probs = gen_log_probs[
            ..., 1:
        ].contiguous()  # [batch_size, seq_len-1]

        batch_size, seq_len, vocab_size = shift_logits.shape

        # Flatten for loss computation but keep batch structure
        shift_logits_flat = shift_logits.view(
            -1, vocab_size
        )  # [batch_size * seq_len, vocab_size]
        shift_labels_flat = shift_labels.view(-1)  # [batch_size * seq_len]
        shift_gen_log_probs_flat = shift_gen_log_probs.view(
            -1
        )  # [batch_size * seq_len]

        shift_labels_flat = shift_labels_flat.to(shift_logits.device)
        shift_gen_log_probs_flat = shift_gen_log_probs_flat.to(shift_logits.device)

        # Compute per-token loss (no reduction)
        per_token_loss = F.cross_entropy(
            shift_logits_flat, shift_labels_flat, reduction="none", ignore_index=-100
        )

        # Now we have to compute the importance weights, first we get the log prob of the logits
        valid_mask = shift_labels_flat != -100

        # Compute log weights only on valid positions
        log_importance_weights = torch.zeros_like(per_token_loss)
        log_importance_weights[valid_mask] = (
            -per_token_loss.detach()[valid_mask] - shift_gen_log_probs_flat[valid_mask]
        )

        cispo_clip_lower = self.args.cispo_clip_lower
        cispo_clip_higher = self.args.cispo_clip_higher
        assert cispo_clip_lower is not None
        assert cispo_clip_higher is not None


        # Collect the number that should be clipped
        num_clip_higher = (
            (
                log_importance_weights[valid_mask]
                > torch.log(
                    torch.tensor(
                        cispo_clip_higher, device=log_importance_weights.device
                    )
                )
            )
            .sum()
            .item()
        )
        num_clip_lower = (
            (
                log_importance_weights[valid_mask]
                < torch.log(
                    torch.tensor(cispo_clip_lower, device=log_importance_weights.device)
                )
            )
            .sum()
            .item()
        )

        num_train_tokens = valid_mask.sum().item()
        self.log({"num_clip_higher": num_clip_higher})
        self.log({"num_clip_lower": num_clip_lower})
        self.log({"num_train_tokens_in_batch": num_train_tokens})
        self.log({"percent_train_tokens_clipped_upper": num_clip_higher / num_train_tokens})
        self.log({"percent_train_tokens_clipped_lower": num_clip_lower / num_train_tokens})
        self.log({"percent_train_tokens_clipped": (num_clip_higher + num_clip_lower) / num_train_tokens})

        # NOW DO THE CLIPPING
        log_importance_weights[valid_mask] = log_importance_weights[valid_mask].clamp(
            min=torch.log(
                torch.tensor(cispo_clip_lower, device=log_importance_weights.device)
            ),
            max=torch.log(
                torch.tensor(cispo_clip_higher, device=log_importance_weights.device)
            ),
        )

        importance_weights = torch.ones_like(per_token_loss)
        importance_weights[valid_mask] = torch.exp(log_importance_weights[valid_mask])

        average_importance_weights = importance_weights.reshape(-1)[valid_mask].mean().item()
        self.log({"average_importance_weights_post_clipping": average_importance_weights})

        # Reshape back to batch structure
        per_token_loss = per_token_loss.view(
            batch_size, seq_len
        )  # [batch_size, seq_len]
        importance_weights = importance_weights.view(
            batch_size, seq_len
        )  # [batch_size, seq_len]


        # Apply the importance weights to the losses
        per_token_loss = per_token_loss * importance_weights

        sequence_losses = per_token_loss.sum(dim=1)  # [batch_size]

        # Apply batch-level weights
        weighted_losses = sequence_losses * advantages  # [batch_size]

        # This is what CISPO prescribes
        # Alternatively, scale RL would do (weighted_losses.sum() / num_items_in_batch)
        loss = (weighted_losses / num_tokens_in_group).sum() / self.args.groups_per_batch

        # Taken from base class compute_loss function
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

