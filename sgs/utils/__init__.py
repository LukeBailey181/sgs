import os
import shutil
import tempfile
from typing import List, TypeVar, Tuple, Optional
from pathlib import Path
from datetime import datetime
import subprocess
import shlex
from dotenv import load_dotenv

import submitit
from submitit.core.core import Job


from sgs.models.model_types import ResourcesConfig
from sgs.utils.experiment_utils import (
    example_get_verification_resources_config,
    example_get_generation_resources_config,
    example_get_training_resource_config,
    example_get_master_running_config,
    get_local_running_config,
    get_standard_training_config,
    get_deepseek_prover_v2_prover_config,
)

__all__ = [
    "example_get_verification_resources_config",
    "example_get_generation_resources_config",
    "example_get_training_resource_config",
    "example_get_master_running_config",
    "get_local_running_config",
    "get_standard_training_config",
    "get_deepseek_prover_v2_prover_config",
]


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

T = TypeVar("T")


def time_string_to_minutes(time_str: str) -> int:
    """
    Convert time string in format 'DD-HH:MM:SS' or 'HH:MM:SS' to minutes.

    Args:
        time_str: Time string in format 'DD-HH:MM:SS' or 'HH:MM:SS'
    Returns:
        Total minutes as integer
    """
    if not time_str:
        return 0

    # Handle DD-HH:MM:SS format
    if "-" in time_str:
        days_part, time_part = time_str.split("-", 1)
        days = int(days_part)
    else:
        days = 0
        time_part = time_str

    # Parse HH:MM:SS
    time_components = time_part.split(":")
    if len(time_components) == 3:
        hours, minutes, _ = map(int, time_components)
    elif len(time_components) == 2:
        hours, minutes = map(int, time_components)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    # Convert to total minutes
    total_minutes = days * 24 * 60 + hours * 60 + minutes

    return total_minutes


def chunk_list(lst: List[T], n_chunks: int) -> List[List[T]]:
    """Split list into exactly n_chunks chunks, distributing items as evenly as possible."""

    if len(lst) < n_chunks:
        # If the list is smaller than the number of chunks, return a list of lists with one item each
        return [[x] for x in lst]

    chunk_size = len(lst) // n_chunks
    remainder = len(lst) % n_chunks

    chunks = []
    start = 0
    for i in range(n_chunks):
        # Add one extra item to the first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start : start + current_chunk_size])
        start += current_chunk_size

    # Check that the chunks are correct
    assert sum([len(chunk) for chunk in chunks]) == len(lst)

    return chunks


def get_submitit_executor(
    resources_config: ResourcesConfig,
    log_dir: Optional[str] = None,
) -> submitit.AutoExecutor:
    # convert time to minutes as submitit premting doesn't support time strings
    time = time_string_to_minutes(resources_config.time)

    if log_dir is None:
        log_dir = resources_config.log_dir

    for k in list(os.environ):
        if k.startswith("SLURM_"):
            if "SLURM_JOB_ID" in k:
                pass
            else:
                os.environ.pop(k, None)

    os.environ["SLURM_CPU_BIND"] = "none"  # set a safe default for the new job

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=resources_config.account,
        slurm_partition=resources_config.partition,
        slurm_gres=resources_config.gres,
        slurm_mem=resources_config.mem,
        slurm_time=time,
        slurm_cpus_per_task=resources_config.cpus_per_task,
        slurm_exclude=resources_config.exclude,
        slurm_ntasks_per_node=resources_config.jobs_per_node,
        slurm_constraint=resources_config.constraints,
        slurm_additional_parameters={
            "export": "ALL,SLURM_CPU_BIND=none",
        },
    )

    if resources_config.node_list is not None:
        executor.update_parameters(
            slurm_nodelist=resources_config.node_list,
        )

    if resources_config.exclusive:
        executor.update_parameters(
            slurm_exclusive=True,
        )

    return executor


class SubmititCleanupExecutor:
    def __init__(
        self,
        resources_config: ResourcesConfig,
        keep_on_success: bool = False,
    ):
        self.rc = resources_config
        self.base = resources_config.log_dir
        self.keep_on_success = keep_on_success

        self.folder: Path | None = None
        self.executor: submitit.AutoExecutor | None = None

    def __enter__(self):
        # Create a unique temp directory under the chosen base
        tmp_path = tempfile.mkdtemp(prefix="submitit_", dir=self.base)
        self.folder = Path(tmp_path)

        self.executor = get_submitit_executor(self.rc, str(self.folder))

        return self.executor

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception happened, keep the folder (so logs remain)
        if exc_type is not None:
            return False  # re-raise the exception after __exit__

        # On success, delete the folder unless user asked to keep it
        if not self.keep_on_success and self.folder and self.folder.exists():
            shutil.rmtree(self.folder, ignore_errors=True)

        return False


def cleanup_submitit_job(job: Job):
    path: Path
    for path in [
        job.paths.stderr,
        job.paths.stdout,
        job.paths.submission_file,
        job.paths.submitted_pickle,
        job.paths.result_pickle,
    ]:
        if path.exists():
            path.unlink()


def get_job_start_and_end_times(
    job_id: str,
) -> List[Tuple[str, float, Optional[float]]]:
    """
    Returns a list of (state, start_ts, end_ts) per attempt (chronological).
    Timestamps are Unix seconds as floats. end_ts is None if not finished yet.
    """
    out = (
        subprocess.check_output(
            shlex.split(f"sacct -j {job_id} -n -P -X -D -o JobID,State,Start,End"),
            text=True,
        )
        .strip()
        .splitlines()
    )

    attempts = []
    for line in out:
        jid, state, start, end = (line.split("|") + ["", "", "", ""])[:4]
        if jid != job_id:
            continue

        def to_ts(s: str) -> Optional[float]:
            if not s or s in ("Unknown", "N/A"):
                return None
            # sacct gives ISO-like "YYYY-MM-DDTHH:MM:SS"
            return datetime.fromisoformat(s).timestamp()

        start_ts = to_ts(start)
        if start_ts is None:
            continue  # skip attempts without a real start yet
        end_ts = to_ts(end)  # None if still running or not recorded

        attempts.append((state, start_ts, end_ts))

    attempts.sort(key=lambda x: x[1])  # chronological by start
    return attempts


def export():
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    load_dotenv()

