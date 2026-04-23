from typing import List, Optional, Dict, Any
import json
import pickle
import os
import tempfile
from pydantic import BaseModel, Field
from enum import Enum


def _atomic_json_dump(path: str, data: Dict[str, Any]) -> None:
    """Atomically write JSON to avoid NFS partial/corrupt writes."""
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _assert_and_get_file_ext(path: str) -> str:
    """Return the lowercase file extension (including dot) and assert it's supported."""
    ext = os.path.splitext(path)[1].lower()
    assert ext in {
        ".json",
        ".pkl",
    }, f"Expected path ending in .json or .pkl, got: {path}"
    return ext


class Proof(BaseModel):
    proof_str: str
    full_generation: str
    full_generation_logprobs: Optional[List[float]] = None
    full_generation_tokens: Optional[List[int]] = None
    is_correct: Optional[bool] = None
    review: Optional[float] = None
    review_cot: Optional[str] = None
    verification_dict: Optional[Dict[str, Any]] = None
    iteration_created: Optional[int] = None

    # For deduplication
    def __hash__(self):
        return hash(self.proof_str)

    def __eq__(self, other):
        if not isinstance(other, Proof):
            return NotImplemented
        return self.proof_str == other.proof_str


# ----------------------------------
# Prover Classes
# ----------------------------------


class StatementTag(Enum):
    TARGET = "statement"
    CONJECTURE = "conjecture"


class Statement(BaseModel):
    id: str  # Unique ID, normally this can just be the name of the problems
    header: str
    theorem: str
    tag: str
    source: Optional[str] = None
    proofs: List[Proof] = Field(default_factory=list)

    @property
    def num_proofs(self) -> int:
        return len(self.proofs)


class ProverIterationData(BaseModel):
    iteration: int
    iter_data: List[Statement]

    def __init__(self, **data):
        super().__init__(**data)
        # Check valid iter_data
        for statement in self.iter_data:
            # Make sure there are no statements that have all incorrect proofs
            if all([not proof.is_correct for proof in statement.proofs]):
                # Remove the statement
                raise ValueError(f"Statement {statement.id} has all incorrect proofs")


class ProverDataset(BaseModel):
    # Training data for each iteration
    iterations: List[ProverIterationData] = Field(default_factory=list)

    # The target statements that we are trying to prove, with most recent proofs of each
    # Target statements with proofs are used to seed the conjecturer
    # For all target statements, proof or not, we sample solution proofs of
    # This is just handy for keeping track, we can construct all previous proofs from iterations
    # Maps `theorem` -> `Statement`
    target_statements: Dict[str, Statement] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "rb") as f:
                dataset: ProverDataset = pickle.load(f)
            return dataset

        # Read whole file into memory quickly (minimize time holding the fd open)
        with open(path, "rb") as f:
            raw = f.read()

        # Decode + parse from memory (no further filesystem interaction)
        text = raw.decode("utf-8")
        data = json.loads(text)
        return cls(**data)

    def save(self, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return

        _atomic_json_dump(path, self.model_dump())


# ----------------------------------
# Conjecturer Classes
# ----------------------------------


class Conjecture(BaseModel):
    seed_theorem: str
    header: str
    conjecture: str
    conjecture_full_generation: str
    conjecture_full_generation_tokens: Optional[List[int]] = None
    conjecture_full_generation_logprobs: Optional[List[float]] = None
    proofs: List[Proof] = Field(default_factory=list)
    seed_proof: Optional[Proof] = None
    solve_rate: Optional[float] = None
    # seed_proof: Optional[str] = None

    @property
    def num_proofs(self) -> int:
        return len(self.proofs)


class ConjectureIterationData(BaseModel):
    iteration: int
    iter_data: List[Conjecture]

    def __init__(self, **data):
        super().__init__(**data)
        # Check valid iter_data
        for conjecture in self.iter_data:
            # Assert at least one proof is correct

            # Make sure there are no statements that have all incorrect proofs
            if all([not proof.is_correct for proof in conjecture.proofs]):
                raise ValueError(f"Conjecture {conjecture.id} has all incorrect proofs")


class ConjectureList(BaseModel):
    conjectures: List[Conjecture]

    @classmethod
    def load(cls, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "rb") as f:
                dataset: ConjectureList = pickle.load(f)
            return dataset

        # Read whole file into memory quickly (minimize time holding the fd open)
        with open(path, "rb") as f:
            raw = f.read()

        # Decode + parse from memory (no further filesystem interaction)
        text = raw.decode("utf-8")
        data = json.loads(text)
        return cls(**data)

    def save(self, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return

        _atomic_json_dump(path, self.model_dump())


class ConjecturerDataset(BaseModel):
    iterations: List[ConjectureIterationData] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "rb") as f:
                dataset: ConjecturerDataset = pickle.load(f)
            return dataset

        # Read whole file into memory quickly (minimize time holding the fd open)
        with open(path, "rb") as f:
            raw = f.read()

        # Decode + parse from memory (no further filesystem interaction)
        text = raw.decode("utf-8")
        data = json.loads(text)
        return cls(**data)

    def save(self, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return

        _atomic_json_dump(path, self.model_dump())


# ----------------------------------
# Target Statement Classes
# ----------------------------------


class TargetStatementsDataset(BaseModel):
    """For keeping track of how we are doing on the target statements."""

    statements: List[Statement]

    @classmethod
    def load(cls, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "rb") as f:
                dataset: TargetStatementsDataset = pickle.load(f)
            return dataset

        # Read whole file into memory quickly (minimize time holding the fd open)
        with open(path, "rb") as f:
            raw = f.read()

        # Decode + parse from memory (no further filesystem interaction)
        text = raw.decode("utf-8")
        data = json.loads(text)
        return cls(**data)


# ----------------------------------
# Utilization
# ----------------------------------


class JobType(Enum):
    VERIFICATION = "verification"
    GENERATION = "generation"
    TRAINING = "training"
    EVAL_VERIFICATION = "eval_verification"
    EVAL_GENERATION = "eval_generation"


class GPUSample(BaseModel):
    index: int
    name: str
    util_percent: float
    mem_used_mb: float
    mem_total_mb: float
    power_watts: Optional[float] = None
    temperature_c: Optional[float] = None


class SystemSample(BaseModel):
    timestamp: float
    cpu_percent: float
    mem_used_gb: float
    mem_percent: float
    load_avg_1: Optional[float]
    load_avg_5: Optional[float]
    load_avg_15: Optional[float]
    process_cpu_percent: Optional[float]
    process_mem_mb: Optional[float]
    process_num_threads: Optional[int]
    io_read_mb: Optional[float]
    io_write_mb: Optional[float]
    net_sent_mb: Optional[float]
    net_recv_mb: Optional[float]
    gpus: List[GPUSample]

    # Aggregated over full process tree (parent + children)
    process_tree_cpu_percent: Optional[float] = None
    process_tree_mem_mb: Optional[float] = None
    top_children: Optional[List[Dict[str, Any]]] = None

    # cgroup-based CPU util normalized by assigned CPUs (SLURM)
    cgroup_cpu_util_percent: Optional[float] = None
    cgroup_assigned_cpus: Optional[int] = None


class UtilizationReport(BaseModel):
    start_time: float
    end_time: float
    duration_sec: float
    hostname: str
    platform: str
    node_name: str
    cpu_count: Optional[int]
    mem_total_gb: Optional[float]
    slurm_env: Dict[str, str]
    samples: List[SystemSample]
    summary: Dict[str, Any]
    num_examples: Optional[int] = None
    # ^ Number of examples processed by the job
    num_errors: Optional[int] = None
    # ^ If we are processing lots of examples, how many errored?
    job_type: Optional[str] = None
    outer_job_start_time: Optional[float] = None
    outer_job_end_time: Optional[float] = None
    # ^ If you are running as a submitit job, these can be computed in master process and filled in

    """
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "hostname": self.hostname,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "mem_total_gb": self.mem_total_gb,
            "slurm_env": self.slurm_env,
            "samples": [
                {
                    **{k: v for k, v in asdict(s).items() if k != "gpus"},
                    "gpus": [asdict(g) for g in s.gpus],
                }
                for s in self.samples
            ],
            "summary": self.summary,
        }
    """


# ----------------------------------
# Storing results from each round
# ----------------------------------


class IterationMetadata(BaseModel):
    num_generated_conjectures: int
    num_target_statements: int
    proofs_per_statement: int
    num_generated_proofs: int
    num_generations: Optional[int] = None
    num_generated_tokens: Optional[int] = None
    num_input_tokens: Optional[int] = None
    util_reports: Optional[List[UtilizationReport]] = None
    total_num_generations: Optional[int] = None


class EvaluationStatements(BaseModel):
    """For saving evaluation proofs"""

    statements: Dict[str, List[Statement]]
    iteration_metadata: Optional[IterationMetadata] = (
        None  # Set to none for backwards compatibility
    )

    @classmethod
    def load(cls, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "rb") as f:
                dataset: EvaluationStatements = pickle.load(f)
            return dataset

        # Read whole file into memory quickly (minimize time holding the fd open)
        with open(path, "rb") as f:
            raw = f.read()

        text = raw.decode("utf-8")
        data = json.loads(text)
        return cls(**data)

    def save(self, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return

        _atomic_json_dump(path, self.model_dump())


class SeriesEvaluationStatements(BaseModel):
    """For storing evaluation proofs for a series of rounds."""

    iterations: List[EvaluationStatements]

    @classmethod
    def load(cls, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "rb") as f:
                dataset: SeriesEvaluationStatements = pickle.load(f)
            return dataset

        # Read whole file into memory quickly (minimize time holding the fd open)
        with open(path, "rb") as f:
            raw = f.read()

        text = raw.decode("utf-8")
        data = json.loads(text)
        return cls(**data)

    def save(self, path: str):
        ext = _assert_and_get_file_ext(path)
        if ext == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return

        _atomic_json_dump(path, self.model_dump())


# ----------------------------------
# Eval Dataset Classes
# ----------------------------------


class DatasetType(Enum):
    D_3K = "d_3k"
