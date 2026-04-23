"""
step1_data_gen.py

Functions for first step of STP pipeline which involves data generations:
1. Generate conjectures from seed theorems and proofs
2. Prepare full sampling dataset from conjectures and target statements
    2.1. Deduplicate theorems in this dataset
3. Sample proofs on the resulting dataset (on conjectures and statements)
    3.1. Deduplicate proofs in this dataset
4. Prune dataset to only include provable samples


Number of conjectures generated in step 2 depends on `conjectures_per_statement`
Size of pruned dataset in step 4 is upper bounded according to `conjecture_multiplier`


At a high level,
INPUTS TO FILE:
- A model that HAS BEEN FINETUNED
- target statement datsaet

OUTPUTS FROM FILE:
- Raw data dumps:
    - dataset of all generated conjectures
    - dataset of all conjecture proof pairs
    - dataset of all statements proof pairs (use for seeding next conjecturer generation)
- Data used in next step of pipeline:
    - prover dataset of all statements and conjectures with proofs
    - filtered conjecturer training dataset

"""

from typing import List, Set, Optional, Any, Dict, TypeVar, Sequence
import logging
import copy
import wandb
import json
import random
import string
import time
import math
import os
import hashlib
from collections import defaultdict, Counter

from matplotlib import pyplot as plt


from sgs.models.model_types import (
    ConjecturerConfig,
    ConjecturerSetup,
    ResourcesConfig,
    ProverConfig,
)
from sgs.utils.monitor import UtilizationReport, log_utilization_report_timings
from sgs.utils.prompts import (
    NO_CONJECTURE_FOUND_TAG,
    NO_CODE_FOUND_TAG,
)
from sgs.models.query import query_model_batch, QueryResult, log_token_counts
from sgs.verification.verify_client import (
    verify_lean_code,
    VerificationOutput,
)
from sgs.pipeline.config import StatementSelectionMode
from sgs.data.dataset_types import (
    ProverDataset,
    Statement,
    Conjecture,
    Proof,
    ProverIterationData,
    ConjectureIterationData,
    ConjecturerDataset,
    DatasetType,
    StatementTag,
    IterationMetadata,
    EvaluationStatements,
    ConjectureList,
)
from sgs.pipeline.pipeline_pv import run_pipeline_proving_and_verification

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_data(
    wandb_run: wandb.sdk.wandb_run.Run,
    data: Any,
    iteration: int,
    num_generations: Optional[int] = None,
):
    """
    Log data to wandb

    Args:
        wandb_run: The wandb run to log to
        data: The data to log
        iteration: The iteration number
        num_generations: The number of generations

    We seperate iteration and num_generations as these are extra x axis plotting values.
    """
    if wandb_run is None:
        return

    new_data = {**data, "iteration": iteration}
    if num_generations is not None:
        new_data["num_generations"] = num_generations

    wandb_run.log(new_data)

def idxs_hash(idxs: List[int]) -> str:
    h = hashlib.blake2b(digest_size=16)  # 128-bit
    for x in idxs:
        h.update(x.to_bytes(4, "little", signed=False))  # use 8 bytes if needed
    return h.hexdigest()

def load_check(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_check(path: str, d: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)

T = TypeVar("T")
def deterministic_epoch_batch(
    items: Sequence[T],
    iteration: int,
    batch_size: int,
    seed: int,
    check_file_path: str,
) -> List[T]:
    assert batch_size > 0
    N = len(items)
    if N == 0:
        return []

    num_batches = math.ceil(N / batch_size)
    epoch = iteration // num_batches
    batch_idx = iteration % num_batches

    rng = random.Random(seed + epoch)

    # shuffle indices, not objects (keeps behavior stable if objects are unhashable)
    idxs = list(range(N))
    rng.shuffle(idxs)
    # ^ These indexes should be the same for each epoch

    # after rng.shuffle(idxs):
    cur_hash = idxs_hash(idxs)
    check = load_check(check_file_path)

    epoch_key = str(epoch)
    prev_key = str(epoch - 1)

    # within-epoch reproducibility
    if epoch_key in check:
        assert check[epoch_key]["hash"] == cur_hash
    else:
        check[epoch_key] = {"n": len(idxs), "hash": cur_hash}
        save_check(check_file_path, check)

    # across-epoch change (best-effort)
    if prev_key in check:
        assert check[prev_key]["hash"] != cur_hash


    start = batch_idx * batch_size
    end = min(start + batch_size, N)
    batch_items = [items[i] for i in idxs[start:end]]
    return batch_items


def data_gen(
    conjecturer_config: ConjecturerConfig,
    prover_config: ProverConfig,
    conjecturer_dataset_path: str,
    prover_dataset_path: str,
    iteration: int,  # If -1 then pick up from last iteration in the dataset
    conjectures_per_statement: int | Dict[DatasetType, int],
    verifier_address: str,
    verifier_timeout: int,
    gen_resources_config: ResourcesConfig,
    current_num_generations: int,
    statement_selection_mode: StatementSelectionMode,
    proofs_per_sample: int = 16,
    subsample_target_statements: Optional[int] = None,
    batch_target_statements: Optional[int] = None,
    batching_check_file_path: Optional[str] = None,
    conjecturer_dataset_save_path: Optional[str] = None,
    prover_dataset_save_path: Optional[str] = None,
    verification_resources_config: Optional[ResourcesConfig] = None,
    num_master_verification_workers: int = 0,
    pipeline_proving_and_verification: bool = False,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> IterationMetadata:
    """
    Generate conjectures from seed theorems and proofs

    Args:
        conjectures_per_statement: Number of conjectures to generate per statement (that has a proof)
        conjecture_multiplier: Multiplier for number of conjectures to statements in final dataset
        target_statements_dataset_path: Path to dataset we use to train the prover so far
        prover_dataset_path: Path to dataset we use to train the conjecturer so far
        conjecturer_dataset_path: Path to dataset we use to train the conjecturer so far
        model: Model endpoint to use for generating conjectures and proofs
        proofs_to_cache: Number of proofs to store per target statement in prover_dataset.target_statements
        conjecturer_dataset_save_path: Path to save the conjecturer dataset
            If None then we overwrite the existing dataset, save to conjecturer_dataset_path
        prover_dataset_save_path: Path to save the prover dataset
            If None then we overwrite the existing dataset, save to prover_dataset_path
        verification_resources_config: Resources config for verifying the lean files.
            If verifier_address == "local" then this is required, else ignored.
    """

    start_data_gen_time = time.time()

    num_generated_tokens = 0
    num_input_tokens = 0

    # Load all of our datasets up into memory
    prover_dataset = ProverDataset.load(prover_dataset_path)
    conjecturer_dataset = ConjecturerDataset.load(conjecturer_dataset_path)

    # Do a sanity check
    target_problems_ids = [x.id for x in prover_dataset.target_statements.values()]
    assert len(target_problems_ids) == len(set(target_problems_ids)), "Target problems ids are not unique"

    prover_iters: Set[int] = set([x.iteration for x in prover_dataset.iterations])
    conjecturer_iters: Set[int] = set(
        [x.iteration for x in conjecturer_dataset.iterations]
    )

    if iteration == -1:
        max_prover_iter = max(prover_iters)
        max_conjecturer_iter = max(conjecturer_iters)
        assert (
            max_prover_iter == max_conjecturer_iter
        ), "Prover and conjecturer must have same iterations"
        iteration = max_prover_iter + 1

    # ------------------------------------------------------------
    # Step 1: Generate conjectures from seed theorems and proofs
    # ------------------------------------------------------------


    prompts: List[str] = []
    seed_theorems: List[str] = []
    seed_proofs: List[None | Proof] = []
    headers: List[str] = []

    target_statements: List[Statement] = list(prover_dataset.target_statements.values())

    unsolved_target_statement_ids: List[str] = set()
    for statement in target_statements:
        proofs = statement.proofs
        assert all([proof.is_correct for proof in proofs]), "All proofs must be correct for unsolved target statements"
        if len(proofs) == 0:
            unsolved_target_statement_ids.add(statement.id)
    assert len(unsolved_target_statement_ids) == len(set(unsolved_target_statement_ids)), "Unsolved target statement ids are not unique"

    # For now we just select a random subsample of the target statements
    # Eventually we may want to stream this to avoid repeating data
    if subsample_target_statements is not None:
        assert batch_target_statements is None, "Cannot use both subsample_target_statements and batch_target_statements"
        target_statements = random.sample(
            target_statements, subsample_target_statements
        )

    if batch_target_statements is not None:
        assert subsample_target_statements is None, "Cannot use both subsample_target_statements and batch_target_statements"
        assert batching_check_file_path is not None, "batching_check_file_path is required when batching target statements"

        start_batch_time = time.time()
        # Now we do some accounting to get the epoch and iteration
        N = len(target_statements)
        num_batches_in_epoch = math.ceil(N / batch_target_statements)
        # Got to make sure they are stable
        target_statements.sort(key=lambda s: s.id)

        target_statements = deterministic_epoch_batch(
            items=target_statements,
            iteration=iteration,
            batch_size=batch_target_statements,
            seed=42,
            check_file_path=batching_check_file_path,
        )
        end_batch_time = time.time()
        logger.info(f"Batching target statements time: {end_batch_time - start_batch_time} seconds")

        if wandb_run is not None:
            wandb_run.log({
                "timing/batching_target_statements_time": end_batch_time - start_batch_time,
            })


    num_seed_statements = 0
    for statement in target_statements:
        if isinstance(conjectures_per_statement, int):
            num_to_add = conjectures_per_statement
        else:
            num_to_add = conjectures_per_statement[DatasetType(statement.source)]

        if conjecturer_config.setup == ConjecturerSetup.SEED_STATEMENT:
            if len(statement.proofs) == 0:
                continue

            num_seed_statements += 1

            # Generate conjectures
            num_added = 0
            while num_added < num_to_add:
                # Cycle through the proofs
                statement_proof_objs: List[Proof] = random.sample(
                    statement.proofs, len(statement.proofs)
                )
                for proof in statement_proof_objs:
                    assert proof.is_correct, "All proofs must be correct for seeding"

                    conjecturer_prompt = conjecturer_config.prompt_getter(
                        seed_theorem=statement.theorem,
                        seed_proof=proof.proof_str,
                        conjecturer_config=conjecturer_config,
                    )

                    seed_proofs.append(proof)
                    prompts.append(conjecturer_prompt)
                    seed_theorems.append(statement.theorem)
                    headers.append(statement.header)

                    num_added += 1

                    if num_added >= num_to_add:
                        break

        elif conjecturer_config.setup in [
            ConjecturerSetup.TARGET_STATEMENT,
            ConjecturerSetup.TARGET_STATEMENT_ONLY_UNSOLVED,
        ]:
            # We generate problems for all target statements

            if (
                conjecturer_config.setup
                == ConjecturerSetup.TARGET_STATEMENT_ONLY_UNSOLVED
            ):
                if len(statement.proofs) > 0:
                    # If we have already proven this statement then we can skip it
                    continue

            num_seed_statements += 1

            conjecturer_prompt = conjecturer_config.prompt_getter(
                seed_theorem=statement.theorem,
                conjecturer_config=conjecturer_config,
            )

            prompts.extend([conjecturer_prompt] * num_to_add)
            seed_theorems.extend([statement.theorem] * num_to_add)

            if statement.proofs != []:
                # We actually have proofs for this statement
                # We just pick the first one as representative
                proof = statement.proofs[0]
                assert proof.is_correct
                seed_proofs.extend([copy.deepcopy(proof) for _ in range(num_to_add)])
            else:
                seed_proofs.extend([None] * num_to_add)

            headers.extend([statement.header] * num_to_add)

    for prompt in prompts:
        if not isinstance(prompt, str):
            print("PROMPT IS NOT A STRING")
            print(prompt)
            assert False

    start_conjecturer_time = time.time()
    # Now sample conjectures
    model_responses: List[QueryResult]
    conjecture_gen_util_reports: List[UtilizationReport]
    model_responses, conjecture_gen_util_reports = query_model_batch(
        prompts=prompts,
        model_config=conjecturer_config,
        resources_config=gen_resources_config,
    )
    end_conjecturer_time = time.time()

    log_utilization_report_timings(
        conjecture_gen_util_reports,
        wandb_run,
        f"iteration_{iteration}/data_gen_conjecturer_jobs",
    )

    for response in model_responses:
        num_generated_tokens += response.output_token_count
        num_input_tokens += response.input_token_count

    num_conjectures_generated = len(model_responses)
    log_token_counts(wandb_run, model_responses, "conjectures")

    raw_model_responses: List[str] = [
        response.response_text for response in model_responses
    ]
    log_probs: List[List[float]] = [response.log_probs for response in model_responses]  # type: ignore
    output_tokens: List[List[int]] = [response.output_tokens for response in model_responses]  # type: ignore

    # Cast to set to deduplicate
    conjecture_strs: List[str] = [
        conjecturer_config.output_extractor(generation)
        for generation in raw_model_responses
    ]

    # Here we need to filter out any invalid conjectures
    num_invalid_conjectures = sum(
        [1 for conjecture in conjecture_strs if NO_CONJECTURE_FOUND_TAG in conjecture]
    )

    assert (
        len(conjecture_strs) == len(prompts) == len(seed_theorems) == len(seed_proofs)
    )

    # Clean up datastructure
    conjectures: List[Conjecture] = [
        Conjecture(
            seed_theorem=seed_theorems[i],
            seed_proof=seed_proofs[i],
            header=headers[i],
            conjecture=conjecture_strs[i],
            conjecture_full_generation=raw_model_responses[i],
            conjecture_full_generation_tokens=output_tokens[i],
            conjecture_full_generation_logprobs=log_probs[i],
        )
        for i in range(len(conjecture_strs))
        if NO_CONJECTURE_FOUND_TAG not in conjecture_strs[i]
    ]

    # Now deduplicate the conjectures
    conjecture_str_set: Set[str] = set(
        statement.theorem for statement in target_statements
    )
    deduplicated_conjectures: List[Conjecture] = []
    for conjecture in conjectures:
        if conjecture.conjecture not in conjecture_str_set:
            conjecture_str_set.add(conjecture.conjecture)
            deduplicated_conjectures.append(conjecture)

    logger.info("=" * 100)
    logger.info("STEP 1 - Finishing")
    logger.info(f"STEP 1 - Num starting target statements: {len(target_statements)}")
    logger.info(f"STEP 1 - Num starting seed statements: {num_seed_statements}")
    logger.info(
        f"STEP 1 - Num samples drawn from conjecturer model: {len(conjecture_strs)}"
    )
    logger.info(
        f"STEP 1 - Of these samples, {num_invalid_conjectures} we could not extract a conjecture"
    )
    logger.info(
        f"STEP 1 - This left us with {len(conjectures)} conjectures before deduplication conjecture theorem statement"
    )
    logger.info(
        f"STEP 1 - After deduplication of theorem statement, we are left with {len(deduplicated_conjectures)} conjectures"
    )
    logger.info(
        f"STEP 1 - Conjecturer Sampling Time: {end_conjecturer_time - start_conjecturer_time} seconds"
    )
    logger.info("=" * 100)

    table = wandb.Table(columns=["prompt", "response", "extracted_conjecture"])
    for i in range(min(20, len(prompts))):
        if len(prompts[i]) <= i or len(raw_model_responses[i]) <= i:
            break
        table.add_data(
            prompts[i],
            raw_model_responses[i],
            conjecturer_config.output_extractor(raw_model_responses[i]),
        )

    average_entropy = 0.0
    num_with_entropy = 0
    for response in model_responses:
        if response.average_entropy is not None:
            average_entropy += response.average_entropy
            num_with_entropy += 1

    if num_with_entropy == 0:
        print("WARNING: FOUND NO ENTROPY FOR ANY CONJECTURES")
        average_entropy = -1.0
    else:
        average_entropy = average_entropy / num_with_entropy

    current_num_generations += num_conjectures_generated
    log_data(
        wandb_run,
        {
            f"iteration_{iteration}/pipeline_data_samples/step_1_conjecturer_samples": table,
            # Timing
            "timing/sampling_conjectures_throughput(generations/sec)": len(
                conjecture_strs
            )
            / (end_conjecturer_time - start_conjecturer_time),
            "timing/sampling_conjectures_time(mins)": (
                end_conjecturer_time - start_conjecturer_time
            )
            / 60,
            # Data Gen
            "data_gen/num_seed_statements": num_seed_statements,
            "data_gen/num_raw_conjecture_strings": len(conjecture_strs),
            "data_gen/num_malformed_conjectures": num_invalid_conjectures,
            "data_gen/num_deduplicated_conjectures": len(deduplicated_conjectures),
            "conjectures/average_entropy": average_entropy,
        },
        iteration=iteration,
        num_generations=current_num_generations,
    )

    conjectures = deduplicated_conjectures

    # ------------------------------------------------------------
    # Sampling proofs and verifying them
    # ------------------------------------------------------------

    logger.info("=" * 100)
    logger.info(f"LOADING UP MODEL {prover_config.model_name} FOR PROVING")
    logger.info("=" * 100)

    # Set up new list to store our new target proofs
    copy_target_statements: List[Statement] = []
    for statement in target_statements:

        statement_copy = copy.deepcopy(statement)
        statement_copy.proofs = []

        if statement_selection_mode == StatementSelectionMode.LESS_16_PROOFS:
            # We actually only query if there are less than 16 valid proofs on this problem already
            valid_proofs = [x for x in statement.proofs if x.is_correct]
            if len(valid_proofs) < 16:
                copy_target_statements.append(statement_copy)

        else:
            copy_target_statements.append(statement_copy)

    log_data(
        wandb_run,
        {
            "data_gen/num_target_statements_total": len(copy_target_statements),
        },
        iteration=iteration,
        num_generations=current_num_generations,
    )

    if pipeline_proving_and_verification:
        # Here we need to decide how we can pipeline these components together

        copy_target_statements, conjectures = run_pipeline_proving_and_verification(
            target_statements=copy_target_statements,
            conjectures=conjectures,
            prover_config=prover_config,
            verification_resources_config=verification_resources_config,  # type: ignore
            gen_resources_config=gen_resources_config,
            proofs_per_sample=proofs_per_sample,
            num_master_verification_workers=num_master_verification_workers,
            iteration=iteration,
            verifier_timeout=verifier_timeout,
            wandb_run=wandb_run,
        )
        # Align downstream expectations used later (iteration metadata + util aggregation)
        # - proof_gen_util_reports and verification_reports exist in the non-pipelined path
        # - num_generated_proofs is used for iteration metadata and throughput
        proof_gen_util_reports: List[UtilizationReport] = []
        verification_reports: List[UtilizationReport] = []
        num_generated_proofs = sum(len(s.proofs) for s in copy_target_statements) + sum(
            len(c.proofs) for c in conjectures
        )
        # Keep current_num_generations accounting consistent with non-pipelined path
        current_num_generations += num_generated_proofs

    else:
        # ------------------------------------------------------------
        # Step 2: Sample proofs on all statements and conjectures
        # ------------------------------------------------------------

        # Now we sample proofs for statements and conjectures
        num_unique_statements = len(target_statements)
        num_unique_conjectures = len(conjectures)
        statement_proof_prompts = [
            prover_config.prompt_getter(
                header=statement.header, theorem=statement.theorem
            )
            for statement in target_statements
        ] * proofs_per_sample
        conjecture_proof_prompts = [
            prover_config.prompt_getter(
                header=conjecture.header, theorem=conjecture.conjecture
            )
            for conjecture in conjectures
        ] * proofs_per_sample
        all_prompts = statement_proof_prompts + conjecture_proof_prompts

        start_prover_time = time.time()
        proof_gen_util_reports: List[UtilizationReport]  # type: ignore
        model_responses, proof_gen_util_reports = query_model_batch(  # type: ignore
            prompts=all_prompts,
            model_config=prover_config,
            resources_config=gen_resources_config,
        )
        end_prover_time = time.time()

        num_generated_proofs = len(model_responses)

        log_utilization_report_timings(
            proof_gen_util_reports,
            wandb_run,
            f"iteration_{iteration}/data_gen_prover_jobs",
        )

        log_token_counts(wandb_run, model_responses, "data_gen/prover_token_counts")

        raw_model_responses: List[str] = [  # type: ignore
            response.response_text for response in model_responses
        ]
        log_probs = [  # type: ignore
            response.log_probs  # type: ignore
            for response in model_responses  # type: ignore
        ]
        output_tokens: List[List[int]] = [  # type: ignore
            response.output_tokens  # type: ignore
            for response in model_responses  # type: ignore
        ]

        proofs: List[str] = [
            prover_config.output_extractor(generation)
            for generation in raw_model_responses
        ]
        num_invalid_proofs = sum([1 for proof in proofs if NO_CODE_FOUND_TAG in proof])

        statement_proofs: List[str] = proofs[: len(statement_proof_prompts)]  # type: ignore
        statement_full_generations: List[str] = raw_model_responses[
            : len(statement_proof_prompts)
        ]  # type: ignore
        statement_full_generation_logprobs: List[List[float]] = log_probs[
            : len(statement_proof_prompts)
        ]  # type: ignore
        statement_full_generation_tokens: List[List[int]] = output_tokens[
            : len(statement_proof_prompts)
        ]  # type: ignore

        conjecture_proofs: List[str] = proofs[len(statement_proof_prompts) :]
        conjecture_full_generations: List[str] = raw_model_responses[
            len(statement_proof_prompts) :
        ]  # type: ignore
        conjecture_full_generation_logprobs: List[List[float]] = log_probs[
            len(statement_proof_prompts) :
        ]  # type: ignore
        conjecture_full_generation_tokens: List[List[int]] = output_tokens[
            len(statement_proof_prompts) :
        ]  # type: ignore

        assert len(statement_proofs) == len(statement_full_generations)
        assert len(conjecture_proofs) == len(conjecture_full_generations)
        assert len(statement_full_generation_tokens) == len(statement_full_generations)
        assert len(conjecture_full_generation_tokens) == len(conjecture_full_generations)

        for i, (proof_str, full_generation, full_generation_logprobs, full_generation_tokens) in enumerate(
            zip(
                statement_proofs,
                statement_full_generations,
                statement_full_generation_logprobs,
                statement_full_generation_tokens,
            )
        ):
            # Could be error tag
            assert proof_str is not None
            statement_idx = i % num_unique_statements
            copy_target_statements[statement_idx].proofs.append(
                Proof(
                    proof_str=proof_str,
                    full_generation=full_generation,
                    full_generation_logprobs=full_generation_logprobs,
                    full_generation_tokens=full_generation_tokens,
                    iteration_created=iteration,
                )
            )

        for i, (proof_str, full_generation, full_generation_logprobs, full_generation_tokens) in enumerate(
            zip(
                conjecture_proofs,
                conjecture_full_generations,
                conjecture_full_generation_logprobs,
                conjecture_full_generation_tokens,
            )
        ):
            # Could be error tag
            assert proof_str is not None
            conjecture_idx = i % num_unique_conjectures
            conjectures[conjecture_idx].proofs.append(
                Proof(
                    proof_str=proof_str,
                    full_generation=full_generation,
                    full_generation_logprobs=full_generation_logprobs,
                    full_generation_tokens=full_generation_tokens,
                    iteration_created=iteration,
                )
            )

        logger.info("=" * 100)
        logger.info("STEP 2 - Finishing")
        logger.info(f"STEP 2 - Num unique statements: {len(target_statements)}")
        logger.info(f"STEP 2 - Num unique conjectures: {len(conjectures)}")
        logger.info(
            f"STEP 2 - Total num problems to sample for: {len(target_statements) + len(conjectures)}"
        )
        logger.info(
            f"STEP 2 - Sampling {proofs_per_sample} proofs per problem leads to {len(raw_model_responses)} total model samples"
        )
        logger.info(
            f"STEP 2 - Of these samples, {num_invalid_proofs} we could not extract a proof from"
        )
        logger.info(
            f"STEP 2 - Prover Sampling Time: {end_prover_time - start_prover_time} seconds"
        )
        logger.info("=" * 100)

        table = wandb.Table(columns=["prompt", "response", "extracted_proof"])
        for i in range(min(20, len(all_prompts))):
            if len(all_prompts) <= i or len(raw_model_responses) <= i:
                break
            table.add_data(
                all_prompts[i],
                raw_model_responses[i],
                prover_config.output_extractor(raw_model_responses[i]),
            )
        log_data(
            wandb_run,
            {
                f"iteration_{iteration}/pipeline_data_samples/step_2_statement_proofs": table
            },
            iteration=iteration,
            num_generations=current_num_generations,
        )

        table = wandb.Table(columns=["prompt", "response", "extracted_proof"])
        for i in range(min(20, len(all_prompts))):
            i += len(statement_proof_prompts)
            if len(all_prompts) <= i or len(raw_model_responses) <= i:
                break
            table.add_data(
                all_prompts[i],
                raw_model_responses[i],
                prover_config.output_extractor(raw_model_responses[i]),
            )

        current_num_generations += num_generated_proofs

        average_entropy = 0.0
        num_with_entropy = 0
        for response in model_responses:
            if response.average_entropy is not None:
                average_entropy += response.average_entropy
                num_with_entropy += 1

        if num_with_entropy == 0:
            print("WARNING: FOUND NO ENTROPY FOR ANY PROOFS")
            average_entropy = -1.0
        else:
            average_entropy = average_entropy / num_with_entropy

        log_data(
            wandb_run,
            {
                f"iteration_{iteration}/pipeline_data_samples/step_2_conjecture_proofs": table,
                # Timing
                "timing/sampling_proofs_throughput(generations/sec)": len(all_prompts)
                / (end_prover_time - start_prover_time),
                "timing/sampling_proofs_time(mins)": (
                    end_prover_time - start_prover_time
                )
                / 60,
                # Data Gen
                "data_gen/num_proof_samples": len(raw_model_responses),
                "data_gen/num_invalid_proofs": num_invalid_proofs,
                "data_gen/prover_token_counts/average_entropy": average_entropy,
            },
            iteration=iteration,
            num_generations=current_num_generations,
        )

        # ------------------------------------------------------------
        # Step 3: Verify the correctness of all proofs
        # ------------------------------------------------------------

        number_of_chars_in_prover_proofs = []
        number_of_chars_in_conjecture = []

        # Now we validate all the lean files
        statement_lean_files: List[str] = []
        for statement in copy_target_statements:
            statement_lean_files.extend(
                [
                    statement.header + statement.theorem + proof.proof_str
                    for proof in statement.proofs
                ]
            )

            number_of_chars_in_prover_proofs.extend(
                [len(proof.proof_str) for proof in statement.proofs]
            )

        conjecture_lean_files: List[str] = []
        for conjecture in conjectures:
            conjecture_lean_files.extend(
                [
                    conjecture.header + conjecture.conjecture + proof.proof_str
                    for proof in conjecture.proofs
                ]
            )

            number_of_chars_in_prover_proofs.extend(
                [len(proof.proof_str) for proof in conjecture.proofs]
            )

            number_of_chars_in_conjecture.append(len(conjecture.conjecture))

        average_number_of_chars_in_prover_proofs = sum(
            number_of_chars_in_prover_proofs
        ) / max(len(number_of_chars_in_prover_proofs), 1)
        average_number_of_chars_in_conjecture = sum(
            number_of_chars_in_conjecture
        ) / max(len(number_of_chars_in_conjecture), 1)

        all_lean_files: List[str] = statement_lean_files + conjecture_lean_files

        start_verification_time = time.time()
        logger.info(f"Submitting {len(all_lean_files)} lean files to verifier")

        verification_outputs: List[VerificationOutput]
        verification_reports: List[UtilizationReport]  # type: ignore

        verification_outputs, verification_reports = verify_lean_code(
            verifier_address=verifier_address,
            lean_code=all_lean_files,
            resources_config=verification_resources_config,
            timeout=verifier_timeout,
            master_num_workers=num_master_verification_workers,
        )
        end_verification_time = time.time()

        log_utilization_report_timings(
            verification_reports,
            wandb_run,
            f"iteration_{iteration}/data_gen_verification_timings",
        )

        num_system_errors = sum(
            [1 for output in verification_outputs if output.system_error]
        )
        verification_times: List[float] = [
            output.output.get("verify_time", -100) for output in verification_outputs
        ]
        num_timeout_errors = sum(
            [1 for time in verification_times if time > verifier_timeout - 0.5]
        )

        # Log a simple histogram of verifier runtimes (excluding missing sentinel values).
        valid_verification_times: List[float] = [
            t
            for t in verification_times
            if isinstance(t, (int, float)) and t >= 0 and math.isfinite(t)
        ]
        if len(valid_verification_times) > 0:
            fig = plt.figure(figsize=(8, 6))
            max_x = max(max(valid_verification_times), float(verifier_timeout))
            plt.hist(
                valid_verification_times,
                bins=min(60, max(10, int(len(valid_verification_times) ** 0.5))),
                range=(0.0, max_x),
                edgecolor="black",
            )
            plt.axvline(
                float(verifier_timeout),
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"timeout={verifier_timeout}s",
            )
            plt.xlabel("Verification time (seconds)")
            plt.ylabel("Count")
            plt.title(
                f"Distribution of verification times (n={len(valid_verification_times)})"
            )
            plt.legend(loc="best")

            log_data(
                wandb_run,
                {"data_gen/verification_time_hist": wandb.Image(fig)},
                iteration=iteration,
                num_generations=current_num_generations,
            )
            plt.close(fig)

        assert len(verification_outputs) == len(all_lean_files)

        statement_verification_outputs: List[VerificationOutput] = verification_outputs[
            : len(statement_lean_files)
        ]
        conjecture_verification_outputs: List[VerificationOutput] = (
            verification_outputs[len(statement_lean_files) :]
        )

        for i, verification_output in enumerate(statement_verification_outputs):
            statement_ids = i // proofs_per_sample
            proof_idx = i % proofs_per_sample

            statement = copy_target_statements[statement_ids]
            proof = statement.proofs[proof_idx]
            proof.is_correct = verification_output.verdict

            # Now assert that the verification code is what we expect
            verified_code = verification_output.output.get("verified_code", None) # type: ignore
            if verified_code is not None:
                code_that_should_be_verified = statement.header + statement.theorem + proof.proof_str
                assert verified_code == code_that_should_be_verified, f"Verified code does not match proof. SHOULD BE: \n{code_that_should_be_verified}\nBUT IS: \n{verified_code}"
            
            proof.verification_dict = verification_output.output # type: ignore

        for i, verification_output in enumerate(conjecture_verification_outputs):
            conjecture_ids = i // proofs_per_sample
            proof_idx = i % proofs_per_sample

            conjecture = conjectures[conjecture_ids]
            proof = conjecture.proofs[proof_idx]
            proof.is_correct = verification_output.verdict

            # Now assert that the verification code is what we expect
            verified_code = verification_output.output.get("verified_code", None) # type: ignore
            if verified_code is not None:
                code_that_should_be_verified = conjecture.header + conjecture.conjecture + proof.proof_str
                assert verified_code == code_that_should_be_verified, f"Verified code does not match proof. SHOULD BE: \n{code_that_should_be_verified}\nBUT IS: \n{verified_code}"
            
            proof.verification_dict = verification_output.output # type: ignore

        logger.info("=" * 100)
        logger.info("STEP 3 - Finishing")
        logger.info(f"STEP 3 - Num verification system errors: {num_system_errors}")
        logger.info(f"STEP 3 - Num timeout errors: {num_timeout_errors}")
        logger.info("STEP 3 - All valid proofs verified")
        logger.info(
            f"STEP 3 - Verification Time: {end_verification_time - start_verification_time} seconds"
        )
        logger.info("=" * 100)

        statement_table = wandb.Table(columns=["lean_file", "verification_output"])
        for i in range(min(20, len(statement_lean_files))):
            if (
                len(statement_lean_files) <= i
                or len(statement_verification_outputs) <= i
            ):
                break
            statement_table.add_data(
                statement_lean_files[i],
                json.dumps(statement_verification_outputs[i].output, indent=4),
            )

        conjecture_table = wandb.Table(columns=["lean_file", "verification_output"])
        for i in range(min(20, len(conjecture_lean_files))):
            if (
                len(conjecture_lean_files) <= i
                or len(conjecture_verification_outputs) <= i
            ):
                break
            conjecture_table.add_data(
                conjecture_lean_files[i],
                json.dumps(conjecture_verification_outputs[i].output, indent=4),
            )

        log_data(
            wandb_run,
            {
                f"iteration_{iteration}/pipeline_data_samples/step_3_statement_verification_outputs": statement_table,
                f"iteration_{iteration}/pipeline_data_samples/step_3_conjecture_verification_outputs": conjecture_table,
                # Timing
                "timing/verification_time(mins)": (
                    end_verification_time - start_verification_time
                )
                / 60,
                "timing/verification_throughput(proofs/sec)": len(all_lean_files)
                / (end_verification_time - start_verification_time),
                # Data Gen
                "data_gen/num_verification_system_errors": num_system_errors,
                "data_gen/num_verification_timeout_errors": num_timeout_errors,
                "data_gen/prop_verification_errors": num_system_errors
                / len(all_lean_files),
                "data_gen/prop_verification_timeout_errors": num_timeout_errors
                / len(all_lean_files),
                # Summary
                "data_gen/prover_token_counts/average_number_of_chars_in_prover_proofs": average_number_of_chars_in_prover_proofs,
                "conjectures/average_number_of_chars_in_conjecture": average_number_of_chars_in_conjecture,
            },
            iteration=iteration,
            num_generations=current_num_generations,
        )

    # -------------------------------------------------------------------
    # Step 4: Filter statements and conjectures according to provability
    # -------------------------------------------------------------------

    # Filter for provable statements and conjectures, ande deduplicate the proofs
    # Provable statements and provable conjectures are deepcopies, so no bother with reference issues
    provable_statements: List[Statement] = []
    statements_selected_for_training_prover: List[Statement] = []

    provable_conjectures: List[Conjecture] = []
    conjectures_selected_for_training_conjecturer: List[Conjecture] = []
    conjectures_selected_for_training_prover: List[Conjecture] = []

    statement_solve_rates = []

    num_correct_proofs_for_statements = 0
    num_proofs_for_statements = 0
    for statement in copy_target_statements:
        valid_proofs: List[Proof] = [x for x in statement.proofs if x.is_correct]
        # Here we also sample for hardness. If we prove the statement more than 1/2 of the time, we assume we know it well
        num_correct_proofs_for_statements += len(valid_proofs)
        num_proofs_for_statements += len(statement.proofs)

        statement_solve_rate = len(valid_proofs) / proofs_per_sample
        statement_solve_rates.append(statement_solve_rate)

        if len(valid_proofs) > 0:
            new_statement = copy.deepcopy(statement)
            # new_statement.proofs = list(set(valid_proofs))
            # ^ We used to do this when we were not using groups
            provable_statements.append(new_statement)

        new_statement = copy.deepcopy(statement)
        if statement_selection_mode == StatementSelectionMode.HARD:
            if statement_solve_rate <= 0.5 and statement_solve_rate > 0.0:
                statements_selected_for_training_prover.append(new_statement)
        elif statement_selection_mode == StatementSelectionMode.UNSOLVED:
            if statement_solve_rate > 0.0 and statement.id in unsolved_target_statement_ids:
                statements_selected_for_training_prover.append(new_statement)
        elif statement_selection_mode == StatementSelectionMode.ALL_NONE_0_1:
            if statement_solve_rate > 0.0 and statement_solve_rate < 1.0:
                statements_selected_for_training_prover.append(new_statement)
        elif statement_selection_mode == StatementSelectionMode.ALL:
            if statement_solve_rate > 0.0:
                statements_selected_for_training_prover.append(new_statement)
        elif statement_selection_mode == StatementSelectionMode.LESS_16_PROOFS:
            # We only ended up selecting the statements that we can prove.
            if statement_solve_rate > 0.0:
                statements_selected_for_training_prover.append(new_statement)
        else:
            raise NotImplementedError(f"Statement selection mode {statement_selection_mode} not implemented")


    conjecture_solve_rates = []
    num_provable_in_any_way = 0

    # First we will get all non 0 and 1 solve rates
    solve_rates = []
    for conjecture in conjectures:
        valid_proofs = [x for x in conjecture.proofs if x.is_correct]
        solve_rate = len(valid_proofs) / proofs_per_sample
        if solve_rate > 0 and solve_rate < 1:
            solve_rates.append(solve_rate)

    # Now order from smallest to largest
    solve_rates.sort()
    if solve_rates:
        assert solve_rates[0] <= solve_rates[-1]
        solve_rate_70th_percentile = solve_rates[int(len(solve_rates) * 0.7)]
    else:
        # There were no non 0 or 1 anyway
        print("WARNING: NO NON 0 OR 1 SOLVE RATES FOUND")
        solve_rate_70th_percentile = 0.0

    # Now find out what th 80th percentile is

    num_correct_proofs_for_conjectures = 0
    num_proofs_for_conjectures = 0
    for conjecture in conjectures:
        valid_proofs = [x for x in conjecture.proofs if x.is_correct]
        num_correct_proofs_for_conjectures += len(valid_proofs)
        num_proofs_for_conjectures += len(conjecture.proofs)

        conjecture_solve_rate = len(valid_proofs) / proofs_per_sample
        conjecture_solve_rates.append(conjecture_solve_rate)

        # We record all problems that were provable but "not trivial" (the model got some wrong).
        # We then weight the rewards by this value.
        solve_rate = len(valid_proofs) / proofs_per_sample

        if solve_rate > 0:
            num_provable_in_any_way += 1

        if solve_rate > 0 and solve_rate <= solve_rate_70th_percentile:
            new_conjecture = copy.deepcopy(conjecture)
            # new_conjecture.proofs = list(set(valid_proofs))
            new_conjecture.solve_rate = conjecture_solve_rate
            conjectures_selected_for_training_conjecturer.append(new_conjecture)


        # Now decide what which ones we will use to train the prover
        new_conjecture = copy.deepcopy(conjecture)
        if statement_selection_mode == StatementSelectionMode.HARD:
            if solve_rate <= 0.5 and solve_rate > 0.0:
                conjectures_selected_for_training_prover.append(new_conjecture)
        elif statement_selection_mode == StatementSelectionMode.UNSOLVED:
            if solve_rate > 0.0:
                conjectures_selected_for_training_prover.append(new_conjecture)
        elif statement_selection_mode == StatementSelectionMode.ALL_NONE_0_1:
            if solve_rate > 0.0 and solve_rate < 1.0:
                conjectures_selected_for_training_prover.append(new_conjecture)
        elif statement_selection_mode == StatementSelectionMode.ALL:
            if solve_rate > 0.0:
                conjectures_selected_for_training_prover.append(new_conjecture)
        elif statement_selection_mode == StatementSelectionMode.LESS_16_PROOFS:
            if solve_rate > 0.0:
                conjectures_selected_for_training_prover.append(new_conjecture)
        else:
            raise NotImplementedError(f"Statement selection mode {statement_selection_mode} not implemented")

    # For backward compatibility with logging code
    provable_conjectures = conjectures_selected_for_training_conjecturer

    logger.info("=" * 100)
    logger.info("STEP 4 - Finishing")
    logger.info(
        f"STEP 4 - Num provable statements: {len(provable_statements)}/{len(copy_target_statements)}"
    )
    logger.info(
        f"STEP 4 - Total number proofs for statements: {num_proofs_for_statements}"
    )
    logger.info(f"STEP 4 - Num provable conjectures: {num_provable_in_any_way}")
    logger.info(
        f"STEP 4 - Num provable conjectures in difficulty range: {len(provable_conjectures)}/{len(conjectures)}"
    )
    logger.info(
        f"STEP 4 - Total number proofs for conjectures: {num_proofs_for_conjectures}"
    )
    logger.info("=" * 100)

    statement_table = wandb.Table(columns=["theorem", "proof", "passed"])
    for i, statement in enumerate(provable_statements):
        if i >= 10:
            break
        for proof in statement.proofs:
            statement_table.add_data(
                statement.theorem, proof.proof_str, proof.is_correct
            )

    conjecture_table = wandb.Table(columns=["conjecture", "proof", "passed"])
    for i, conjecture in enumerate(provable_conjectures):
        if i >= 10:
            break
        for proof in conjecture.proofs:
            conjecture_table.add_data(
                conjecture.conjecture, proof.proof_str, proof.is_correct
            )

    log_data(
        wandb_run,
        {
            f"iteration_{iteration}/pipeline_data_samples/step_4_provable_statements": statement_table,
            f"iteration_{iteration}/pipeline_data_samples/step_4_provable_conjectures": conjecture_table,
            "data_gen/num_provable_statements": len(provable_statements),
            "data_gen/num_total_correct_proofs_statements": num_correct_proofs_for_statements,
            "data_gen/pass_at_1": num_correct_proofs_for_statements / (len(target_statements) * proofs_per_sample),
            "data_gen/num_provable_conjectures": num_provable_in_any_way,
            "data_gen/num_conjectures_to_train_conjecturer": len(
                conjectures_selected_for_training_conjecturer
            ),
            "data_gen/num_conjectures_to_train_prover": len(
                conjectures_selected_for_training_prover
            ),
            "data_gen/num_total_proofs_conjectures_in_difficulty_range": num_correct_proofs_for_conjectures,
            "conjectures/num_total_proofs_conjectures_in_difficulty_range": num_correct_proofs_for_conjectures,
            "data_gen/average_conjecture_solve_rate": (
                sum(conjecture_solve_rates) / len(conjecture_solve_rates)
                if len(conjecture_solve_rates) > 0
                else 0.0
            ),
        },
        iteration=iteration,
        num_generations=current_num_generations,
    )

    for label, solve_rates in [
        ("conjecture", conjecture_solve_rates),
        ("statement", statement_solve_rates),
    ]:
        if len(solve_rates) == 0:
            continue

        # Histogram of conjecture solve rates and log solve rates
        fig = plt.figure(figsize=(8, 6))

        # Count occurrences of each discrete value
        counts = Counter(solve_rates)
        values = sorted(counts.keys())
        frequencies = [counts[v] for v in values]

        # Create bar plot
        bin_width = 1.0 / proofs_per_sample if proofs_per_sample > 0 else 0.05
        plt.bar(
            values,
            frequencies,
            width=bin_width * 0.9,
            align="center",
            edgecolor="black",
        )
        
        # Add count labels on top of each bar
        for val, freq in zip(values, frequencies):
            plt.text(val, freq, str(freq), ha='center', va='bottom')
        
        plt.xlabel(f"{label} Solve Rate")
        plt.ylabel("Count")
        plt.title(f"Distribution of {label} Solve Rates")

        # Show all possible discrete solve rates and keep bars within axes
        tick_values = [i / proofs_per_sample for i in range(proofs_per_sample + 1)]
        plt.xticks(tick_values)
        plt.xlim(-bin_width / 2, 1 + bin_width / 2)

        log_data(
            wandb_run,
            {f"data_gen/{label}_solve_rate_distribution": wandb.Image(fig)},
            iteration=iteration,
            num_generations=current_num_generations,
        )

        plt.yscale("log")  # Keep log scale if you still want it
        plt.ylabel("Count (log scale)")

        log_data(
            wandb_run,
            {f"data_gen/{label}_solve_rate_distribution_log_scale": wandb.Image(fig)},
            iteration=iteration,
            num_generations=current_num_generations,
        )

        plt.close()

    # -------------------------------------------------------------------
    # Step 6: Save datasets and log the results
    # -------------------------------------------------------------------

    # Need to create new iteration of prover datset
    # Need to create new iteration for conjecturer dataset
    # Need to update the target statements dataset

    # Start with the target statements dataset
    for statement in provable_statements:
        # Recall that statements are copies, so no reference issues
        proofs_list: List[Proof] = prover_dataset.target_statements[statement.id].proofs

        statement_valid_proofs: List[Proof] = [
            x for x in statement.proofs if x.is_correct
        ]

        if len(proofs_list) <= 16:
            # How many proofs do we need to get to 8
            num_proofs_to_add = 16 - len(proofs_list)
            proofs_list.extend(statement_valid_proofs[:num_proofs_to_add])
            # Deduplicate proofs
            proofs_list = list(set(proofs_list))

    # Now add all provable statements and conjectures to the new iteration

    # We only add the provable but not easy statements to the most recent iteration
    iteration_statements: List[Statement] = [
        copy.deepcopy(x) for x in statements_selected_for_training_prover
    ]
    iteration_statements.extend(
        [
            Statement(
                id="".join(random.choices(string.ascii_letters + string.digits, k=8)),
                header=x.header,
                theorem=x.conjecture,
                tag=StatementTag.CONJECTURE.value,
                proofs=x.proofs,
            )
            for x in conjectures_selected_for_training_prover
        ]
    )
    new_prover_iteration = ProverIterationData(
        iteration=iteration, iter_data=iteration_statements
    )
    prover_dataset.iterations.append(new_prover_iteration)
    # Overwrite

    if prover_dataset_save_path is not None:
        prover_dataset.save(prover_dataset_save_path)
    else:
        prover_dataset.save(prover_dataset_path)

    # Now add the new conjectures to the conjecturer dataset
    new_conjecturer_iteration = ConjectureIterationData(
        iteration=iteration, iter_data=[copy.deepcopy(x) for x in conjectures_selected_for_training_conjecturer]
    )
    conjecturer_dataset.iterations.append(new_conjecturer_iteration)

    if conjecturer_dataset_save_path is not None:
        conjecturer_dataset.save(conjecturer_dataset_save_path)
    else:
        conjecturer_dataset.save(conjecturer_dataset_path)

    # Finally we log per-statement pass rates to wandb
    passes = []
    for statement in copy_target_statements:
        pass_rate = sum([1 for proof in statement.proofs if proof.is_correct]) / len(
            statement.proofs
        )
        passes.append(pass_rate > 0)

    log_data(
        wandb_run,
        {"data_gen/overall_solve_rate": sum(passes) / len(passes)},
        iteration=iteration,
        num_generations=current_num_generations,
    )

    logger.info("=" * 100)
    logger.info("STEP 7 - Finishing")
    logger.info(
        f"STEP 7 - We have added {len(conjectures_selected_for_training_conjecturer)} conjectures to conjecturer dataset"
    )
    logger.info(
        f"STEP 7 - We have added {len(statements_selected_for_training_prover)} statements and {len(conjectures_selected_for_training_prover)} conjectures to prover dataset"
    )
    num_proofs_to_train_on = sum(
        [len(statement.proofs) for statement in iteration_statements]
    )
    logger.info(f"STEP 7 - We have {num_proofs_to_train_on} proofs to train on")
    logger.info("=" * 100)

    # get the number of target problems that we have solved
    num_target_problems_solved: Dict[str, int] = defaultdict(lambda: 0)
    num_target_problems: Dict[str, int] = defaultdict(lambda: 0)
    for id, statement in prover_dataset.target_statements.items():
        # check that all proofs are correct
        assert all([proof.is_correct for proof in statement.proofs])

        assert statement.source is not None
        num_target_problems[statement.source] += 1
        if len(statement.proofs) > 0:
            num_target_problems_solved[statement.source] += 1

    percent_solved = {
        dataset_type: num_target_problems_solved[dataset_type]
        / num_target_problems[dataset_type]
        for dataset_type in num_target_problems
    }

    log_data(
        wandb_run,
        {
            "data_gen/num_statements_added_to_prover_dataset": len(
                iteration_statements
            ),
            "data_gen/num_conjectures_added_to_conjecturer_dataset": len(
                conjectures_selected_for_training_conjecturer
            ),
            "data_gen/num_proofs_to_train_prover_on": num_proofs_to_train_on,
        },
        iteration=iteration,
        num_generations=current_num_generations,
    )

    for dataset_type, num_solved in num_target_problems_solved.items():
        log_data(
            wandb_run,
            {
                f"data_gen/cumulative_num_target_problems_solved_{dataset_type}": num_solved,
                f"data_gen/percent_target_problems_solved_{dataset_type}": percent_solved[
                    dataset_type
                ],
            },
            iteration=iteration,
            num_generations=current_num_generations,
        )

    all_util_reports = (
        conjecture_gen_util_reports + verification_reports + proof_gen_util_reports
    )
    iteration_metadata = IterationMetadata(
        num_generated_conjectures=num_conjectures_generated,
        num_target_statements=num_seed_statements,
        proofs_per_statement=proofs_per_sample,
        num_generated_proofs=num_generated_proofs,
        num_generations=num_conjectures_generated + num_generated_proofs,
        num_generated_tokens=num_generated_tokens,
        num_input_tokens=num_input_tokens,
        util_reports=all_util_reports,
        total_num_generations=current_num_generations,
    )

    end_data_gen_time = time.time()

    # log data_gen throughput
    # This is the number of proofs / total time
    log_data(
        wandb_run,
        {
            "data_gen/throughput": num_generated_proofs
            / (end_data_gen_time - start_data_gen_time),
        },
        iteration=iteration,
        num_generations=current_num_generations,
    )

    target_statements_dict = defaultdict(list)
    for statement in copy_target_statements:
        target_statements_dict[statement.source].append(statement)

    # Now we will log an eval dataset of how we did on the target statements
    target_evaluation_statements = EvaluationStatements(
        statements=target_statements_dict,
        iteration_metadata=iteration_metadata,
    )

    # Get the path to save this
    if prover_dataset_save_path is not None:
        if ".json" in prover_dataset_save_path:
            target_eval_save_path = prover_dataset_save_path.replace(
                ".json", "_eval.json"
            )
        else:
            target_eval_save_path = prover_dataset_save_path + "_eval.json"

        target_evaluation_statements.save(target_eval_save_path)

    if conjecturer_dataset_save_path is not None:
        if ".json" in conjecturer_dataset_save_path:
            iteration_conjecture_list_save_path = conjecturer_dataset_save_path.replace(
                ".json", "_iteration_conjecture_list.json"
            )
        else:
            iteration_conjecture_list_save_path = conjecturer_dataset_save_path + "_iteration_conjecture_list.json"
    else:
        iteration_conjecture_list_save_path = conjecturer_dataset_path + "_iteration_conjecture_list.json"

    iteration_conjecture_list = ConjectureList(
        conjectures=conjectures
    )
    iteration_conjecture_list.save(iteration_conjecture_list_save_path)

    return iteration_metadata
