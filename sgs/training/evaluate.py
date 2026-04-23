from typing import List, Dict, Any, Optional
import random
import copy
from pathlib import Path

import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from sgs.models.model_types import ProverConfig, ResourcesConfig
from sgs.data.dataset_types import (
    DatasetType,
    Statement,
    Proof,
    EvaluationStatements,
    IterationMetadata,
    SeriesEvaluationStatements,
    JobType,
)
from sgs.data.load_dataset import load_eval_dataset
from sgs.models.model_types import QueryResult
from sgs.models.query import query_model_batch, log_token_counts
from sgs.utils.prompts import (
    NO_CODE_FOUND_TAG,
)
from sgs.verification.verify_client import verify_lean_code, VerificationOutput
from sgs.utils.logging_config import get_logger
from sgs.utils.monitor import (
    UtilizationReport,
    log_utilization_report_timings,
)

logger = get_logger(__name__)


def evaluate_prover(
    prover_config: ProverConfig,
    gen_resources_config: ResourcesConfig,
    eval_datasets: List[DatasetType],
    verifier_address: str,
    best_of_n: int = 16,
    verification_resources_config: Optional[ResourcesConfig] = None,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    save_proofs_path: Optional[str] = None,
    compute_bootstrap: bool = False,
    iteration_metadata: Optional[IterationMetadata] = None,
    master_num_workers: int = 0,
    lean_version: str = "4.15",
    _debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate a prover model.

    Args:
        prover_config: The config for the prover model.
        gen_resources_config: The resources config for the generation set.
        eval_datasets: The datasets to evaluate on.
        verifier_address: The address of the verifier server. If 'local' then the verifier will be run locally (possibly through submitit)
        best_of_n: The number of generations to use for each problem.
        verification_resources_config: The resources config used for running the lean compiler. If verifier_address not 'local' then this is ignored.
        wandb_run: The wandb run to log to.
        _debug: Whether to run in debug mode.

    Returns:
        Data that can be logged to wandb.
    """

    to_log: List[Dict[str, Any]] = []

    logger.info("=" * 100)
    logger.info(f"STEP 1 - Preparing eval data for {eval_datasets}")
    logger.info("=" * 100)

    # Load the eval dataset
    queries: List[str] = []
    headers: List[str] = []
    theorems: List[str] = []
    all_statements: List[Statement] = []
    num_problems_per_dataset: List[int] = []
    for eval_dataset in eval_datasets:
        statements: List[Statement] = load_eval_dataset(eval_dataset)

        if _debug:
            # If we are in debug mode then we just select the first 10 statements
            statements = statements[:10]

        all_statements.extend(statements)

        num_problems_per_dataset.append(len(statements))
        for statement in statements:
            queries.extend(
                [
                    prover_config.prompt_getter(
                        header=statement.header,
                        theorem=statement.theorem,
                    )
                ]
                * best_of_n
            )
            headers.extend([statement.header] * best_of_n)
            theorems.extend([statement.theorem] * best_of_n)

    logger.info("=" * 100)
    logger.info(f"STEP 2 - Querying model for {len(queries)} prompts")
    logger.info("=" * 100)

    # Query the model
    results: List[QueryResult]
    gen_reports: List[UtilizationReport]

    results, gen_reports = query_model_batch(
        queries, prover_config, gen_resources_config
    )
    for report in gen_reports:
        # Update the job type
        report.job_type = JobType.EVAL_GENERATION.value

    log_utilization_report_timings(
        gen_reports,
        wandb_run,
        "eval_generation_timings",
    )

    current_index = 0
    for i, eval_dataset in enumerate(eval_datasets):
        eval_dataset_results = results[
            current_index : current_index + num_problems_per_dataset[i] * best_of_n
        ]
        to_log.extend(
            log_token_counts(
                wandb_run,
                model_responses=eval_dataset_results,
                log_prefix=f"eval_{eval_dataset.value}",
            )
        )
        current_index += num_problems_per_dataset[i] * best_of_n

    # Get proofs
    proofs: List[str] = []
    full_generations: List[str] = []
    for result in results:
        proofs.append(prover_config.output_extractor(result.response_text))
        full_generations.append(result.response_text)

    assert (
        len(queries) == len(headers) == len(theorems) == len(proofs)
    ), "Number of queries, headers, theorems, and proofs must match"

    lean_files = []
    for header, theorem, proof in zip(headers, theorems, proofs):
        if "by" not in theorem and "by" not in proof:
            # Add "by" in because our parsing removed it
            theorem = theorem + " by\n"

        lean_files.append(header + theorem + proof)

    # Collect number of proofs couldnt extract
    num_proofs_couldnt_extract: int = sum(
        [1 for proof in proofs if NO_CODE_FOUND_TAG in proof]
    )

    # Save intermediate results (before verification) so we don't lose generation progress
    if save_proofs_path is not None:
        intermediate_dataset: Dict[str, List[Statement]] = {}
        proof_idx = 0
        stmt_idx = 0
        for i, eval_dataset in enumerate(eval_datasets):
            intermediate_dataset[eval_dataset.value] = []
            for j in range(num_problems_per_dataset[i]):
                statement = copy.deepcopy(all_statements[stmt_idx])
                stmt_idx += 1
                for k in range(best_of_n):
                    statement.proofs.append(
                        Proof(
                            proof_str=proofs[proof_idx],
                            full_generation=full_generations[proof_idx],
                            is_correct=None,
                            review=None,
                        )
                    )
                    proof_idx += 1
                intermediate_dataset[eval_dataset.value].append(statement)
        intermediate_save = EvaluationStatements(
            statements=intermediate_dataset,
            iteration_metadata=iteration_metadata,
        )
        intermediate_save.save(save_proofs_path)
        logger.info(f"Saved intermediate (unverified) proofs to {save_proofs_path}")

    logger.info("=" * 100)
    logger.info(f"STEP 3 - Verifying {len(lean_files)} proofs")
    logger.info("=" * 100)

    # Verify the proofs
    verification_outputs: List[VerificationOutput]
    verification_reports: List[UtilizationReport]

    verification_outputs, verification_reports = verify_lean_code(
        verifier_address=verifier_address,
        lean_code=lean_files,
        resources_config=verification_resources_config,
        timeout=500,  # Longer timeout when we are running evaluation
        master_num_workers=master_num_workers,
        lean_version=lean_version,
    )

    for report in verification_reports:
        # Update the job type
        report.job_type = JobType.EVAL_VERIFICATION.value

    if iteration_metadata is not None:
        if iteration_metadata.util_reports is not None:
            iteration_metadata.util_reports.extend(verification_reports)
        else:
            iteration_metadata.util_reports = verification_reports

    log_utilization_report_timings(
        verification_reports,
        wandb_run,
        "eval_verification_timings",
    )

    verdicts: List[bool] = [output.verdict for output in verification_outputs]

    logger.info("=" * 100)
    logger.info("STEP 4 - Calculating metrics")
    logger.info("=" * 100)

    # Now we process each dataset scores ones at a time
    current_index = 0
    statement_idx: int = 0
    dataset_scores: Dict[DatasetType, Any] = {}
    master_pass_rate: Dict[DatasetType, float] = {}

    # Reconstruct the datasets with only correct proofs
    correct_proofs_dataset: Dict[str, List[Statement]] = {}
    all_proofs_dataset: Dict[str, List[Statement]] = {}

    for i, eval_dataset in enumerate(eval_datasets):
        dataset_scores[eval_dataset] = {}

        correct_proofs_dataset[eval_dataset.value] = []
        all_proofs_dataset[eval_dataset.value] = []
        system_errors = []

        for j in range(num_problems_per_dataset[i]):
            # Get the data for this dataset
            statement = copy.deepcopy(all_statements[statement_idx])
            statement_id = all_statements[statement_idx].id
            statement_idx += 1

            # Get the verdicts for this statement
            statement_verdicts = verdicts[current_index : current_index + best_of_n]

            # Get all the system errors and put these in a wandb table and log them
            system_error_vers: List[VerificationOutput] = [
                x
                for x in verification_outputs[current_index : current_index + best_of_n]
                if x.system_error
            ]
            system_errors.extend(
                [x.output.get("system_errors", None) for x in system_error_vers]
            )

            # First we sort out the all proofs dataset
            all_proof_statement = copy.deepcopy(statement)
            for k, verdict in enumerate(statement_verdicts):
                all_proof_statement.proofs.append(
                    Proof(
                        proof_str=proofs[current_index + k],
                        full_generation=full_generations[current_index + k],
                        is_correct=verdict,
                        review=None,
                    )
                )

            # Now we get the correct proofs dataset
            for k, verdict in enumerate(statement_verdicts):
                if verdict:
                    statement.proofs.append(
                        Proof(
                            proof_str=proofs[current_index + k],
                            full_generation=full_generations[current_index + k],
                            is_correct=verdict,
                            review=None,
                        )
                    )

            correct_proofs_dataset[eval_dataset.value].append(statement)
            all_proofs_dataset[eval_dataset.value].append(all_proof_statement)

            dataset_scores[eval_dataset][statement_id] = statement_verdicts

            current_index += best_of_n

        system_error_table = wandb.Table(
            data=[[x] for x in system_errors], columns=["system_error"]
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"eval_{eval_dataset.value}/system_errors": system_error_table,
                }
            )
        else:
            to_log.append(
                {f"eval_{eval_dataset.value}/system_errors": system_error_table}
            )

    eval_statements_to_save = EvaluationStatements(
        statements=all_proofs_dataset,
        iteration_metadata=iteration_metadata,
    )

    # Now we calculate all of the scores
    for eval_dataset in eval_datasets:
        # Get the pass rate
        master_table = wandb.Table(columns=["problem_name", "pass_rate", "pass"])
        passes = 0
        for problem_name, verdict_list in dataset_scores[eval_dataset].items():
            if any(verdict_list):
                passes += 1

            pass_rate = sum(verdict_list) / len(verdict_list)
            master_table.add_data(
                problem_name,
                pass_rate,
                any(verdict_list),
            )

        pass_rate = passes / len(dataset_scores[eval_dataset])
        master_pass_rate[eval_dataset] = pass_rate

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"eval_{eval_dataset.value}/pass_rate": pass_rate,
                }
            )
            wandb_run.log(
                {
                    f"eval_{eval_dataset.value}/master_table": master_table,
                }
            )
        else:
            to_log.append({f"eval_{eval_dataset.value}/pass_rate": pass_rate})
            to_log.append({f"eval_{eval_dataset.value}/master_table": master_table})

        # Now do some bootstrap

        if compute_bootstrap:
            ks = []
            average_pass_rates = []
            all_bootstrap_pass_rates = []
            for best_of in list(range(1, best_of_n + 1)):
                boostrap_pass_rates = []

                # Get 1000 boostrap samples
                for i in range(100):
                    # Sample best_of_n statements
                    num_passes = 0
                    for statement_id, verdict_list in dataset_scores[
                        eval_dataset
                    ].items():
                        # random sample with replacement best_of verdicts
                        sampled_verdicts = random.choices(verdict_list, k=best_of)
                        if any(sampled_verdicts):
                            num_passes += 1

                    boostrap_pass_rates.append(
                        num_passes / len(dataset_scores[eval_dataset])
                    )

                # We now have 1000 boostrap pass rates
                ks.append(best_of)
                average_pass_rates.append(
                    sum(boostrap_pass_rates) / len(boostrap_pass_rates)
                )
                all_bootstrap_pass_rates.append(boostrap_pass_rates)

            # Now lets do some nice logging
            pass_rate_at_k_data = [[x, y] for (x, y) in zip(ks, average_pass_rates)]
            pass_rate_at_k_table = wandb.Table(
                data=pass_rate_at_k_data, columns=["k", "average_accuracy"]
            )
            pass_rate_at_k_plot = wandb.plot.line(
                pass_rate_at_k_table,
                "k",
                "average_accuracy",
                title="Average Accuracy vs k",
            )

            bootstrap_scatter_data = []
            for k, bootstrap_scores in enumerate(all_bootstrap_pass_rates):
                for score in bootstrap_scores:
                    bootstrap_scatter_data.append([ks[k], score])

            bootstrap_scatter_table = wandb.Table(
                data=bootstrap_scatter_data, columns=["k", "accuracy"]
            )
            bootstrap_scatter_plot = wandb.plot.scatter(
                bootstrap_scatter_table,
                "k",
                "accuracy",
                title="Bootstrap Accuracy vs k",
            )

            bootstrap_pass_at_1 = average_pass_rates[0]

            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval_{eval_dataset.value}/pass_rate_at_k": pass_rate_at_k_plot,
                    }
                )

                wandb_run.log(
                    {
                        f"eval_{eval_dataset.value}/pass_rate_at_k_scatter": bootstrap_scatter_plot,
                    }
                )

                wandb_run.log(
                    {
                        f"eval_{eval_dataset.value}/bootstrap_pass_at_1": bootstrap_pass_at_1,
                    }
                )
            else:
                to_log.append(
                    {f"eval_{eval_dataset.value}/pass_rate_at_k": pass_rate_at_k_plot}
                )
                to_log.append(
                    {
                        f"eval_{eval_dataset.value}/pass_rate_at_k_scatter": bootstrap_scatter_plot
                    }
                )
                to_log.append(
                    {
                        f"eval_{eval_dataset.value}/bootstrap_pass_at_1": bootstrap_pass_at_1
                    }
                )

    for eval_dataset in eval_datasets:
        # We technically log the number of code not found as same for all datasets
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"eval_{eval_dataset.value}/num_code_not_found": num_proofs_couldnt_extract,
                }
            )
        else:
            to_log.append(
                {
                    f"eval_{eval_dataset.value}/num_code_not_found": num_proofs_couldnt_extract
                }
            )

    current_index = 0
    num_no_verification_time = {}
    num_system_errors = {}
    for i, eval_dataset in enumerate(eval_datasets):
        # get num statements
        num_statements = num_problems_per_dataset[i]
        # get verification times for this dataset
        dataset_ver_ouputs: List[VerificationOutput] = verification_outputs[
            current_index : current_index + num_statements * best_of_n
        ]

        # We do a sanity check plot of verification times
        verification_times: List[float] = [
            output.output.get("verify_time", -1) for output in dataset_ver_ouputs
        ]

        # Log how many system errors there are
        num_system_errors[eval_dataset] = sum(
            [1 for output in dataset_ver_ouputs if output.system_error]
        )

        # Count number of -1s
        num_no_verification_time[eval_dataset] = sum(
            [1 for time in verification_times if time == -1]
        )

        fig = plt.figure(figsize=(10, 6))
        plt.hist(verification_times, bins=100)
        plt.xlabel("Verification Time (s)")
        plt.ylabel("Frequency")
        plt.title("Verification Time Distribution")

        if wandb_run is not None:
            # Now we plot the verification times
            wandb_run.log(
                {
                    f"eval_{eval_dataset.value}/verification_time_distribution": wandb.Image(
                        fig
                    ),
                    f"eval_{eval_dataset.value}/num_system_errors": num_system_errors[
                        eval_dataset
                    ],
                }
            )
        else:
            to_log.append(
                {
                    f"eval_{eval_dataset.value}/verification_time_distribution": copy.deepcopy(
                        fig
                    )
                }
            )
            to_log.append(
                {
                    f"eval_{eval_dataset.value}/num_system_errors": num_system_errors[
                        eval_dataset
                    ]
                }
            )

        plt.close()

        current_index += num_statements * best_of_n

    if save_proofs_path is not None:
        eval_statements_to_save.save(save_proofs_path)

    # No matter if we have wandb log or not we print summary scores
    logger.info("=" * 100)
    logger.info("EVAL RESULTS")
    for eval_dataset in eval_datasets:
        logger.info(f"Eval dataset: {eval_dataset.value}")
        logger.info(f"Pass rate: {master_pass_rate[eval_dataset]}")
        logger.info(f"Number of problems: {len(dataset_scores[eval_dataset])}")
        logger.info(
            f"Number of passes: {sum([1 for verdict_list in dataset_scores[eval_dataset].values() if any(verdict_list)]) / len(dataset_scores[eval_dataset])}"
        )
        logger.info(
            f"Number of problems with no verification time: {num_no_verification_time[eval_dataset]}"
        )
        logger.info(f"Number of system errors: {num_system_errors[eval_dataset]}")
        logger.info("\n")

    logger.info(f"Number of proofs couldnt extract: {num_proofs_couldnt_extract}")
    logger.info("=" * 100)

    return to_log


def generate_proofs(
    prover_config: ProverConfig,
    gen_resources_config: ResourcesConfig,
    eval_dataset: DatasetType = None,
    statements: List[Statement] = None,
    statements_name: str = None,
    best_of_n: int = 16,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    save_proofs_path: Optional[str] = None,
    _debug: bool = False,
) -> EvaluationStatements:
    """
    Generate (but do NOT verify) a batch of proofs for an evaluation dataset.

    This mirrors the generation portion of `evaluate_prover`, but skips all verification
    and metric computation. Generated proofs are stored in `EvaluationStatements` with
    `Proof.is_correct=None` so they can be verified later.
    """

    if eval_dataset is not None and statements is not None:
        raise ValueError("Either eval_dataset or statements must be provided, not both")
    if eval_dataset is None and statements is None:
        raise ValueError("Either eval_dataset or statements must be provided")

    if statements is not None:
        assert (
            statements_name is not None
        ), "statements_name must be provided if statements are provided"

    if eval_dataset is not None:
        statements: List[Statement] = load_eval_dataset(eval_dataset)
        if _debug:
            statements = statements[:2]

        statements_name = eval_dataset.value

    logger.info("=" * 100)
    logger.info(f"GENERATE PROOFS - Preparing eval data for {statements_name}")
    logger.info("=" * 100)

    queries: List[str] = []
    for statement in statements:
        queries.extend(
            [
                prover_config.prompt_getter(
                    header=statement.header,
                    theorem=statement.theorem,
                )
            ]
            * best_of_n
        )

    logger.info("=" * 100)
    logger.info(f"GENERATE PROOFS - Querying model for {len(queries)} prompts")
    logger.info("=" * 100)

    results: List[QueryResult]
    gen_reports: List[UtilizationReport]
    results, gen_reports = query_model_batch(
        queries,
        prover_config,
        gen_resources_config,
    )

    for report in gen_reports:
        report.job_type = JobType.EVAL_GENERATION.value

    log_utilization_report_timings(
        gen_reports,
        wandb_run,
        "eval_generation_timings",
    )

    # Optional token logging (mirrors evaluate_prover behavior)
    _ = log_token_counts(
        wandb_run,
        model_responses=results,
        log_prefix=f"eval_{statements_name}",
    )

    # Extract proof snippets + keep full generations (and optional token/logprob traces)
    proofs: List[str] = [
        prover_config.output_extractor(r.response_text) for r in results
    ]

    num_proofs_couldnt_extract: int = sum(
        1 for proof in proofs if NO_CODE_FOUND_TAG in proof
    )
    logger.info(
        f"GENERATE PROOFS - Could not extract code for {num_proofs_couldnt_extract}/{len(proofs)} generations"
    )

    # Reconstruct statements with proofs (is_correct intentionally left as None)
    statements_with_proofs: List[Statement] = []
    current_index = 0

    for statement in statements:
        statement_copy = copy.deepcopy(statement)
        statement_copy.proofs = []

        for k in range(best_of_n):
            r = results[current_index + k]
            statement_copy.proofs.append(
                Proof(
                    proof_str=proofs[current_index + k],
                    full_generation=r.response_text,
                    full_generation_logprobs=None,
                    full_generation_tokens=r.output_tokens,
                    is_correct=None,
                    review=None,
                    review_cot=None,
                )
            )

        statements_with_proofs.append(statement_copy)
        current_index += best_of_n

    # Log the input and output token counts

    eval_statements_to_save = EvaluationStatements(
        statements={statements_name: statements_with_proofs},
        iteration_metadata=None,
    )

    if save_proofs_path is not None:
        eval_statements_to_save.save(save_proofs_path)
        logger.info(f"GENERATE PROOFS - Saved proofs to {save_proofs_path}")
    else:
        logger.info(
            "GENERATE PROOFS - No save path provided; returning results without saving"
        )

    return eval_statements_to_save


def verify_proofs(
    path_to_evaluation_statements: str,
    path_to_save_verified_statements: str,
    resources_config: ResourcesConfig,
    verifier_address: str = "server",
    timeout: int = 500,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    """
    Verify proofs stored in an `EvaluationStatements` JSON file.

    This is intended to pair with `generate_proofs()`: generate a batch of proofs
    (with `Proof.is_correct=None`) and then later verify them and write verdicts
    back into the same JSON file.
    """

    logger.info("=" * 100)
    logger.info(
        f"VERIFY PROOFS - Loading evaluation statements from {path_to_evaluation_statements}"
    )
    logger.info("=" * 100)

    eval_statements: EvaluationStatements = EvaluationStatements.load(
        path_to_evaluation_statements
    )

    # Flatten all proofs across datasets/statements so we can verify in one batch
    lean_files: List[str] = []
    proof_ptrs: List[Proof] = []

    dataset: str
    statements: List[Statement]

    num_no_code_found = 0
    total_proofs = 0
    for dataset, statements in eval_statements.statements.items():
        for statement in statements:
            for proof in statement.proofs:
                theorem = statement.theorem
                proof_str = proof.proof_str

                total_proofs += 1

                assert "by" in theorem, "Theorem must contain 'by'"

                lean_files.append(statement.header + theorem + proof_str)
                proof_ptrs.append(proof)

    logger.info(f"VERIFY PROOFS - Total proofs: {total_proofs}")
    logger.info(f"VERIFY PROOFS - Number of no code found: {num_no_code_found}")
    logger.info(
        f"VERIFY PROOFS - Percent no code found: {num_no_code_found / total_proofs}"
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/num_no_code_found": num_no_code_found,
                "eval/total_proofs": total_proofs,
                "eval/percent_no_code_found": num_no_code_found / total_proofs,
            }
        )

    assert len(lean_files) > 0, "No proofs found to verify in EvaluationStatements"
    assert len(lean_files) == len(
        proof_ptrs
    ), "Internal mismatch building proof mapping"

    logger.info("=" * 100)
    logger.info(f"VERIFY PROOFS - Verifying {len(lean_files)} proofs")
    logger.info("=" * 100)

    verification_outputs: List[VerificationOutput]
    verification_reports: List[UtilizationReport]
    verification_outputs, verification_reports = verify_lean_code(
        verifier_address=verifier_address,
        lean_code=lean_files,
        resources_config=resources_config,
        timeout=timeout,
    )

    # Some logging things
    num_system_errors = sum(
        [1 for output in verification_outputs if output.system_error]
    )
    verification_times: List[float] = [
        output.output.get("verify_time", -100) for output in verification_outputs
    ]
    num_timeout_errors = sum([1 for time in verification_times if time > timeout - 0.5])
    # Create histogram of verification times
    fig = plt.figure(figsize=(10, 6))
    plt.hist(verification_times, bins=100)
    plt.xlabel("Verification Time (s)")
    plt.ylabel("Frequency")
    plt.title("Verification Time Distribution")
    if wandb_run is not None:
        wandb_run.log(
            {
                "infra/verification_time_distribution": wandb.Image(fig),
                "infra/num_system_errors": num_system_errors,
                "infra/num_timeout_errors": num_timeout_errors,
            }
        )

    for report in verification_reports:
        report.job_type = JobType.EVAL_VERIFICATION.value

    # If the file already has iteration metadata, extend util reports for bookkeeping
    if eval_statements.iteration_metadata is not None:
        if eval_statements.iteration_metadata.util_reports is not None:
            eval_statements.iteration_metadata.util_reports.extend(verification_reports)
        else:
            eval_statements.iteration_metadata.util_reports = verification_reports

    # Write verdicts back onto the proofs in-order
    assert (
        len(verification_outputs) == len(proof_ptrs)
    ), f"Expected {len(proof_ptrs)} verification outputs, got {len(verification_outputs)}"
    num_system_errors = 0
    num_correct = 0
    for proof, output in zip(proof_ptrs, verification_outputs):
        proof.is_correct = bool(output.verdict)
        if output.verdict:
            num_correct += 1
        if output.system_error:
            num_system_errors += 1

    # Persist results (overwrite in place)
    eval_statements.save(path_to_save_verified_statements)

    logger.info("=" * 100)
    logger.info("VERIFY PROOFS - Done")
    logger.info(f"VERIFY PROOFS - Correct: {num_correct}/{len(proof_ptrs)}")
    logger.info(f"VERIFY PROOFS - System errors: {num_system_errors}/{len(proof_ptrs)}")
    logger.info("=" * 100)

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/num_correct": num_correct,
                "eval/num_total_problems": len(proof_ptrs),
                "eval/percent_correct": num_correct / len(proof_ptrs),
            }
        )
