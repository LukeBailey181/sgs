from typing import List, Set, Optional, Any, Dict
import wandb
import json
import time
from dataclasses import dataclass
from matplotlib import pyplot as plt


from sgs.models.model_types import (
    ResourcesConfig,
    ProverConfig,
)
from sgs.utils.monitor import UtilizationReport, log_utilization_report_timings
from sgs.utils.prompts import (
    NO_CODE_FOUND_TAG,
)
from sgs.models.query import QueryResult, log_token_counts
from sgs.verification.verify_client import (
    VerificationOutput,
)
from sgs.data.dataset_types import (
    Statement,
    Conjecture,
    Proof,
)
from sgs.models.query import QueryServer
from sgs.verification.verify_server import VerifyServer


def run_pipeline_proving_and_verification(
    target_statements: List[Statement],
    conjectures: List[Conjecture],
    prover_config: ProverConfig,
    verification_resources_config: ResourcesConfig,
    gen_resources_config: ResourcesConfig,
    proofs_per_sample,
    num_master_verification_workers: int,
    iteration: int,
    verifier_timeout: int = 500,
    wandb_run=None,
):
    """
    Note that launching master workers will occur for query server equal to number of GPUs on master machine.
    """

    # We will make a list of problem info dataclasses
    # They will have prompt, statement, conjecture

    @dataclass
    class QueryVerifySample:
        source_obj: Statement | Conjecture
        index_in_source_list: int
        prompt: str
        proof: Optional[str] = None
        full_generation: Optional[str] = None
        full_generation_logprobs: Optional[List[float]] = None
        full_generation_tokens: Optional[List[int]] = None
        query_server_task_id: Optional[str] = None
        verify_server_task_id: Optional[str] = None

    query_verify_samples: List[QueryVerifySample] = []

    # Now we sample proofs for statements and conjectures
    for i, statement in enumerate(target_statements):
        for _ in range(proofs_per_sample):
            query_verify_samples.append(
                QueryVerifySample(
                    source_obj=statement,
                    index_in_source_list=i,
                    prompt=prover_config.prompt_getter(
                        header=statement.header, theorem=statement.theorem
                    ),
                    proof=None,
                )
            )
    for i, conjecture in enumerate(conjectures):
        for _ in range(proofs_per_sample):
            query_verify_samples.append(
                QueryVerifySample(
                    source_obj=conjecture,
                    index_in_source_list=i,
                    prompt=prover_config.prompt_getter(
                        header=conjecture.header, theorem=conjecture.conjecture
                    ),
                    proof=None,
                )
            )

    all_prompts = [sample.prompt for sample in query_verify_samples]

    # Mapping dicts and holders
    query_task_id_to_sample: Dict[str, QueryVerifySample] = {}
    verify_task_id_to_sample: Dict[str, QueryVerifySample] = {}
    seen_query_task_ids: Set[str] = set()

    # NOTE! These are in no specific order, we return them for logging
    query_results: List[QueryResult] = []
    with QueryServer(
        worker_resources_config=gen_resources_config,
        monitor=True,
        model_config=prover_config,
    ) as query_server:
        # Wait for query guy to start up
        time.sleep(5)

        all_verification_times: List[float] = []
        with VerifyServer(
            worker_resources_config=verification_resources_config,
            monitor=True,
            #verify_timeout=VERIFIER_TIMEOUT,
            verify_timeout=verifier_timeout,
            allow_idling=False,  # Allow idling as it can complete faster than generator can feed it tasks
        ) as verify_server:
            start_prover_time = time.time()
            # 1) Submit all prompts to query server and launch workers
            query_task_ids: List[str] = query_server.add_tasks(all_prompts)
            query_server.launch_workers()
            query_server.launch_master_worker()  # type: ignore

            assert len(query_task_ids) == len(
                query_verify_samples
            ), "Number of query task ids does not match number of query verify samples"
            for qt_id, sample in zip(query_task_ids, query_verify_samples):
                query_task_id_to_sample[qt_id] = sample
                sample.query_server_task_id = qt_id

            # 2) Stream query → verify
            verify_workers_launched = False
            start_verification_time: Optional[float] = None
            while True:
                if len(seen_query_task_ids) == len(query_task_ids):
                    break

                # Snapshot results and compute delta
                query_results_snapshot: Dict[str, Any] = query_server.results()
                new_task_ids: List[str] = [
                    t
                    for t in query_results_snapshot.keys()
                    if t not in seen_query_task_ids
                ]
                if not new_task_ids:
                    time.sleep(0.5)
                    continue

                # Prepare Lean files and map to samples
                lean_files_batch: List[str] = []
                batch_samples: List[QueryVerifySample] = []
                for qt_id in new_task_ids:
                    r = query_results_snapshot[qt_id]
                    # r may be a dict or a dataclass-like; access defensively

                    response_text = r.get("response_text")
                    query_result = QueryResult(
                        response_text=r.get("response_text"),
                        input_token_count=r.get("input_token_count"),
                        output_token_count=r.get("output_token_count"),
                        is_error=r.get("is_error"),
                        cost=r.get("cost", 0),
                        log_probs=r.get("log_probs"),
                        output_tokens=r.get("output_tokens"),
                        average_entropy=r.get("average_entropy"),
                    )
                    query_results.append(query_result)

                    # Extract and record proof and raw generation
                    sample = query_task_id_to_sample[qt_id]
                    sample.full_generation = response_text
                    sample.full_generation_logprobs = query_result.log_probs
                    sample.full_generation_tokens = query_result.output_tokens
                    sample.proof = prover_config.output_extractor(response_text)

                    # Construct lean file
                    if isinstance(sample.source_obj, Statement):
                        lean_file = f"{sample.source_obj.header}{sample.source_obj.theorem}{sample.proof}"
                    else:
                        lean_file = f"{sample.source_obj.header}{sample.source_obj.conjecture}{sample.proof}"
                    lean_files_batch.append(lean_file)
                    batch_samples.append(sample)

                # Add to verify server in a single batch
                verify_task_ids_batch: List[str] = verify_server.add_tasks(
                    lean_files_batch
                )
                assert len(verify_task_ids_batch) == len(
                    batch_samples
                ), "Mismatch between verify tasks and samples"
                for vt_id, sample in zip(verify_task_ids_batch, batch_samples):
                    sample.verify_server_task_id = vt_id
                    verify_task_id_to_sample[vt_id] = sample

                # Launch verify workers on first batch
                if not verify_workers_launched:
                    start_verification_time = time.time()
                    verify_server.launch_workers()
                    if (
                        num_master_verification_workers
                        and num_master_verification_workers > 0
                    ):
                        print(
                            f"Launching {num_master_verification_workers} master verification workers"
                        )
                        verify_server.launch_master_worker(  # type: ignore
                            num_master_verification_workers
                        )  # type: ignore
                    verify_workers_launched = True

                # Mark these query tasks as seen
                seen_query_task_ids.update(new_task_ids)

                # Avoid tight loop
                time.sleep(0.5)

            # 3) All verify tasks submitted; wait for completion
            verify_server.wait_until_done()
            end_prover_time = time.time()

            # NOTE! These are in no specific order, we include them for logging
            verification_verdicts: List[VerificationOutput] = []

            # 4) Gather verify results and append Proofs back to sources
            verify_results: Dict[str, Any] = verify_server.results()
            for vt_id, r in verify_results.items():
                sample = verify_task_id_to_sample.get(vt_id)  # type: ignore
                if sample is None:
                    continue

                verdict = r.get("verdict")
                compiler_output = r.get("output")
                system_error = r.get("system_error")

                # Optional sanity-check: verified code should exactly match header + theorem/conjecture + proof.
                if (
                    compiler_output.get("verified_code", None) 
                ):
                    verified_code = compiler_output.get("verified_code")
                    if isinstance(sample.source_obj, Statement):
                        expected_code = (
                            sample.source_obj.header
                            + sample.source_obj.theorem
                            + sample.proof
                        )
                    else:
                        expected_code = (
                            sample.source_obj.header
                            + sample.source_obj.conjecture
                            + sample.proof
                        )
                    assert verified_code == expected_code, (
                        "Verified code does not match expected code.\n"
                        f"EXPECTED:\n{expected_code}\n\n"
                        f"GOT:\n{verified_code}"
                    )

                verification_verdicts.append(
                    VerificationOutput(
                        verdict=verdict,
                        output=compiler_output,
                        system_error=system_error,
                    )
                )

                verification_time = r.get("verify_time", None)
                if verification_time is not None:
                    all_verification_times.append(verification_time)



                # Append proof back to the correct object
                proof_obj = Proof(
                    proof_str=sample.proof,
                    full_generation=sample.full_generation,
                    full_generation_logprobs=sample.full_generation_logprobs,
                    full_generation_tokens=sample.full_generation_tokens,
                    is_correct=bool(verdict),
                    verification_dict=r.get("output"),
                    iteration_created=iteration,
                )

                sample.source_obj.proofs.append(proof_obj)

            # 5) Collect util reports
            query_util_reports: List[UtilizationReport] = query_server.util_reports()
            verify_util_reports: List[UtilizationReport] = verify_server.util_reports()

    # Do some sanity checking
    for statement in target_statements:
        assert (
            len(statement.proofs) == proofs_per_sample
        ), "Number of proofs does not match"
    for conjecture in conjectures:
        assert (
            len(conjecture.proofs) == proofs_per_sample
        ), "Number of proofs does not match"

    assert len(query_results) == len(
        query_verify_samples
    ), "Number of query results does not match number of query verify samples"
    assert len(verification_verdicts) == len(
        query_verify_samples
    ), "Number of verification verdicts does not match number of query verify samples"

    # ------------------------------------------------------------------------------------- #
    # Now we need to do loads of logging about all this stuff
    # ------------------------------------------------------------------------------------- #
    try:
        if wandb_run is not None:


            if len(all_verification_times) > 0:
                fig = plt.figure(figsize=(8, 6))
                max_x = max(max(all_verification_times), float(verifier_timeout))
                plt.hist(
                    all_verification_times,
                    bins=min(60, max(10, int(len(all_verification_times) ** 0.5))),
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
                    f"Distribution of verification times (n={len(all_verification_times)})"
                )
                plt.legend(loc="best")

                wandb_run.log(
                    {
                        "pipeline_pv/verification_time_hist": wandb.Image(fig),
                    }
                )
                plt.close(fig)

            # Prover token counts and entropy
            log_token_counts(
                wandb_run, query_results, "pipeline_pv/prover_token_counts"
            )
            average_entropy = 0.0
            count_entropy = 0
            for response in query_results:
                if getattr(response, "average_entropy", None) is not None:
                    average_entropy += float(response.average_entropy)  # type: ignore[arg-type]
                    count_entropy += 1
            average_entropy = (
                (average_entropy / count_entropy) if count_entropy > 0 else -1
            )

            # Sample tables: prompts, responses, extracted proofs
            table = wandb.Table(columns=["prompt", "response", "extracted_proof"])
            max_rows = min(20, len(query_verify_samples))
            for i in range(max_rows):
                s = query_verify_samples[i]
                table.add_data(s.prompt, s.full_generation or "", s.proof or "")
            wandb_run.log(
                {"pipeline_pv/samples/statement_and_conjecture_proofs": table}
            )

            # Build lean files and summary stats
            statement_lean_files: List[str] = []
            conjecture_lean_files: List[str] = []
            number_of_chars_in_prover_proofs: List[int] = []
            number_of_chars_in_conjecture: List[int] = []

            for statement in target_statements:
                statement_lean_files.extend(
                    [
                        f"{statement.header}{statement.theorem}{p.proof_str}"
                        for p in statement.proofs
                    ]
                )
                number_of_chars_in_prover_proofs.extend(
                    [len(p.proof_str) for p in statement.proofs]
                )

            for conjecture in conjectures:
                conjecture_lean_files.extend(
                    [
                        f"{conjecture.header}{conjecture.conjecture}{p.proof_str}"
                        for p in conjecture.proofs
                    ]
                )
                number_of_chars_in_prover_proofs.extend(
                    [len(p.proof_str) for p in conjecture.proofs]
                )
                number_of_chars_in_conjecture.append(len(conjecture.conjecture))

            average_number_of_chars_in_prover_proofs = sum(
                number_of_chars_in_prover_proofs
            ) / max(len(number_of_chars_in_prover_proofs), 1)
            average_number_of_chars_in_conjecture = sum(
                number_of_chars_in_conjecture
            ) / max(len(number_of_chars_in_conjecture), 1)

            # Verification error stats
            num_system_errors = sum(1 for v in verification_verdicts if v.system_error)
            verification_times: List[float] = [
                float(v.output.get("verify_time", -100)) for v in verification_verdicts
            ]
            num_timeout_errors = sum(
                1 for t in verification_times if t > (verifier_timeout - 0.5)
            )

            # Timing and throughput (best-effort; prover/verify overlap in pipeline)
            total_prompts = len(all_prompts)
            prover_duration = max(1e-6, (end_prover_time - start_prover_time))
            prover_throughput = total_prompts / prover_duration
            if start_verification_time is not None:
                # End of verification is close to end_prover_time in our implementation (post wait)
                end_verification_time = end_prover_time
                verification_duration = max(
                    1e-6, (end_verification_time - start_verification_time)
                )
                verification_throughput = (
                    len(statement_lean_files) + len(conjecture_lean_files)
                ) / verification_duration
            else:
                verification_duration = 0.0
                verification_throughput = 0.0

            # Log utilization timelines (same namespaces as non-pipelined path)
            log_utilization_report_timings(
                query_util_reports,
                wandb_run,
                f"iteration_{iteration}/data_gen_prover_jobs",
            )
            log_utilization_report_timings(
                verify_util_reports,
                wandb_run,
                f"iteration_{iteration}/data_gen_verification_timings",
            )

            # Log statement/conjecture verification tables (first N) with identical namespaces
            stmt_table = wandb.Table(columns=["lean_file", "verification_output"])
            max_stmt_rows = min(20, len(statement_lean_files))
            stmt_v_outputs = verification_verdicts[: len(statement_lean_files)]
            for i in range(max_stmt_rows):
                if i >= len(stmt_v_outputs):
                    break
                stmt_table.add_data(
                    statement_lean_files[i],
                    json.dumps(stmt_v_outputs[i].output, indent=4),
                )
            wandb_run.log(
                {
                    f"iteration_{iteration}/pipeline_data_samples/step_3_statement_verification_outputs": stmt_table
                }
            )

            conj_table = wandb.Table(columns=["lean_file", "verification_output"])
            max_conj_rows = min(20, len(conjecture_lean_files))
            conj_v_outputs = verification_verdicts[len(statement_lean_files) :]
            for i in range(max_conj_rows):
                if i >= len(conj_v_outputs):
                    break
                conj_table.add_data(
                    conjecture_lean_files[i],
                    json.dumps(conj_v_outputs[i].output, indent=4),
                )
            wandb_run.log(
                {
                    f"iteration_{iteration}/pipeline_data_samples/step_3_conjecture_verification_outputs": conj_table
                }
            )

            # Split proof samples by type to match step_2 tables
            stmt_samples = [
                s for s in query_verify_samples if isinstance(s.source_obj, Statement)
            ]
            conj_samples = [
                s for s in query_verify_samples if isinstance(s.source_obj, Conjecture)
            ]
            stmt_samples_table = wandb.Table(
                columns=["prompt", "response", "extracted_proof"]
            )
            conj_samples_table = wandb.Table(
                columns=["prompt", "response", "extracted_proof"]
            )
            for s in stmt_samples[: min(20, len(stmt_samples))]:
                stmt_samples_table.add_data(
                    s.prompt, s.full_generation or "", s.proof or ""
                )
            for s in conj_samples[: min(20, len(conj_samples))]:
                conj_samples_table.add_data(
                    s.prompt, s.full_generation or "", s.proof or ""
                )
            wandb_run.log(
                {
                    f"iteration_{iteration}/pipeline_data_samples/step_2_statement_proofs": stmt_samples_table
                }
            )
            wandb_run.log(
                {
                    f"iteration_{iteration}/pipeline_data_samples/step_2_conjecture_proofs": conj_samples_table
                }
            )

            # Aggregate scalar logs (match non-pipelined namespaces)
            wandb_run.log(
                {
                    # Timing
                    "timing/sampling_proofs_throughput(generations/sec)": prover_throughput,
                    "timing/sampling_proofs_time(mins)": prover_duration / 60.0,
                    "timing/verification_time(mins)": verification_duration / 60.0,
                    "timing/verification_throughput(proofs/sec)": verification_throughput,
                    # Data Gen
                    "data_gen/num_proof_samples": len(query_results),
                    # Estimating invalid proofs via extraction tag
                    "data_gen/num_invalid_proofs": sum(
                        1
                        for s in query_verify_samples
                        if (s.proof is not None and NO_CODE_FOUND_TAG in s.proof)
                    ),
                    "data_gen/prover_token_counts/average_entropy": average_entropy,
                    # Verification stats
                    "data_gen/num_verification_system_errors": num_system_errors,
                    "data_gen/num_verification_timeout_errors": num_timeout_errors,
                    "data_gen/prop_verification_errors": (
                        num_system_errors
                        / max(1, len(statement_lean_files) + len(conjecture_lean_files))
                    ),
                    "data_gen/prop_verification_timeout_errors": (
                        num_timeout_errors
                        / max(1, len(statement_lean_files) + len(conjecture_lean_files))
                    ),
                    # Summary
                    "data_gen/prover_token_counts/average_number_of_chars_in_prover_proofs": average_number_of_chars_in_prover_proofs,
                    "conjectures/average_number_of_chars_in_conjecture": average_number_of_chars_in_conjecture,
                }
            )
            # Token histograms and averages, same prefix as other path
            log_token_counts(
                wandb_run,
                query_results,
                "data_gen/prover_token_counts",
                iteration=iteration,
                num_generations=None,
            )
    except Exception as e:
        # Logging should never crash the pipeline
        print(f"[pipeline_pv logging] Skipped logging due to error: {e}")

    return target_statements, conjectures
