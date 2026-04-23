"""
step2_train.py

Functions for second step of STP pipeline which involves training:
1. Train prover model
2. Train conjecturer model
3. Evaluate prover on held out set
4. Save models
"""

from typing import List, Dict, Tuple, Optional
import random
import logging
import time
import re
from pathlib import Path
import copy

import wandb
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass
import uuid
import torch
import json

from sgs.pipeline.config import StatementSelectionMode
from sgs.data.dataset_types import (
    ProverDataset,
    ProverIterationData,
    Statement,
    ConjecturerDataset,
    Conjecture,
    StatementTag,
    ConjectureIterationData,
)
from sgs.models.model_types import (
    ConjecturerConfig,
    ResourcesConfig,
    ProverConfig,
    ConjecturerSetup,
    ModelConfig,
)
from sgs.training.training_types import TrainingConfig, TrainingSampleDatum
from sgs.training.train import finetune_model
from sgs.models.guide import Guide
from sgs.utils import SubmititCleanupExecutor
from sgs.utils import chunk_list

import math


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def length_reward(model_config, generation_tokens: List[int]) -> float:
    total_length = len(generation_tokens)

    gamma = 0.8
    start_negative_reward = model_config.max_tokens * gamma

    if total_length <= start_negative_reward:
        return 0.0

    negative_reward_interval = model_config.max_tokens - start_negative_reward
    if negative_reward_interval <= 0:
        return -1.0  # degenerate case

    amount_too_long = total_length - start_negative_reward
    reward = -1 * amount_too_long / negative_reward_interval

    return max(-1.0, reward)

def stp_proof_length_reward(proof_str: str) -> float:
    # This is the stp weighting on correct proofs, which acts on the character level
    return math.exp(-0.001 * len(proof_str))


def prepare_conjecturer_train_data(
    conjecturer_dataset_path: str,
    conjecturer_config: ConjecturerConfig,
    weight_by_review: bool,
    iteration: int,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    iterations_in_buffer: int = 1,  # We set this to 1 as we normally use most recent, if -1 then we use all the data
) -> List[Tuple[str, str, float, List[float], List[int], float, int]]:
    """
    Prepare the train data for the conjecturer

    This is easier than preparing the prover dataset as we simply train on all
    of the conjectures from the last round in the conjecturer dataset.

    Args:
        conjecturer_dataset_path: Path to conjecturer dataset
    """
    conjecturer_dataset = ConjecturerDataset.load(conjecturer_dataset_path)

    iter_nums = [x.iteration for x in conjecturer_dataset.iterations]
    # assert all different
    assert len(iter_nums) == len(set(iter_nums)), "Iterations must be different"

    # Now we must select the proofs to use
    iterations: List[ConjectureIterationData] = sorted(
        conjecturer_dataset.iterations, key=lambda x: x.iteration, reverse=True
    )
    assert (
        iterations[0].iteration >= iterations[-1].iteration
    ), "Iterations must be in reverse order"

    # Get the last `iterations_in_buffer` iterations
    if iterations_in_buffer == -1:
        # Use all the iterations!
        pass
    else:
        iterations = iterations[:iterations_in_buffer]

    conjectures: List[Conjecture] = [
        conjecture for iteration in iterations for conjecture in iteration.iter_data
    ]

    # Review the data if it has reviews
    conjectures_with_reviews: List[Tuple[Conjecture, float]] = []

    solve_rates: List[float] = [conjecture.solve_rate for conjecture in conjectures]  # type: ignore
    for solve_rate in solve_rates:
        assert solve_rate is not None, "All conjectures must have a solve rate"
        assert 0 < solve_rate < 1, "Solve rate must be between 0 and 1 non inclusive"

    if weight_by_review:
        num_conjectures_before_review = len(conjectures)
        # For each conjecture we have a review for each proof
        # So we take the average review over the proofs

        for conjecture in conjectures:
            # Get the average review over the proofs

            assert (
                len(conjecture.proofs) > 0
            ), "All conjectures must have at least one proof"
            for proof in conjecture.proofs:
                assert proof.review is not None, "All proofs must have a review"

            average_review = sum([proof.review for proof in conjecture.proofs]) / len(  # type: ignore
                conjecture.proofs
            )
            conjectures_with_reviews.append((conjecture, average_review))

        reviews = [review for _, review in conjectures_with_reviews]
        average_review = sum(reviews) / max(len(reviews), 1)

        review_rewards = reviews

        if wandb_run is not None:
            wandb_run.log(
                {
                    "conjectures/average_review": average_review,
                    "iteration": iteration,
                }
            )

            # Create a histogram of the reviews
            # Create histogram of reviews
            fig = plt.figure(figsize=(8, 6))
            plt.hist(reviews, bins=50)
            plt.xlabel("Review Score")
            plt.ylabel("Count")
            plt.title("Distribution of Conjecture Review Scores")

            # Log to wandb if available
            wandb_run.log(
                {
                    "conjectures/review_distribution": wandb.Image(fig),
                    "iteration": iteration,
                }
            )

            plt.close()

            wandb_run.log(
                {
                    "conjectures/num_conjectures_before_review": num_conjectures_before_review,
                    "iteration": iteration,
                }
            )
    else:
        review_rewards = [1.0] * len(conjectures)

    assert len(review_rewards) == len(solve_rates)

    # Now we will get the rewards
    solve_rate_rewards = [1 - solve_rate for solve_rate in solve_rates]
    average_solve_rate_reward = sum(solve_rate_rewards) / max(
        len(solve_rate_rewards), 1
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "conjectures/average_solve_rate_reward": average_solve_rate_reward,
                "iteration": iteration,
            }
        )

    rewards = [
        review_reward * (1 - solve_rate)
        for review_reward, solve_rate in zip(review_rewards, solve_rates)
    ]

    average_reward = sum(rewards) / max(len(rewards), 1)

    if wandb_run is not None:
        wandb_run.log(
            {"conjectures/average_reward": average_reward, "iteration": iteration}
        )

    # Linearly project the reviews to the range [0, 1]
    min_reward = min(rewards)
    max_reward = max(rewards)
    if max_reward == min_reward:
        normalized_rewards = [1.0 for _ in rewards]
    else:
        normalized_rewards = [
            (reward - min_reward) / (max_reward - min_reward) for reward in rewards
        ]

    # --------- Take a break to do some logging --------- #

    if wandb_run is not None:
        fig = plt.figure(figsize=(8, 6))
        plt.hist(solve_rates, bins=50)
        plt.xlabel("Solve Rates")
        plt.ylabel("Count")
        plt.title("Distribution of Conjecture Solve Rates")
        wandb_run.log(
            {
                "conjectures/solve_rate_distribution": wandb.Image(fig),
                "iteration": iteration,
            }
        )

        plt.close()
        fig = plt.figure(figsize=(8, 6))
        plt.hist(rewards, bins=20)
        plt.xlabel("Rewards")
        plt.ylabel("Count")
        plt.title("Distribution of Conjecture Rewards")
        wandb_run.log(
            {
                "conjectures/reward_distribution": wandb.Image(fig),
                "iteration": iteration,
            }
        )
        plt.close()

        fig = plt.figure(figsize=(8, 6))
        plt.hist(normalized_rewards, bins=20)
        plt.xlabel("Normalized Rewards (0-1)")
        plt.ylabel("Count")
        plt.title("Distribution of Normalized Conjecture Rewards (0-1)")
        wandb_run.log(
            {
                "conjectures/normalized_reward_distribution": wandb.Image(fig),
                "iteration": iteration,
            }
        )
        plt.close()

    # --------------------------------------------------- #

    train_data: List[Tuple[str, str, float, List[float], List[int]]] = []

    assert len(conjectures) == len(
        normalized_rewards
    ), "Number of conjectures and normalized rewards must match"

    for conjecture, weight in zip(conjectures, normalized_rewards):
        assert any(
            proof.is_correct for proof in conjecture.proofs
        ), "All conjectures must have at least one correct proof"

        if conjecturer_config.setup == ConjecturerSetup.SEED_STATEMENT:
            assert conjecture.seed_proof is not None
            prompt = conjecturer_config.prompt_getter(
                seed_theorem=conjecture.seed_theorem,
                seed_proof=conjecture.seed_proof.proof_str,
                conjecturer_config=conjecturer_config,
            )
        elif conjecturer_config.setup in [
            ConjecturerSetup.TARGET_STATEMENT,
            ConjecturerSetup.TARGET_STATEMENT_ONLY_UNSOLVED,
        ]:
            prompt = conjecturer_config.prompt_getter(
                seed_theorem=conjecture.seed_theorem,
                conjecturer_config=conjecturer_config,
            )
        else:
            raise ValueError(f"Invalid conjecturer setup: {conjecturer_config.setup}")

        target = conjecture.conjecture_full_generation
        target_tokens = conjecture.conjecture_full_generation_tokens
        log_probs: List[float] = conjecture.conjecture_full_generation_logprobs  # type: ignore

        if conjecturer_config.dapo_length_penalty:
            len_reward = length_reward(
                conjecturer_config,
                target_tokens,
            )
            if len_reward < 0.0:
                weight = len_reward

        train_datum = TrainingSampleDatum(
            prompt_str=prompt,
            gen_str=target,
            reward=weight,
            advantage=float("nan"),   # For conjecturer, we do not calculate advantages for now
            log_prob_over_gen=log_probs,
            gen_tokens=target_tokens,
            group_id=str(uuid.uuid4())[:4],
            num_tokens_in_group=len(target_tokens),
        )

        train_data.append(train_datum)

    # Shuffle the data
    random.shuffle(train_data)

    return train_data


def prepare_prover_train_data(
    prover_dataset_path: str,
    iterations_in_buffer: int,  # If -1 then we use all the data
    model_config: ProverConfig,
    num_train_examples: Optional[int] = None,
    statement_selection_mode: Optional[StatementSelectionMode] = None,
    conjecture_multiplier: Optional[int] = None,
    using_groups: bool = False,
) -> Tuple[List[Tuple[str, str, float, List[float], List[int]]], Dict]:
    """
    Prepare the train data for the prover

    Args:
        prover_dataset_path: Path to dataset we use to train the prover so far
        iterations_in_buffer: Number of iterations to train on
        num_train_examples: Number of examples to train on (can have repeat statements with different proofs)
            If None then we use all of the data that we have
        conjecture_multiplier: Multiplier for number of conjectures to statements in final dataset
            If None then we use all of the data that we have
    """

    logger.info("=" * 100)
    logger.info("Preparing prover train data")
    logger.info(f"Loading prover dataset from {prover_dataset_path}")
    logger.info("=" * 100)
    prover_dataset = ProverDataset.load(prover_dataset_path)

    to_log = {}

    iter_nums = [x.iteration for x in prover_dataset.iterations]
    # assert all different
    assert len(iter_nums) == len(set(iter_nums)), "Iterations must be different"

    # Now we must select the proofs to use
    iterations: List[ProverIterationData] = sorted(
        prover_dataset.iterations, key=lambda x: x.iteration, reverse=True
    )
    assert (
        iterations[0].iteration >= iterations[-1].iteration
    ), "Iterations must be in reverse order"

    if iterations_in_buffer == -1:
        # We use all the data!
        pass
    else:
        # Get the last `iterations_in_buffer` iterations
        iterations = iterations[:iterations_in_buffer]

    # Combine to get all statements
    target_statements: List[Statement] = [
        statement
        for iteration in iterations
        for statement in iteration.iter_data
        if statement.tag == StatementTag.TARGET.value
    ]
    



    # Random shuffle statements
    random.shuffle(target_statements)

    chosen_target_statements: Dict[str, Statement] = {}
    # Go through all the statements and add all of the proofs
    for statement in target_statements:
        if statement.theorem in chosen_target_statements:
            # Add proofs to the existing statement
            existing_statement = chosen_target_statements[statement.theorem]
            # Dedup
            existing_statement.proofs = list(
                set(existing_statement.proofs + statement.proofs)
            )
        else:
            chosen_target_statements[statement.theorem] = statement

    # Then go through and prune one by one. The problems with the most proofs get pruned first
    if num_train_examples is not None:
        chosen_target_statements = prune_dict(
            chosen_target_statements, num_train_examples
        )

    # Now we must select the conjectures to use
    conjecture_statements: List[Statement] = [
        statement
        for iteration in iterations
        for statement in iteration.iter_data
        if statement.tag == StatementTag.CONJECTURE.value
    ]
    chosen_conjectures: Dict[str, Statement] = {}
    for statement in conjecture_statements:
        if statement.theorem in chosen_conjectures:
            # Add proofs to the existing statement
            existing_statement = chosen_conjectures[statement.theorem]
            # Dedup
            existing_statement.proofs = list(
                set(existing_statement.proofs + statement.proofs)
            )
        else:
            chosen_conjectures[statement.theorem] = statement

    # Now combine all the values into a single list
    train_statements = list(chosen_target_statements.values()) + list(
        chosen_conjectures.values()
    )

    if statement_selection_mode == StatementSelectionMode.LESS_16_PROOFS:
        # We do something a bit different here

        assert not using_groups, "Using groups is not supported with statement selection mode LESS_16_PROOFS"

        statements_that_were_conjectured = [
            statement for statement in train_statements if statement.tag == StatementTag.CONJECTURE.value
        ]

        train_statements = []
        # now we will actually get the target statements from the last three iterations
        current_iteration = iterations[0].iteration

        num_proofs_in_this_iteration = 0
        num_proofs_in_prev_iterations = 0
        num_proofs_in_prev_prev_iterations = 0
        for statement in prover_dataset.target_statements.values():

            valid_proofs = [x for x in statement.proofs if x.is_correct]
            if len(valid_proofs) >= 0:
                # Create a copy of the statement
                new_statement = copy.deepcopy(statement)
                proofs_in_last_three_iterations = [x for x in valid_proofs if x.iteration_created > current_iteration - 3]

                for proof in proofs_in_last_three_iterations:
                    if proof.iteration_created == current_iteration:
                        num_proofs_in_this_iteration += 1
                    elif proof.iteration_created == current_iteration - 1:
                        num_proofs_in_prev_iterations += 1
                    elif proof.iteration_created == current_iteration - 2:
                        num_proofs_in_prev_prev_iterations += 1

                if len(proofs_in_last_three_iterations) > 0:
                    new_statement.proofs = proofs_in_last_three_iterations
                    train_statements.append(new_statement)

        train_statements += statements_that_were_conjectured

        to_log = to_log | {
            "data_gen/K2_num_proofs_in_this_iteration": num_proofs_in_this_iteration,
            "data_gen/K2_num_proofs_in_prev_iterations": num_proofs_in_prev_iterations,
            "data_gen/K2_num_proofs_in_prev_prev_iterations": num_proofs_in_prev_prev_iterations,
        }
    

    logging.info("-" * 100)
    logging.info(
        f"Number of target statements proof pairs in prover training set: {len(chosen_target_statements)}"
    )
    logging.info(
        f"Number of conjectures proof pairs in prover training set: {len(chosen_conjectures)}"
    )
    logging.info("-" * 100)


    # And shuffle
    random.shuffle(train_statements)

    grouped_train_data: List[List[Tuple[str, str, float, List[float], List[int]]]] = []

    num_proofs_with_try = 0
    num_proofs_under_1000_chars = 0
    total_num_proofs = 0
    num_length_penalized = 0
    num_double_lean_penalized = 0

    num_train_statements = len(train_statements)
    for statement in train_statements:
        group_train_data: List[Tuple[str, str, float, List[float], List[int]]] = []

        group_id = str(uuid.uuid4())[:4]
        for proof in statement.proofs:
            if not using_groups:
                if not proof.is_correct:
                    # There is nothing to learn from incorrect proofs so continue
                    continue

            if proof.is_correct:
                weight = 1.0  # Correct proofs get positive weight
            else:
                weight = 0.0

            prompt = model_config.prompt_getter(
                header=statement.header, theorem=statement.theorem
            )
            target = proof.full_generation

            total_num_proofs += 1

            # Now lets see how long that wretched proof string is
            proof_str = proof.proof_str
            # Need to strip out the comments?
            # Coments start with \-- and end with -\
            proof_str = re.sub(r"/-[\s\S]*?-/", "", proof_str)
            proof_str = re.sub(r"--.*", "", proof_str)

            proof_len = len(proof_str)

            if model_config.penalize_long_proof_str_over_1000:

                if proof_len > 1000:
                    weight = 0.0

                    if not using_groups:
                        continue


            if proof_len <= 1000:
                num_proofs_under_1000_chars += 1


            if model_config.penalize_try:
                if "try" in proof_str:
                    # We do not want the model to get into habbit of always using try
                    num_proofs_with_try += 1
                    weight = 0.0

                    if not using_groups:
                        continue

            if model_config.dapo_length_penalty:
                len_reward = length_reward(
                    model_config, 
                    proof.full_generation_tokens, 
                )

                if len_reward < 0.0:
                    # We are too long
                    weight = len_reward
                    num_length_penalized += 1

            if model_config.stp_length_penalty:
                full_proof_str = proof.proof_str
                stp_len_weight = stp_proof_length_reward(full_proof_str)

                if weight == 1.0:
                    # This means so far the proof is correct, so we will reduce to weight to stp_len_weight
                    weight = stp_len_weight


            if model_config.penalize_double_lean:
                num_lean4_in_target = target.count("```lean4")
                if num_lean4_in_target > 1:
                    num_double_lean_penalized += 1
                    weight = 0.0

                    if not using_groups:
                        continue

            group_train_data.append(
                {
                    "prompt_str": prompt,
                    "gen_str": target,
                    "reward": weight,
                    "log_prob_over_gen": proof.full_generation_logprobs,
                    "gen_tokens": proof.full_generation_tokens,
                    "group_id": group_id,
                }
            )

        group_train_datums = []

        # log the length penalized

        if all(x["reward"] == 0.0 for x in group_train_data):
            # Don't add, when we got rid of tries and too long, this problem has no correct proofs
            continue

        rewards = [x["reward"] for x in group_train_data]
        average_reward = sum(rewards) / len(rewards)
        std_reward = torch.tensor(rewards).std(unbiased=False).item()

        if using_groups:
            advantages = [(x["reward"] - average_reward) / (std_reward + 1e-6) for x in group_train_data]
        else:
            # If we are not using groups, then we do not have a notion of advantages
            advantages = [x["reward"] for x in group_train_data]

        tokens_in_group = sum([len(x["gen_tokens"]) for x in group_train_data])


        for i, traj_dict in enumerate(group_train_data):
            train_datum = TrainingSampleDatum(
                prompt_str=traj_dict["prompt_str"],
                gen_str=traj_dict["gen_str"],
                reward=traj_dict["reward"],
                advantage=advantages[i],
                log_prob_over_gen=traj_dict["log_prob_over_gen"],
                gen_tokens=traj_dict["gen_tokens"],
                group_id=group_id,
                num_tokens_in_group=tokens_in_group,
            )
            group_train_datums.append(train_datum)


        grouped_train_data.append(group_train_datums)

    # Shuffle the data
    random.shuffle(grouped_train_data)

    if using_groups:
        # Need to check that all the groups are the same size
        group_lengths = [len(group) for group in grouped_train_data]
        assert (
            len(set(group_lengths)) == 1
        ), f"All groups must have the same length, we got {set(group_lengths)}"

    # Now flatten the grouped train data
    train_data: List[TrainingSampleDatum] = []
    for group in grouped_train_data:
        train_data.extend(group)

    if not using_groups:
        # We don't need to preserve the grouped structure
        random.shuffle(train_data)

    if total_num_proofs == 0:
        percent_proofs_with_try = 0.0
        percent_num_proofs_under_1000_chars = 0.0
        percent_num_proofs_pass_pruning = 0.0
        percent_length_penalized = 0.0
        percent_double_lean_penalized = 0.0
    else:
        percent_proofs_with_try = num_proofs_with_try / total_num_proofs
        percent_num_proofs_under_1000_chars = num_proofs_under_1000_chars / total_num_proofs
        percent_num_proofs_pass_pruning = len(train_data) / total_num_proofs
        percent_length_penalized = num_length_penalized / total_num_proofs
        percent_double_lean_penalized = num_double_lean_penalized / total_num_proofs


    to_log = to_log | {
        "data_gen/num_proofs_with_try": num_proofs_with_try,
        "data_gen/percent_proofs_with_try": percent_proofs_with_try,
        "data_gen/percent_num_proofs_under_1000_chars": percent_num_proofs_under_1000_chars,
        "data_gen/num_proofs_pre_pruning": total_num_proofs,
        "data_gen/percent_num_proofs_pass_pruning": percent_num_proofs_pass_pruning,
        "data_gen/num_length_penalized": num_length_penalized,
        "data_gen/percent_length_penalized": percent_length_penalized,
        "data_gen/number_of_prover_train_samples_in_prepare_prover_train_data" : len(train_data),
        "data_gen/number_of_prover_statements_in_prepare_prover_train_data" : num_train_statements,
    }

    if model_config.penalize_double_lean:
        to_log["data_gen/num_double_lean_penalized"] = num_double_lean_penalized
        to_log["data_gen/percent_double_lean_penalized"] = percent_double_lean_penalized


    return train_data, to_log


def prune_dict(d: Dict[str, Statement], num_examples: int) -> Dict[str, Statement]:
    total_count = sum(len(x.proofs) for x in d.values())
    while total_count > num_examples:
        # Find the statement with the most proofs
        statement = max(d.values(), key=lambda x: len(x.proofs))
        # Prune one proof
        statement.proofs = statement.proofs[:-1]
        # If we have no proofs left, remove the statement
        if len(statement.proofs) == 0:
            del d[statement.theorem]
        total_count -= 1

    return d


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Main train function for step 2
# ------------------------------------------------------------


def train_prover_and_conjecturer(
    iteration: int,
    # Prover information
    prover_dataset_path: str,
    prover_config: ProverConfig,
    prover_iterations_in_buffer: int,
    conjecture_multiplier: int | None,
    num_prover_train_examples: int | None,
    # Conjecturer information
    conjecturer_dataset_path: str,
    conjecturer_config: ConjecturerConfig,
    conjecturer_iterations_in_buffer: int,
    # Shared information
    prover_model_save_path: str,
    conjecturer_model_save_path: str,
    training_config: TrainingConfig,
    resources_config: ResourcesConfig,
    checkpoint_dir: str,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    stp_round: Optional[int] = None,
    parameter_sharing: bool = True,
    freeze_prover: bool = False,
    freeze_conjecturer: bool = False,
    statement_selection_mode: Optional[StatementSelectionMode] = None,
    # guide specific
    guide: Optional[Guide] = None,
    guide_resources_config: Optional[ResourcesConfig] = None,
) -> float:
    """
    Train the prover and conjecturer as a single model
    by mixing the prover and conjecturer datasets together
    """

    if parameter_sharing:
        # Do some sanity checks
        assert (
            prover_config.model_name == conjecturer_config.model_name
        ), "Prover and conjecturer must use the same model"
        assert (
            prover_config.model_type == conjecturer_config.model_type
        ), "Prover and conjecturer must use the same model type"
        assert (
            prover_config.dtype == conjecturer_config.dtype
        ), "Prover and conjecturer must use the same dtype"
        assert (
            prover_model_save_path == conjecturer_model_save_path
        ), "Prover and conjecturer must use the same model save path if parameter sharing is enabled"
        assert (
            training_config.prover_trainer_cls
            == training_config.conjecturer_trainer_cls
            and training_config.prover_trainer_cls == "WeightedTrainer"
        ), "Prover and conjecturer must use the same trainer class if parameter sharing is enabled, and must be WeightedTrainer"

    conjectures: ConjecturerDataset = ConjecturerDataset.load(conjecturer_dataset_path)
    conjectures_to_review: List[Conjecture] = conjectures.iterations[-1].iter_data

    if len(conjectures.iterations) == 0:
        logger.info("No conjectures to review, skipping review")

    start_review_time = time.time()

    cost: float = 0.0
    if guide is not None and len(conjectures_to_review) > 0:
        assert (
            guide_resources_config is not None
        ), "Guide resources config must be provided if guide is provided"

        # We load up the conjectures, review them, and then save them

        logger.info("=" * 100)
        logger.info("Reviewing conjectures before training prover and conjecturer")
        logger.info(
            f"Reviewing with model from path: {guide.guide_config.prover_config.model_name}"
        )
        logger.info(f"Number of conjectures to review: {len(conjectures_to_review)}")
        logger.info("=" * 100)

        # This will load up the conjectures and add review scores, then save to the same path
        reviewed_conjectures: List[Conjecture]
        reviewed_conjectures, cost = review_conjectures(
            conjectures=conjectures_to_review,
            guide=guide,
            resources_config=guide_resources_config,
            wandb_run=wandb_run,
        )

        conjectures.iterations[-1].iter_data = reviewed_conjectures
        conjectures.save(conjecturer_dataset_path)

    end_review_time = time.time()
    # Log review time
    logger.info(f"Review time: {end_review_time - start_review_time} seconds")
    if wandb_run is not None:
        wandb_run.log(
            {
                "timing/review_time(mins)": (end_review_time - start_review_time) / 60,
                "iteration": iteration,
            }
        )

    prover_using_groups = (
        training_config.prover_trainer_cls == "GroupedImportanceSampledWeightedTrainer"
    )

    prover_train_data: List[TrainingSampleDatum]
    prover_train_data, data_metrics_to_log = prepare_prover_train_data(
        prover_dataset_path=prover_dataset_path,
        iterations_in_buffer=prover_iterations_in_buffer,
        num_train_examples=num_prover_train_examples,
        conjecture_multiplier=conjecture_multiplier,
        statement_selection_mode=statement_selection_mode,
        model_config=prover_config,
        using_groups=prover_using_groups,
    )

    if len(conjectures.iterations[-1].iter_data) > 0:
        conjecturer_train_data: List[TrainingSampleDatum] = (
            prepare_conjecturer_train_data(
                conjecturer_dataset_path=conjecturer_dataset_path,
                conjecturer_config=conjecturer_config,
                weight_by_review=guide is not None,
                wandb_run=wandb_run,
                iterations_in_buffer=conjecturer_iterations_in_buffer,
                iteration=iteration,
            )
        )
    else:
        logger.info("No conjectures to review, skipping review")
        conjecturer_train_data = []

    logger.info("=" * 100)
    logger.info("Training data prepared for prover and conjecturer")
    logger.info(f"Prover train data: {len(prover_train_data)} examples")
    logger.info(f"Conjecturer train data: {len(conjecturer_train_data)} examples")
    logger.info(f"Conjecturer model load path: {conjecturer_config.model_name}")
    logger.info(f"Prover model load path: {prover_config.model_name}")
    logger.info(f"Conjecturer model save path: {conjecturer_model_save_path}")
    logger.info(f"Prover model save path: {prover_model_save_path}")
    logger.info("=" * 100)

    if wandb_run is not None:
        wandb_run.log(
            {
                "train_prover_and_conjecturer/num_prover_train_data": len(
                    prover_train_data
                ),
                "train_prover_and_conjecturer/num_conjecturer_train_data": len(
                    conjecturer_train_data
                ),
                "iteration": iteration,
            }
            | data_metrics_to_log
        )

    start_train_time = time.time()

    # Train the prover

    if prover_using_groups and parameter_sharing:
        raise ValueError(
            "Using groups in training and parameter sharing is not supported."
        )

    if parameter_sharing:
        if (
            training_config.prover_trainer_cls
            != training_config.conjecturer_trainer_cls
        ):
            raise ValueError(
                "Prover and conjecturer must use the same trainer class if parameter sharing is enabled."
            )

    if parameter_sharing:
        if stp_round is not None:
            wandb_log_prefix = f"train_round_{stp_round}_prover_and_conjecturer"
        else:
            wandb_log_prefix = "train_prover_and_conjecturer_train"

        # Now we randomly shuffle the data together
        train_data: List[TrainingSampleDatum] = (
            prover_train_data + conjecturer_train_data
        )

        # We can shuffle the data here as we do not care about preserving the prover dataset grouped structure
        random.shuffle(train_data)

        finetune_model(
            group_size=training_config.prover_group_size,
            trainer_cls=training_config.prover_trainer_cls,
            model_save_path=prover_model_save_path,
            train_data=train_data,
            training_config=training_config,
            resources_config=resources_config,
            model_config=prover_config,  # Prover config and conjecturer config the same for purposes of finetune_model
            wandb_run=wandb_run,
            wandb_log_prefix=wandb_log_prefix,
            iteration=iteration,
        )

    else:
        # Train the prover and conjecturer separately
        if stp_round is not None:
            prover_wandb_log_prefix = f"train_round_{stp_round}_prover"
            conjecturer_wandb_log_prefix = f"train_round_{stp_round}_conjecturer"
        else:
            prover_wandb_log_prefix = "train_prover"
            conjecturer_wandb_log_prefix = "train_conjecturer"

        # Train the prover
        if not freeze_prover:

            # Set learning rate to prover learning rate
            if training_config.prover_learning_rate is not None:
                training_config.learning_rate = training_config.prover_learning_rate

            finetune_model(
                group_size=training_config.prover_group_size,
                trainer_cls=training_config.prover_trainer_cls,
                model_save_path=prover_model_save_path,
                train_data=prover_train_data,
                training_config=training_config,
                resources_config=resources_config,
                model_config=prover_config,
                wandb_run=wandb_run,
                wandb_log_prefix=prover_wandb_log_prefix,
                iteration=iteration,
            )

        # Train the conjectuer
        random.shuffle(conjecturer_train_data)
        if not freeze_conjecturer:

            # Set learning rate to conjecturer learning rate
            if training_config.conjecturer_learning_rate is not None:
                training_config.learning_rate = training_config.conjecturer_learning_rate

            finetune_model(
                group_size=training_config.conjecturer_group_size,
                trainer_cls=training_config.conjecturer_trainer_cls,
                model_save_path=conjecturer_model_save_path,
                train_data=conjecturer_train_data,
                training_config=training_config,
                resources_config=resources_config,
                model_config=conjecturer_config,
                wandb_run=wandb_run,
                wandb_log_prefix=conjecturer_wandb_log_prefix,
                iteration=iteration,
            )

    end_train_time = time.time()
    logger.info(f"Train time: {end_train_time - start_train_time} seconds")
    if wandb_run is not None:
        wandb_run.log(
            {
                "timing/train_time(mins)": (end_train_time - start_train_time) / 60,
                "iteration": iteration,
            }
        )

    return cost


def review_conjectures(
    conjectures: List[Conjecture],
    guide: Guide,
    resources_config: ResourcesConfig,
    _debug: bool = False,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    iteration: Optional[int] = None,
    wandb_log_prefix: Optional[str] = None,
) -> Tuple[List[Conjecture], float]:
    """
    Review the conjectures
    """


    num_with_review = 0
    for conjecture in conjectures:
        if all(proof.review is not None for proof in conjecture.proofs):
            num_with_review += 1
    
    if num_with_review == len(conjectures):
        logger.info("All conjectures already have reviews, skipping review")
        return conjectures, 0.0

    logger.info("=" * 100)
    logger.info(f"Reviewing {len(conjectures)} conjectures")
    logger.info("=" * 100)

    # Sanity check
    for conjecture in conjectures:
        num_correct = sum(1 for proof in conjecture.proofs if proof.is_correct)
        assert (
            num_correct > 0
        ), "All conjectures must have at least one correct proof for review"

    # Now we need to review this data
    def review_conjectures_local(
        conjectures: List[Conjecture], guide: Guide
    ) -> Tuple[List[Conjecture], List[Dict]]:
        to_log = guide.review(conjectures)
        # This will cause inplace change to conjectures, which we return
        return conjectures, to_log

    if resources_config.submitit:
        with SubmititCleanupExecutor(resources_config=resources_config) as executor:
            # We can consider spitting into different jobs here
            if resources_config.num_jobs > 1:
                # We chunk the data and launch many jobs
                # We chunk the data into resources_config.num_jobs chunks
                conjectures_chunks: List[List[Conjecture]] = chunk_list(
                    conjectures, resources_config.num_jobs
                )

                # We launch many jobs
                jobs = []
                for conjectures_chunk in conjectures_chunks:
                    job = executor.submit(
                        review_conjectures_local,
                        conjectures=conjectures_chunk,
                        guide=guide,
                    )
                    jobs.append(job)

                # We wait for all the jobs to finish
                return_conjectures: List[Conjecture] = []
                to_log: List[Dict] = []
                for job in jobs:
                    conjectures, log_data = job.result()  # type: ignore
                    return_conjectures.extend(conjectures)
                    to_log.extend(log_data)
            else:
                # We submitit call here
                job = executor.submit(
                    review_conjectures_local, conjectures=conjectures, guide=guide
                )

                if _debug:
                    print(dir(job))

                return_conjectures, to_log = job.result()  # type: ignore

    else:
        # Run locally
        return_conjectures, to_log = review_conjectures_local(conjectures, guide)

    # Extract the cost from the log data
    # This is rather janky, but we will do it for now

    found_cost = False
    for log_item in to_log:
        if "guide/cost" in log_item:
            assert not found_cost, "Found multiple cost items in log data"
            cost = log_item["guide/cost"]
            found_cost = True
    if not found_cost:
        cost = 0.0

    if wandb_run is not None:
        if wandb_log_prefix is None:
            wandb_log_prefix = "guide"
        for log_item in to_log:
            for key, value in log_item.items():
                if isinstance(value, Figure):
                    if iteration is not None:
                        wandb_run.log(
                            {
                                f"{wandb_log_prefix}/{key}": wandb.Image(value),
                                "iteration": iteration,
                            }
                        )
                    else:
                        wandb_run.log({f"{wandb_log_prefix}/{key}": wandb.Image(value)})
                else:
                    if iteration is not None:
                        wandb_run.log(
                            {f"{wandb_log_prefix}/{key}": value, "iteration": iteration}
                        )
                    else:
                        wandb_run.log({f"{wandb_log_prefix}/{key}": value})

    return return_conjectures, cost
