"""
Classes that support llm as judge guide
"""

from abc import abstractmethod
from typing import List, Dict, Any
import copy
import torch

import wandb

from sgs.models.guide import Guide
from sgs.data.dataset_types import Conjecture
from sgs.models.query import query_model_batch, QueryResult
from sgs.models.model_types import ResourcesConfig, ModelConfig, ModelType
from sgs.utils.prompts import (
    NO_GUIDE_SCORE_FOUND_TAG,
    NO_REDUNDANCY_SCORE_FOUND_TAG,
    NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG,
    NO_RELEVANCE_SCORE_FOUND_TAG,
    extract_guide_redundancy_score,
    extract_guide_conclusion_complexity_score,
    extract_guide_relevance_score,
    get_guide_prompt,
)


class LLMJudgeGuide(Guide):
    """
    These guides just use c, c_p, and x to make a review prediction
    """

    @abstractmethod
    def get_llm_judge_config(self) -> ModelConfig: ...

    @abstractmethod
    def get_model_guide_prompt(
        self,
        conjecture_prompt: str,
        statement_prompt: str,
    ) -> str: ...

    @abstractmethod
    def get_review_from_response(
        self,
        response: str,
    ) -> float: ...

    def get_extra_log_data_from_response(
        self,
        responses: List[str],
    ) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def get_query_resource_config(
        self,
    ) -> ResourcesConfig: ...

    def review(self, conjectures: List[Conjecture]) -> List[Dict]:
        conjecture_prompts: List[str] = []
        statement_prompts: List[str] = []

        for conjecture in conjectures:
            statement_prompt: str = conjecture.seed_theorem
            conjecture_prompt: str = conjecture.conjecture

            assert any(
                x.is_correct for x in conjecture.proofs
            ), "At least one proof of conjecture must be correct for review"

            conjecture_prompts.append(conjecture_prompt)
            statement_prompts.append(statement_prompt)

        guide_prompts = [
            self.get_model_guide_prompt(
                conjecture_prompt=conjecture_prompt,
                statement_prompt=statement_prompt,
            )
            for conjecture_prompt, statement_prompt in zip(
                conjecture_prompts, statement_prompts
            )
        ]

        responses: List[QueryResult]
        responses, _ = query_model_batch(
            prompts=guide_prompts,
            model_config=self.get_llm_judge_config(),
            resources_config=self.get_query_resource_config(),
        )

        # We will need to log costs here
        reviews = [self.get_review_from_response(x.response_text) for x in responses]
        reviews_cot = [x.response_text for x in responses]

        extra_log_data: Dict[str, Any] = self.get_extra_log_data_from_response(
            [x.response_text for x in responses]
        )

        # Count number of failed reviews
        num_failed_reviews = sum(1 for x in reviews if x == NO_GUIDE_SCORE_FOUND_TAG)

        non_failed_reviews = [x for x in reviews if x != NO_GUIDE_SCORE_FOUND_TAG]

        if len(non_failed_reviews) == 0:
            average_review = 0.0
        else:
            average_review = sum(non_failed_reviews) / len(non_failed_reviews)
        # Replace failed reviews with average
        for i in range(len(reviews)):
            if reviews[i] == NO_GUIDE_SCORE_FOUND_TAG:
                reviews[i] = average_review

        # Now apply the reviews to the conjectures
        assert len(reviews) == len(conjectures)

        for review, review_cot, conjecture in zip(reviews, reviews_cot, conjectures):
            for proof in conjecture.proofs:
                proof.review = review
                proof.review_cot = review_cot

        # Now lets sort out the logging data
        # We will create a table of some of the reviews
        table = wandb.Table(
            columns=[
                "conjecture",
                "statement",
                "prompt",
                "response",
                "extracted_review",
            ]
        )
        num_added = 0
        for conjecture, review, prompt, response in zip(
            conjectures, reviews, guide_prompts, responses
        ):
            response_text = response.response_text
            table.add_data(
                conjecture.conjecture,
                conjecture.seed_theorem,
                prompt,
                response_text,
                review,
            )
            num_added += 1
            if num_added > 20:
                break

        to_log: List[Dict] = []
        to_log.append(extra_log_data)
        to_log.append(
            {
                "guide/num_failed_reviews": num_failed_reviews,
            }
        )
        to_log.append(
            {
                "guide/example_reviews": copy.deepcopy(table),
            }
        )

        total_cost = sum(x.cost for x in responses)
        to_log.append(
            {
                "guide/cost": total_cost,
            }
        )

        return to_log


def sub_scores_to_review(
    relevance_score: float,
    complexity_score: float,
    redundancy_score: float,
) -> float:
    combined_score = relevance_score

    if complexity_score in [3, 4]:
        # If you have a complex conclusion, you automatically get a 0 score
        combined_score = 0
    else:
        # conclusion complexity is bad, so we take this away. Baseline is it
        complexity_bonus = 2 - complexity_score
        redundancy_bonus = 1 - redundancy_score

        combined_score = max(0, combined_score + complexity_bonus + redundancy_bonus)

    return combined_score


class DeepseekProverV2LemmaGuideLocal(LLMJudgeGuide):
    def get_llm_judge_config(self) -> ModelConfig:
        num_gpus = torch.cuda.device_count()
        assert (
            num_gpus > 0
        ), "This guide is hard coded to do inference on the local machine, and we detect there are no GPUs available"

        if self.guide_config.guide_model_path is not None:
            model_name = self.guide_config.guide_model_path
            print(f"Using guide model path: {model_name}")
        else:
            model_name = "deepseek-ai/DeepSeek-Prover-V2-7B"

        return ModelConfig(
            model_name=model_name,
            chat=True,
            max_tokens=8192,
            prompt_getter=str,
            output_extractor=str,
            model_type=ModelType.LOCAL,
        )

    def get_query_resource_config(self) -> ResourcesConfig:
        return ResourcesConfig(
            submitit=False,
            log_dir="./tests/test_logs",
            cpus_per_task=10,
            num_jobs=1,
        )

    def get_model_guide_prompt(
        self, conjecture_prompt: str, statement_prompt: str
    ) -> str:
        return get_guide_prompt(
            seed_theorem=statement_prompt,
            conjecture=conjecture_prompt,
        )

    def get_review_from_response(self, response: str) -> float:
        relevance_score = extract_guide_relevance_score(response)
        redundancy_score = extract_guide_redundancy_score(response)
        conclusion_complexity_score = (
            extract_guide_conclusion_complexity_score(response)
        )

        if (
            relevance_score == NO_RELEVANCE_SCORE_FOUND_TAG
            or redundancy_score == NO_REDUNDANCY_SCORE_FOUND_TAG
            or conclusion_complexity_score == NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG
        ):
            return NO_GUIDE_SCORE_FOUND_TAG

        return sub_scores_to_review(
            relevance_score=relevance_score,
            complexity_score=conclusion_complexity_score,
            redundancy_score=redundancy_score,
        )

    def get_extra_log_data_from_response(
        self,
        responses: List[str],
    ) -> Dict[str, Any]:
        relevance_scores = [
            extract_guide_relevance_score(x) for x in responses
        ]
        redundancy_scores = [
            extract_guide_redundancy_score(x) for x in responses
        ]
        conclusion_complexity_scores = [
            extract_guide_conclusion_complexity_score(x) for x in responses
        ]

        num_no_relevance_scores = sum(
            1 for x in relevance_scores if x == NO_RELEVANCE_SCORE_FOUND_TAG
        )
        num_no_redundancy_scores = sum(
            1 for x in redundancy_scores if x == NO_REDUNDANCY_SCORE_FOUND_TAG
        )
        num_no_conclusion_complexity_scores = sum(
            1
            for x in conclusion_complexity_scores
            if x == NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG
        )

        relevance_scores = [
            x for x in relevance_scores if x != NO_RELEVANCE_SCORE_FOUND_TAG
        ]
        redundancy_scores = [
            x for x in redundancy_scores if x != NO_REDUNDANCY_SCORE_FOUND_TAG
        ]
        conclusion_complexity_scores = [
            x
            for x in conclusion_complexity_scores
            if x != NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG
        ]

        average_relevance_score = sum(relevance_scores) / max(len(relevance_scores), 1)
        average_redundancy_score = sum(redundancy_scores) / max(
            len(redundancy_scores), 1
        )
        average_conclusion_complexity_score = sum(conclusion_complexity_scores) / max(
            len(conclusion_complexity_scores), 1
        )

        return {
            "guide/average_relevance_score": average_relevance_score,
            "guide/average_redundancy_score": average_redundancy_score,
            "guide/average_conclusion_complexity_score": average_conclusion_complexity_score,
            "guide/num_no_relevance_scores": num_no_relevance_scores,
            "guide/num_no_redundancy_scores": num_no_redundancy_scores,
            "guide/num_no_conclusion_complexity_scores": num_no_conclusion_complexity_scores,
        }
