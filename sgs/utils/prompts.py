"""
prompts.py

Helper functions to get prompts for models.
"""

from typing import Optional

from sgs.models.model_types import ConjecturerConfig, ConjecturerSetup


NO_REDUNDANCY_SCORE_FOUND_TAG = -1111
NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG = -1112
NO_RELEVANCE_SCORE_FOUND_TAG = -1113



def get_deepseek_prover_v2_prompt(*, header: str, theorem: str) -> str:
    prompt = f"""
Complete the following Lean 4 code:

```lean4
{header + theorem}
```
""".strip()

    return prompt


def get_deepseek_prover_v2_conjecturer_no_grounding_prompt(
    *,
    seed_theorem: str,
    conjecturer_config: ConjecturerConfig,
    seed_proof: Optional[str] = None,
) -> str:
    prompt = (
        "Please generate a Lean4 theorem statement. The theorem should be non-trivial, but not too difficult to prove. "
        "Choose a random area of mathematics for the theorem, such as algebra, number theory, combinatorics, geometry, calculus, or logic. "
        "The theorem should be interesting and mathematically meaningful, not just a very easy identity or direct restatement of a definition.\n"
        "Output the final theorem as a syntactically correct Lean4 theorem statement between ```lean4 and ``` tags. "
        "The final thing you output should be the theorem statement WITHOUT any proof, and just put `sorry` for the proof."
    )

    return prompt


def get_deepseek_prover_v2_conjecturer_prompt(
    *,
    seed_theorem: str,
    conjecturer_config: ConjecturerConfig,
    seed_proof: Optional[str] = None,
) -> str:
    post_theorem_prompt = (
        "Please generate a lean4 theorem that is a lemma or related theorem that is useful for proving the above statement. It should be possible to use the lemma or related theorem to help prove the above statement. The lemma or related theorem should be simpler to prove than the target statement."
        "It should NOT be identical to the target statement. It should NOT be equivalent through renaming variables and premises.\n"
        "Output the final theorem as a syntactically correct lean4 theorem statement between ```lean4 and ``` tags. The final thing you output should be the theorem statement WITHOUT any proof, just put 'sorry' for the proof."
    )

    if conjecturer_config.setup == ConjecturerSetup.SEED_STATEMENT:
        assert (
            seed_proof is not None
        ), "seed_proof must be provided if conjecturer_config.setup is SEED_STATEMENT"

        seed_post_theorem_prompt = (
            "Please generate a lean4 theorem that is more complex than the above statement and its proof. "
            "The generated theorem should build on the ideas in the seed statement but be harder to prove. "
            "It should NOT be identical to the seed statement. It should NOT be equivalent through renaming variables and premises.\n"
            "Output the final theorem as a syntactically correct lean4 theorem statement between ```lean4 and ``` tags. The final thing you output should be the theorem statement WITHOUT any proof, just put 'sorry' for the proof."
        )

        prompt = (
            f"Here is a Lean 4 problem statement and its proof:\n"
            f"```lean4\n"
            f"{(seed_theorem + seed_proof).strip()}\n"
            f"```\n"
            f"{seed_post_theorem_prompt}"
        )
    elif conjecturer_config.setup in [
        ConjecturerSetup.TARGET_STATEMENT,
        ConjecturerSetup.TARGET_STATEMENT_ONLY_UNSOLVED,
    ]:
        assert (
            seed_proof is None
        ), "seed_proof should be None if setup is TARGET_STATEMENT"

        prompt = (
            f"Here is a Lean 4 problem statement:\n"
            f"```lean4\n"
            f"{seed_theorem}\n"
            f"```\n"
            f"{post_theorem_prompt}"
        )

    return prompt


# ------------------------------------------------------------
# Prompt output parsing utils
# ------------------------------------------------------------

NO_CODE_FOUND_TAG = "NO CODE FOUND"
NO_CONJECTURE_FOUND_TAG = "NO CONJECTURE FOUND"
NO_GUIDE_SCORE_FOUND_TAG = (
    -1234567890
)  # We set this to a majic number that the model wont generate and type checks


def extract_proof_deepseek_v2(generation: str) -> str:
    # First we split on the final occurence of ```lean4
    split_idx = generation.rfind("```lean4")

    if split_idx == -1:
        return NO_CODE_FOUND_TAG

    # Deepseek prover tends to output the theorem statemnt again and the proof

    statement_and_proof = generation[split_idx + len("```lean4") :]
    # To get the proof we need to split on the first occurence of "by"
    by_idx = statement_and_proof.find(":= by")
    if by_idx == -1:
        return NO_CODE_FOUND_TAG

    proof = statement_and_proof[by_idx + len(":= by") :]

    # Remove the final ``` that the model writes at the end
    final_code_idx = proof.rfind("```")
    if final_code_idx != -1:
        proof = proof[:final_code_idx]

    return proof


def extract_proof_deepseek_v2_strict(generation: str) -> str:
    # First we split on the final occurence of ```lean4

    # First it must start with ```
    if not generation.strip().startswith("```lean4"):
        return NO_CODE_FOUND_TAG

    # It must end with ```
    if not generation.strip().endswith("```"):
        return NO_CODE_FOUND_TAG

    # Now count the number of ```lean4`
    if generation.strip().count("```lean4") != 1:
        return NO_CODE_FOUND_TAG
    if generation.strip().count("```") != 2:
        return NO_CODE_FOUND_TAG

    # Now count the number of times we have "theorem" "lemma" and "def"
    num_theorem = generation.strip().count("theorem ")
    num_lemma = generation.strip().count("lemma ")
    num_def = generation.strip().count("def ")

    # We should have 1 occurence of "theorem" or "lemma" or "def"
    if num_theorem + num_lemma + num_def != 1:
        return NO_CODE_FOUND_TAG

    split_idx = generation.rfind("```lean4")

    if split_idx == -1:
        return NO_CODE_FOUND_TAG

    # Deepseek prover tends to output the theorem statemnt again and the proof

    statement_and_proof = generation[split_idx + len("```lean4") :]
    # To get the proof we need to split on the first occurence of "by"
    by_idx = statement_and_proof.find(":= by")
    if by_idx == -1:
        return NO_CODE_FOUND_TAG

    proof = statement_and_proof[by_idx + len(":= by") :]

    # Remove the final ``` that the model writes at the end
    final_code_idx = proof.rfind("```")
    if final_code_idx != -1:
        proof = proof[:final_code_idx]
    elif final_code_idx == -1:
        return NO_CODE_FOUND_TAG

    return proof


def extract_conjecture_deepseek_v2(generation: str) -> str:
    # Find final occurence of ```lean4
    split_idx = generation.rfind("```lean4")
    if split_idx == -1:
        return NO_CONJECTURE_FOUND_TAG

    proposed_problem = generation[split_idx + len("```lean4") :]

    # Now get everything before the final ```
    final_code_idx = proposed_problem.rfind("```")
    if final_code_idx == -1:
        return NO_CONJECTURE_FOUND_TAG

    proposed_problem = proposed_problem[:final_code_idx]

    # Now we need to find the first occurence of "by"

    # If there is more than one occurence of := by this is a fail
    if proposed_problem.count(":= by") != 1:
        return NO_CONJECTURE_FOUND_TAG
    if proposed_problem.count("sorry") not in [0, 1]:
        return NO_CONJECTURE_FOUND_TAG

    proposed_problem = proposed_problem.strip()

    if (not proposed_problem.endswith("sorry")) and (
        not proposed_problem.endswith(":= by")
    ):
        return NO_CONJECTURE_FOUND_TAG

    by_idx = proposed_problem.find(":= by")

    if by_idx == -1:
        return NO_CONJECTURE_FOUND_TAG

    theorem = proposed_problem[: by_idx + len(":= by")].strip()

    return theorem


# ------------------------------------------------------------
# Guide prompts
# ------------------------------------------------------------


def extract_guide_redundancy_score(generation: str) -> float:
    start_idx = generation.rfind("<begin_redundancy_score>")
    end_idx = generation.rfind("<end_redundancy_score>")

    if start_idx == -1 or end_idx == -1:
        return NO_REDUNDANCY_SCORE_FOUND_TAG

    try:
        score = float(generation[start_idx + len("<begin_redundancy_score>") : end_idx])
    except ValueError:
        return NO_REDUNDANCY_SCORE_FOUND_TAG

    return score


def extract_guide_conclusion_complexity_score(generation: str) -> float:
    start_idx = generation.rfind("<begin_conclusion_complexity_score>")
    end_idx = generation.rfind("<end_conclusion_complexity_score>")

    if start_idx == -1 or end_idx == -1:
        return NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG

    try:
        score = float(
            generation[start_idx + len("<begin_conclusion_complexity_score>") : end_idx]
        )
    except ValueError:
        return NO_CONCLUSION_COMPLEXITY_SCORE_FOUND_TAG

    return score


def extract_guide_relevance_score(generation: str) -> float:
    start_idx = generation.rfind("<begin_relevance_score>")
    end_idx = generation.rfind("<end_relevance_score>")

    if start_idx == -1 or end_idx == -1:
        return NO_RELEVANCE_SCORE_FOUND_TAG

    try:
        score = float(generation[start_idx + len("<begin_relevance_score>") : end_idx])
    except ValueError:
        return NO_RELEVANCE_SCORE_FOUND_TAG

    return score


def get_guide_prompt(*, seed_theorem: str, conjecture: str) -> str:
    part_1 = f"""You are a math expert. Here is a seed lean4 problem statement:

```lean4
{seed_theorem}
```

Here is a lemma or related lemma that is supposed to be useful for proving the above statement. It can be useful in that either it is a lemma
that can be directly used to help prove the above statement, or it is a related lemma that plausible requires similar proof techniques to solve.

```lean4
{conjecture}
```

Please rate the relevance of the lemma to the seed problem on a scale of 0 to 5, where 0 is "not at all related" and 5 is "very useful for proving the target statement" If
the lemma is trivial, you should give it a low score.

Here is a rubric for how to score the lemma:
- 0: The lemma is not at all related to the seed problem and is trivial to prove. OR the lemma is identical (including equivalent by renaming of premises and variables) to the seed problem.
- 1: The lemma is not at all related to the seed problem.
- 2: The lemma is related to the seed problem in that it concerns a similar subfield of mathematics, but is not directly useful for proving the seed problem.
- 3: The lemma is related to the seed problem and may be useful for proving the seed problem.
- 4: The lemma is directly useful for solving the seed problem. That is if the lemma was proved, the seed problem would be easier to solve.
- 5: The lemma is very useful for solving the seed problem, and solving the lemma will dramatically reduce the difficulty of the original seed problem

Next decide how redundant the premises are. Rate this as 0 or 1:

- 0: There are no redundant premises.
- 1: There are redundant premises. That is premises that are not needed to prove the conclusion.

Next decide if the conclusion is overly complex. Rate this on a score of 0 to 4:

- 0: The conclusion is minimally complex. That is it is a single, atomic statement (e.g., a simple equality, inequality, or property) that is maximally clear and easy to apply to other problems.
- 1: The conclusion has low complexity. That is it has multiple related parts (e.g., 2-3 conjunctions) but they form a cohesive statement where all parts directly relate to each other and the premises, making it still straightforward to apply.
- 2: The conclusion has moderate complexity. That is it contains disjunctions (or clauses) but they are closely related alternatives, or it contains multiple conjunctions (3-4) that are all on-theme. The structure is clear but requires some thought to apply.
- 3: The conclusion has high complexity. That is it contains multiple unrelated or clauses (2-3 disjunctions), or contains deep nesting of logical operators that obscure the main claim. Different parts address somewhat different aspects with weak connections to each other or the premises, making it difficult to understand when and how to apply.
- 4: The conclusion has very high complexity. That is it is a disjunction of many (3 or more) largely unrelated clauses, or contains deeply nested logical structure that is hard to parse. Parts address completely different mathematical objects or properties which would make it nearly impossible to apply meaningfully to other problems. It feels like multiple lemmas packaged as one.

Once you are done reasoning about all of these things, output the scores between tags:

1) Relevance score: <begin_relevance_score> <end_relevance_score>
2) Redundancy score: <begin_redundancy_score> <end_redundancy_score>
3) Conclusion complexity score: <begin_conclusion_complexity_score> <end_conclusion_complexity_score>

"""

    ic_example_1 = """\n\nHere is an example:

===========================================
INPUT SEED PROBLEM:
```lean4
theorem aime_1983_p9 (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi) :
  12 ≤ (9 * (x ^ 2 * Real.sin x ^ 2) + 4) / (x * Real.sin x) := by
```

INPUT CONJECTURE:
```lean4
theorem china1993_p6 (x : ℝ) (h₀ : 0 < x ∧ x ≤ π) :
    Real.sqrt (1 + Real.sin x) + Real.sqrt (1 + Real.cos x) ≤
      Real.sqrt (9 + 2 * (Real.sin x + Real.cos x)):= by
```

YOUR OUTPUT: <... your reasoning ...>

<begin_relevance_score>
< ... your relevance score ... >
<end_relevance_score>
<begin_redundancy_score>
< ... your redundancy score ... >
<end_redundancy_score>
<begin_conclusion_complexity_score>
< ... your conclusion complexity score ... >
<end_conclusion_complexity_score>
===========================================
"""

    return part_1 + ic_example_1


