from typing import List

from datasets import load_dataset

from sgs.data.dataset_types import DatasetType, Statement, StatementTag


def convert_raw_dict_to_statements(
    raw_dict: List[dict], dataset: DatasetType
) -> List[Statement]:
    if dataset != DatasetType.D_3K:
        raise ValueError(f"Invalid dataset type: {dataset}")

    ids = [row["id"] for row in raw_dict]
    assert len(ids) == len(set(ids)), "Ids are not unique"

    statements: List[Statement] = [
        Statement(
            id=row["id"],
            header=row["header"],
            theorem=row["theorem"],
            tag=StatementTag.TARGET.value,
            source=dataset.value,
            proofs=[],
        )
        for row in raw_dict
    ]

    for statement in statements:
        assert (
            "sorry" not in statement.theorem
        ), f"Statement {statement.id} has a sorry in the theorem"
        assert statement.theorem.endswith(
            ":= by"
        ), f"Statement {statement.id} does not end with ':= by'"

    return statements


def load_eval_dataset(eval_dataset: DatasetType) -> List[Statement]:
    if eval_dataset != DatasetType.D_3K:
        raise ValueError(f"Invalid dataset type: {eval_dataset}")

    raw = load_dataset("LukeBailey181Pub/D_3k")["train"]
    return convert_raw_dict_to_statements([x for x in raw], eval_dataset)


if __name__ == "__main__":
    dataset = load_eval_dataset(DatasetType.D_3K)
    print(f"Loaded {len(dataset)} statements from D_3K")
