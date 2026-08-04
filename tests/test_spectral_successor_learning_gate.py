import torch

from experiments.spectral_successor_learning_gate import (
    AuxiliaryTargets,
    GAP,
    PLAN_TARGET_START,
    SEQUENCE_LENGTH,
    make_batch,
    plan_table,
    structured_plans,
)


def test_structured_plans_are_distinct_permutations_of_the_same_multiset() -> None:
    plans = structured_plans()
    reference = torch.sort(plans[0]).values
    assert torch.unique(plans, dim=0).shape[0] == plans.shape[0]
    for plan in plans:
        torch.testing.assert_close(torch.sort(plan).values, reference)


def test_batch_layout_and_plan_boundary() -> None:
    generator = torch.Generator().manual_seed(1)
    sequence = make_batch("structured", 8, generator, torch.device("cpu"))
    assert sequence.shape == (8, SEQUENCE_LENGTH)
    assert bool((sequence[:, PLAN_TARGET_START] == 4).all())
    assert bool((sequence[:, PLAN_TARGET_START + 1 :] >= 128).all())
    assert sequence.shape[1] == 2 + GAP + 1 + plan_table("structured").shape[1]


def test_spectral_separates_plans_while_bow_does_not() -> None:
    factory = AuxiliaryTargets(torch.device("cpu"))
    filler = torch.full((2, GAP), 96, dtype=torch.long)
    plans = plan_table("structured")[[0, 1]]
    sequence = torch.cat(
        (
            torch.zeros(2, 1, dtype=torch.long),
            torch.tensor([[16], [17]]),
            filler,
            torch.full((2, 1), 4, dtype=torch.long),
            plans,
        ),
        dim=1,
    )
    bow = factory.bow(sequence)
    spectral = factory.spectral(sequence)
    torch.testing.assert_close(
        bow[0, PLAN_TARGET_START], bow[1, PLAN_TARGET_START]
    )
    assert not torch.allclose(
        spectral[0, PLAN_TARGET_START], spectral[1, PLAN_TARGET_START]
    )
