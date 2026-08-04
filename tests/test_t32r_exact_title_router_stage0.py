import pytest

from experiments.t32r_exact_title_router_stage0 import (
    alias_question,
    audit,
    delete_first_title,
    route_rejects,
    route_strings,
    synthetic_overlap_passes,
)


def test_longest_nonoverlapping_match() -> None:
    assert synthetic_overlap_passes()


def test_casefold_route_and_declared_alias_boundary() -> None:
    titles = ("Basil Dean", "Francis Ford Coppola", "Ford")
    question = "Were Basil Dean and Francis Ford Coppola both alive?"
    assert route_strings(question, titles) == ("Basil Dean", "Francis Ford Coppola")
    assert route_strings(question.swapcase(), titles) == ("Basil Dean", "Francis Ford Coppola")
    assert route_rejects(alias_question(question, titles), titles)
    assert route_rejects(delete_first_title(question, titles), titles)


def test_full_frozen_domain_audit_preserves_registered_failure() -> None:
    with pytest.raises(RuntimeError, match="router gate failed"):
        audit()
