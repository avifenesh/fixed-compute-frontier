from __future__ import annotations

from experiments import hotpot_real_prose_t12_data as data


def row(identifier: str, answer: str, titles: tuple[str, str]) -> dict[str, object]:
    return {
        "id": identifier,
        "question": f"Are {titles[0]} and {titles[1]} both musicians?",
        "answer": answer,
        "type": "comparison",
        "supporting_facts": {"title": list(titles), "sent_id": [0, 0]},
        "context": {
            "title": [*titles, "Distractor"],
            "sentences": [[f"{titles[0]} text."], [f"{titles[1]} text."], ["Noise."]],
        },
    }


def identifier_for_bucket(bucket: int) -> str:
    for index in range(10_000):
        value = f"id-{bucket}-{index}"
        if data.stable_bucket(value) == bucket:
            return value
    raise AssertionError("no identifier found")


def test_eligibility_requires_natural_both_question_and_two_exact_titles() -> None:
    good = row("good", "yes", ("First Entity", "Second Entity"))
    assert data.eligible(good)
    assert not data.eligible({**good, "answer": "maybe"})
    assert not data.eligible({**good, "type": "bridge"})
    assert not data.eligible({**good, "question": "Are they musicians?"})


def test_split_drops_evaluation_title_overlap() -> None:
    train = row(identifier_for_bucket(4), "yes", ("A", "B"))
    clean = row(identifier_for_bucket(0), "no", ("C", "D"))
    overlap = row(identifier_for_bucket(1), "yes", ("A", "E"))
    split = data.split_rows((train, clean, overlap))
    assert {item["id"] for item in split.train} == {train["id"]}
    assert {item["id"] for item in split.evaluation} == {clean["id"]}
    assert {item["id"] for item in split.dropped_title_overlap} == {overlap["id"]}


def test_candidate_documents_contain_raw_prose_only() -> None:
    train = row(identifier_for_bucket(4), "yes", ("A", "B"))
    evaluation = row(identifier_for_bucket(0), "no", ("C", "D"))
    documents = data.candidate_documents(data.split_rows((train, evaluation)))
    assert documents
    assert set().union(*(document.keys() for document in documents)) == {
        "document_id",
        "title",
        "text",
    }
    assert all("answer" not in document and "question" not in document for document in documents)
