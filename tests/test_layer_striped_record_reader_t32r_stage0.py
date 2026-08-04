from experiments.layer_striped_record_reader_t32r_stage0 import (
    compute_bound,
    handle_tokenized_length,
    parameter_ledger,
)


class CharacterTokenizer:
    name_or_path = "character"

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text)


def test_handle_length_replaces_exact_two_spans_by_two_ids():
    question = "AA title one BB title two CC"
    spans = ((3, 12), (16, 25))
    assert handle_tokenized_length(question, spans, CharacterTokenizer()) == 12


def test_parameter_ledger_is_exact_and_parameter_neutral():
    ledger = parameter_ledger()
    assert ledger == {
        "removed_ffn_width_per_layer": 84,
        "removed_dense_entries": 967_680,
        "record_digit_entries": 916_305,
        "shared_token_query_projection_entries": 12_288,
        "scan_qkv_entries": 3_072,
        "bounded_local_scan_reserve_entries": 3_072,
        "summary_output_entries": 12_288,
        "candidate_allocated_entries": 947_025,
        "matched_slack_entries": 20_655,
    }


def test_compute_bound_breaks_even_at_four_tokens():
    assert compute_bound(3)["margin_multiplies"] < 0
    assert compute_bound(4)["margin_multiplies"] > 0
    assert compute_bound(4) == {
        "question_tokens": 4,
        "scan_upper_bound_multiplies": 3_749_152,
        "dense_saving_multiplies": 3_870_720,
        "margin_multiplies": 121_568,
    }
