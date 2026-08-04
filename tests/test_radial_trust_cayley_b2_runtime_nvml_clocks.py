from dataclasses import replace

import pytest

import experiments.radial_trust_cayley_b2_runtime_nvml_clocks as clocks
from experiments.radial_trust_cayley_b2_runtime_manifest import ManifestError


def _success(*values: int, capacity: int | None = None) -> clocks.ClockListTranscript:
    allocated = len(values) if capacity is None else capacity
    return clocks.ClockListTranscript(
        first_input_count=0,
        first_pointer_null=True,
        first_return_code=7,
        first_returned_count=allocated,
        second_input_capacity=allocated,
        second_pointer_null=False,
        second_return_code=0,
        second_returned_count=len(values),
        second_buffer=tuple(values) + (0,) * (allocated - len(values)),
    )


def _terminal(code: int) -> clocks.ClockListTranscript:
    return clocks.ClockListTranscript(0, True, code, 0)


def test_pinned_protocol_constants() -> None:
    assert clocks.NVML_SUCCESS == 0
    assert clocks.NVML_ERROR_NOT_SUPPORTED == 3
    assert clocks.NVML_ERROR_INSUFFICIENT_SIZE == 7


def test_terminal_first_calls_record_explicit_inputs_and_no_second_call() -> None:
    unsupported = clocks.evaluate_clock_list_transcript(_terminal(3)).record()
    empty = clocks.evaluate_clock_list_transcript(_terminal(0)).record()
    assert unsupported == {
        "return_code": "3",
        "status": "not_supported",
        "transcript": {
            "first_input_count": "0",
            "first_pointer_null": True,
            "first_return_code": "3",
            "first_returned_count": "0",
        },
    }
    assert empty["values_mhz"] == []
    assert empty["transcript"]["first_pointer_null"] is True


@pytest.mark.parametrize(
    "change",
    (
        {"first_input_count": 1},
        {"first_pointer_null": False},
        {"first_returned_count": 1},
        {"second_input_capacity": 0},
        {"second_pointer_null": True},
        {"second_return_code": 0},
        {"second_returned_count": 0},
        {"second_buffer": ()},
    ),
)
def test_terminal_first_call_rejects_wrong_inputs_counts_and_stale_second_fields(
    change: dict[str, object],
) -> None:
    with pytest.raises(ManifestError):
        clocks.evaluate_clock_list_transcript(replace(_terminal(3), **change))


def test_success_records_raw_index_order_but_publishes_bytewise_decimal_order() -> None:
    record = clocks.evaluate_clock_list_transcript(_success(900, 1000)).record()
    assert record["transcript"]["second_buffer_mhz"] == ["900", "1000"]
    assert record["values_mhz"] == ["1000", "900"]


def test_success_pins_capacity_pointer_counts_complete_buffer_and_zero_tail() -> None:
    transcript = _success(1593, 1215, capacity=4)
    record = clocks.evaluate_clock_list_transcript(transcript).record()
    assert record["transcript"] == {
        "first_input_count": "0",
        "first_pointer_null": True,
        "first_return_code": "7",
        "first_returned_count": "4",
        "second_buffer_mhz": ["1593", "1215", "0", "0"],
        "second_input_capacity": "4",
        "second_pointer_null": False,
        "second_return_code": "0",
        "second_returned_count": "2",
    }
    with pytest.raises(ManifestError, match="capacity differs"):
        clocks.evaluate_clock_list_transcript(
            replace(transcript, second_input_capacity=3)
        )
    with pytest.raises(ManifestError, match="non-NULL"):
        clocks.evaluate_clock_list_transcript(
            replace(transcript, second_pointer_null=True)
        )
    with pytest.raises(ManifestError, match="complete allocated buffer"):
        clocks.evaluate_clock_list_transcript(
            replace(transcript, second_buffer=(1593, 1215))
        )
    with pytest.raises(ManifestError, match="tail"):
        clocks.evaluate_clock_list_transcript(
            replace(transcript, second_buffer=(1593, 1215, 999, 0))
        )


@pytest.mark.parametrize("second_code", (1, 3, 7, 999))
def test_every_second_call_failure_invalidates_without_retry(second_code: int) -> None:
    with pytest.raises(ManifestError, match="retry is forbidden"):
        clocks.evaluate_clock_list_transcript(
            replace(_success(1000), second_return_code=second_code)
        )


@pytest.mark.parametrize(
    ("transcript", "message"),
    (
        (clocks.ClockListTranscript(0, True, 1, 0), "invalid first"),
        (clocks.ClockListTranscript(0, True, 7, 0), "invalid first"),
        (clocks.ClockListTranscript(0, True, 7, 2), "capacity differs"),
        (replace(_success(1000), second_returned_count=0), "exceeds or empties"),
        (replace(_success(1000), second_returned_count=2), "exceeds or empties"),
        (_success(1000, 1000), "duplicate"),
    ),
)
def test_malformed_or_inconsistent_transcripts_fail_closed(
    transcript: clocks.ClockListTranscript,
    message: str,
) -> None:
    with pytest.raises(ManifestError, match=message):
        clocks.evaluate_clock_list_transcript(transcript)


def test_graphics_call_order_must_equal_bytewise_memory_key_order() -> None:
    memory = _success(900, 1000)
    canonical_graphics = (
        clocks.GraphicsClockTranscript(1000, _success(1800, 1600)),
        clocks.GraphicsClockTranscript(900, _success(1500, 1400)),
    )
    record = clocks.compose_supported_clock_records(memory, canonical_graphics)
    assert [
        item["memory_clock_mhz"] for item in record["supported_graphics_clocks"]
    ] == ["1000", "900"]
    with pytest.raises(ManifestError, match="canonical memory-clock order"):
        clocks.compose_supported_clock_records(
            memory,
            tuple(reversed(canonical_graphics)),
        )


def test_graphics_composition_rejects_missing_duplicate_and_extra_keys() -> None:
    memory = _success(1000, 1200)
    with pytest.raises(ManifestError, match="canonical memory-clock order"):
        clocks.compose_supported_clock_records(
            memory,
            (clocks.GraphicsClockTranscript(1000, _success(1500)),),
        )
    with pytest.raises(ManifestError, match="canonical memory-clock order"):
        clocks.compose_supported_clock_records(
            _success(1000),
            (
                clocks.GraphicsClockTranscript(1000, _success(1500)),
                clocks.GraphicsClockTranscript(1000, _success(1500)),
            ),
        )
    with pytest.raises(ManifestError, match="canonical memory-clock order"):
        clocks.compose_supported_clock_records(
            _success(1000),
            (clocks.GraphicsClockTranscript(999, _success(1500)),),
        )


def test_empty_or_not_supported_memory_forbids_graphics_calls() -> None:
    assert clocks.compose_supported_clock_records(
        _terminal(3),
        (),
    )["supported_graphics_clocks"] == []
    assert clocks.compose_supported_clock_records(
        _terminal(0),
        (),
    )["supported_graphics_clocks"] == []
    with pytest.raises(ManifestError, match="canonical memory-clock order"):
        clocks.compose_supported_clock_records(
            _terminal(3),
            (clocks.GraphicsClockTranscript(1000, _success(1500)),),
        )


def test_corresponding_graphics_query_can_be_not_supported() -> None:
    record = clocks.compose_supported_clock_records(
        _success(1000),
        (clocks.GraphicsClockTranscript(1000, _terminal(3)),),
    )
    result = record["supported_graphics_clocks"][0]["result"]
    assert result["status"] == "not_supported"
    assert result["return_code"] == "3"


class _TranscriptSubclass(clocks.ClockListTranscript):
    pass


class _GraphicsSubclass(clocks.GraphicsClockTranscript):
    pass


class _DeceptiveStr(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def test_record_and_primitive_subclasses_are_rejected() -> None:
    ordinary = _success(1000)
    with pytest.raises(ManifestError, match="transcript must be exact"):
        clocks.evaluate_clock_list_transcript(
            _TranscriptSubclass(**ordinary.__dict__)
        )
    graphics = clocks.GraphicsClockTranscript(1000, _success(1500))
    with pytest.raises(ManifestError, match="graphics transcript must be exact"):
        clocks.compose_supported_clock_records(
            ordinary,
            (_GraphicsSubclass(**graphics.__dict__),),
        )
    with pytest.raises(ManifestError, match="exact integer"):
        clocks.evaluate_clock_list_transcript(
            replace(ordinary, second_buffer=(True,))
        )


@pytest.mark.parametrize("status", (_DeceptiveStr("evil"), _DeceptiveStr("not_supported")))
def test_forged_outcome_cannot_override_status_union(status: str) -> None:
    ordinary = clocks.evaluate_clock_list_transcript(_success(1000))
    forged = clocks._ListOutcome(
        status,
        ordinary.return_code,
        ordinary.values,
        ordinary.transcript,
    )
    with pytest.raises(ManifestError, match="status must be exact string"):
        forged.record()
