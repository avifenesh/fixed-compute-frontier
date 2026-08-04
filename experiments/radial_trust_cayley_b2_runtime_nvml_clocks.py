#!/usr/bin/env python3
"""Pure Amendment-2 transcript evaluator for B2 NVML clock lists.

No NVML library is loaded here.  A later audited ABI caller must execute one
size-probe/value-call pair and supply every input and output field represented
below.  A failed second call never retries.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.radial_trust_cayley_b2_runtime_manifest import (
    ManifestError,
    decimal_integer,
)


NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_INSUFFICIENT_SIZE = 7


@dataclass(frozen=True)
class ClockListTranscript:
    first_input_count: int
    first_pointer_null: bool
    first_return_code: int
    first_returned_count: int
    second_input_capacity: int | None = None
    second_pointer_null: bool | None = None
    second_return_code: int | None = None
    second_returned_count: int | None = None
    second_buffer: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GraphicsClockTranscript:
    memory_clock_mhz: int
    transcript: ClockListTranscript


@dataclass(frozen=True)
class _ListOutcome:
    status: str
    return_code: int
    values: tuple[int, ...]
    transcript: ClockListTranscript

    def record(self) -> dict[str, object]:
        _validate_outcome(self)
        fresh = evaluate_clock_list_transcript(self.transcript)
        if (
            fresh.status != self.status
            or fresh.return_code != self.return_code
            or fresh.values != self.values
        ):
            raise ManifestError("clock-list outcome disagrees with its transcript")
        record: dict[str, object] = {
            "return_code": decimal_integer(self.return_code),
            "status": self.status,
            "transcript": _transcript_record(self.transcript),
        }
        if self.status == "ok":
            record["values_mhz"] = [decimal_integer(value) for value in self.values]
        return record


def _uint32(value: int, *, field: str) -> int:
    if type(value) is not int:
        raise ManifestError(f"{field} must be an exact integer")
    if not 0 <= value <= 0xFFFFFFFF:
        raise ManifestError(f"{field} is outside uint32")
    return value


def _return_code(value: int, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 0x7FFFFFFF:
        raise ManifestError(f"{field} must be a nonnegative exact int")
    return value


def _decimal_byte_key(value: int) -> bytes:
    return decimal_integer(value).encode("ascii")


def _no_second_call(transcript: ClockListTranscript) -> None:
    if (
        transcript.second_input_capacity is not None
        or transcript.second_pointer_null is not None
        or transcript.second_return_code is not None
        or transcript.second_returned_count is not None
        or transcript.second_buffer is not None
    ):
        raise ManifestError("terminal first call retained second-call fields")


def _transcript_record(transcript: ClockListTranscript) -> dict[str, object]:
    record: dict[str, object] = {
        "first_input_count": decimal_integer(transcript.first_input_count),
        "first_pointer_null": transcript.first_pointer_null,
        "first_return_code": decimal_integer(transcript.first_return_code),
        "first_returned_count": decimal_integer(transcript.first_returned_count),
    }
    if transcript.second_input_capacity is not None:
        assert transcript.second_pointer_null is not None
        assert transcript.second_return_code is not None
        assert transcript.second_returned_count is not None
        assert transcript.second_buffer is not None
        record.update(
            {
                "second_buffer_mhz": [
                    decimal_integer(value) for value in transcript.second_buffer
                ],
                "second_input_capacity": decimal_integer(
                    transcript.second_input_capacity
                ),
                "second_pointer_null": transcript.second_pointer_null,
                "second_return_code": decimal_integer(transcript.second_return_code),
                "second_returned_count": decimal_integer(
                    transcript.second_returned_count
                ),
            }
        )
    return record


def _validate_outcome(outcome: _ListOutcome) -> None:
    if type(outcome) is not _ListOutcome:
        raise ManifestError("clock-list outcome must be exact")
    if type(outcome.status) is not str:
        raise ManifestError("clock-list outcome status must be exact string")
    code = _return_code(outcome.return_code, field="outcome return code")
    if type(outcome.values) is not tuple:
        raise ManifestError("clock-list outcome values must be an exact tuple")
    values = tuple(_uint32(value, field="outcome clock") for value in outcome.values)
    expected = tuple(sorted(set(values), key=_decimal_byte_key))
    if values != expected:
        raise ManifestError("clock-list outcome values are not unique and bytewise sorted")
    if type(outcome.transcript) is not ClockListTranscript:
        raise ManifestError("clock-list outcome transcript must be exact")
    if outcome.status == "ok":
        if code != NVML_SUCCESS:
            raise ManifestError("successful clock-list outcome has nonzero code")
    elif outcome.status == "not_supported":
        if code != NVML_ERROR_NOT_SUPPORTED or values:
            raise ManifestError("not-supported clock-list outcome is inconsistent")
    else:
        raise ManifestError("unknown clock-list outcome status")


def evaluate_clock_list_transcript(
    transcript: ClockListTranscript,
) -> _ListOutcome:
    if type(transcript) is not ClockListTranscript:
        raise ManifestError("clock-list transcript must be exact")
    if type(transcript.first_input_count) is not int or transcript.first_input_count != 0:
        raise ManifestError("first clock-list input count must be exact zero")
    if type(transcript.first_pointer_null) is not bool or not transcript.first_pointer_null:
        raise ManifestError("first clock-list pointer must be exact NULL")

    first_code = _return_code(transcript.first_return_code, field="first return code")
    first_count = _uint32(transcript.first_returned_count, field="first returned count")
    if first_code == NVML_ERROR_NOT_SUPPORTED:
        if first_count != 0:
            raise ManifestError("not-supported first call changed zero count")
        _no_second_call(transcript)
        return _ListOutcome("not_supported", first_code, (), transcript)

    if first_code == NVML_SUCCESS:
        if first_count != 0:
            raise ManifestError("NULL-buffer success returned a nonempty count")
        _no_second_call(transcript)
        return _ListOutcome("ok", first_code, (), transcript)

    if first_code != NVML_ERROR_INSUFFICIENT_SIZE or first_count == 0:
        raise ManifestError("invalid first call in clock-list protocol")
    if (
        type(transcript.second_input_capacity) is not int
        or transcript.second_input_capacity != first_count
    ):
        raise ManifestError("second input capacity differs from probed count")
    if type(transcript.second_pointer_null) is not bool or transcript.second_pointer_null:
        raise ManifestError("second clock-list pointer must be exact non-NULL")
    if transcript.second_return_code is None or transcript.second_returned_count is None:
        raise ManifestError("insufficient-size first call omitted second outputs")
    second_code = _return_code(transcript.second_return_code, field="second return code")
    second_count = _uint32(
        transcript.second_returned_count,
        field="second returned count",
    )
    if second_code != NVML_SUCCESS:
        raise ManifestError("second clock-list call did not succeed; retry is forbidden")
    if not 1 <= second_count <= first_count:
        raise ManifestError("successful second count exceeds or empties capacity")
    if type(transcript.second_buffer) is not tuple or len(transcript.second_buffer) != first_count:
        raise ManifestError("second call lacks its complete allocated buffer")
    buffer = tuple(
        _uint32(value, field="clock buffer value")
        for value in transcript.second_buffer
    )
    if any(value != 0 for value in buffer[second_count:]):
        raise ManifestError("unused clock-list buffer tail is not zero")
    active = buffer[:second_count]
    if len(active) != len(set(active)):
        raise ManifestError("clock-list success contains duplicate values")
    canonical = tuple(sorted(active, key=_decimal_byte_key))
    return _ListOutcome("ok", second_code, canonical, transcript)


def compose_supported_clock_records(
    memory_transcript: ClockListTranscript,
    graphics_transcripts: tuple[GraphicsClockTranscript, ...],
) -> dict[str, object]:
    if type(graphics_transcripts) is not tuple:
        raise ManifestError("graphics transcripts must be an exact tuple")
    memory = evaluate_clock_list_transcript(memory_transcript)
    expected_order = memory.values
    observed_order: list[int] = []
    graphics_records: list[dict[str, object]] = []
    for transcript in graphics_transcripts:
        if type(transcript) is not GraphicsClockTranscript:
            raise ManifestError("graphics transcript must be exact")
        memory_clock = _uint32(
            transcript.memory_clock_mhz,
            field="graphics memory clock",
        )
        observed_order.append(memory_clock)
        outcome = evaluate_clock_list_transcript(transcript.transcript)
        graphics_records.append(
            {
                "memory_clock_mhz": decimal_integer(memory_clock),
                "result": outcome.record(),
            }
        )
    if tuple(observed_order) != expected_order:
        raise ManifestError(
            "graphics calls do not exactly match bytewise canonical memory-clock order"
        )
    return {
        "supported_graphics_clocks": graphics_records,
        "supported_memory_clocks": memory.record(),
    }


__all__ = [
    "ClockListTranscript",
    "GraphicsClockTranscript",
    "NVML_ERROR_INSUFFICIENT_SIZE",
    "NVML_ERROR_NOT_SUPPORTED",
    "NVML_SUCCESS",
    "compose_supported_clock_records",
    "evaluate_clock_list_transcript",
]
