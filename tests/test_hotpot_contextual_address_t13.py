from __future__ import annotations

import inspect

import torch

from experiments import hotpot_contextual_address_t13 as experiment


def test_t13_has_no_training_surface() -> None:
    assert tuple(inspect.signature(experiment.run).parameters) == (
        "device",
        "checkpoint_path",
    )
    source = inspect.getsource(experiment.run)
    assert "optimizer" not in source
    assert ".backward(" not in source


def test_frozen_integrity_and_interface_constants() -> None:
    assert experiment.CHECKPOINT_BYTES == 146_340_367
    assert experiment.CHECKPOINT_SHA256 == (
        "a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b"
    )
    assert experiment.DOCUMENT_TOKENS == 512
    assert experiment.CHUNK_TOKENS == 128
    assert experiment.LEXICAL_REFERENCE_ACCURACY == 0.5576923076923077
    assert experiment.STATIC_REFERENCE_ACCURACY == 0.6057692307692307


def test_final_ffn_interface_is_exact_gate_input() -> None:
    device = torch.device("cpu")
    model = experiment.t10.core.build_model(91, device).eval()
    tokens = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        captured.append(arguments[0].detach().clone())

    hook = model.blocks[-1].gate.register_forward_pre_hook(capture)
    with torch.inference_mode():
        model.hidden(tokens)
        direct = experiment.final_ffn_interface(model, tokens)
    hook.remove()
    assert len(captured) == 1
    assert torch.equal(direct, captured[0])


def test_right_padding_cannot_change_real_token_states() -> None:
    device = torch.device("cpu")
    model = experiment.t10.core.build_model(101, device).eval()
    sequences = [[21, 22, 23], [31, 32, 33, 34, 35]]
    batched = experiment.encode_sequences(model, sequences, 2, device, batch_size=2)
    separate = [
        experiment.encode_sequences(model, [sequence], 2, device, batch_size=1)[0]
        for sequence in sequences
    ]
    for left, right in zip(batched, separate, strict=True):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_document_chunking_is_frozen_and_lossless_within_cap() -> None:
    identifiers = list(range(600))
    chunks = experiment.document_token_chunks(identifiers)
    assert tuple(len(chunk) for chunk in chunks) == (128, 128, 128, 128)
    assert [identifier for chunk in chunks for identifier in chunk] == identifiers[:512]
