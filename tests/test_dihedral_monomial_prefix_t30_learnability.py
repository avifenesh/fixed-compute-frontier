import numpy as np
import torch

from experiments.dihedral_monomial_prefix_t30_learnability import (
    DOCUMENT_LENGTH,
    EVALUATION_DOCUMENTS,
    IDENTITY_GROUP,
    RELATIONS,
    ROUTE_CHOICES,
    ScreenModel,
    TOKEN_F,
    TOKEN_R,
    TRAIN_DOCUMENTS,
    VALUES,
    WIDTH,
    build_world,
    canonical_group_words,
    compile_documents,
    document_multiset_failures,
    group_training_batch,
    marker_targets,
    numpy_compose_group,
    true_group_for_words,
)


def test_world_split_and_document_multisets():
    world = build_world()
    assert world.document_tokens.shape == (TRAIN_DOCUMENTS + EVALUATION_DOCUMENTS, DOCUMENT_LENGTH)
    assert world.permutations.shape == (TRAIN_DOCUMENTS + EVALUATION_DOCUMENTS, RELATIONS)
    assert len(np.intersect1d(world.train_indices, world.evaluation_indices)) == 0
    assert document_multiset_failures(world.document_tokens) == 0
    assert all(sorted(row.tolist()) == list(range(VALUES)) for row in world.permutations)


def test_true_group_composition_order():
    words = np.asarray(
        [
            [TOKEN_R, TOKEN_F],
            [TOKEN_F, TOKEN_R],
        ],
        dtype=np.int64,
    )
    signs, rotations = true_group_for_words(words, np.asarray([2, 2]))
    rho = (1, 1)
    tau = (-1, 0)
    assert (int(signs[0]), int(rotations[0])) == numpy_compose_group(tau, rho)
    assert (int(signs[1]), int(rotations[1])) == numpy_compose_group(rho, tau)
    assert (int(signs[0]), int(rotations[0])) != (int(signs[1]), int(rotations[1]))


def test_marker_targets_distinguish_all_group_elements():
    signs = np.repeat(np.asarray([1, -1]), WIDTH)
    rotations = np.tile(np.arange(WIDTH), 2)
    targets = marker_targets(signs, rotations)
    assert len({row.tobytes() for row in targets}) == 2 * WIDTH


def test_group_training_batch_contains_single_generators():
    tokens, targets = group_training_batch(3109, 1)
    assert tokens[0, 0] == TOKEN_R
    assert tokens[1, 0] == TOKEN_F
    assert np.array_equal(targets[0], marker_targets(np.asarray([1]), np.asarray([1]))[0])
    assert np.array_equal(targets[1], marker_targets(np.asarray([-1]), np.asarray([0]))[0])


def test_canonical_and_lengthened_words_have_same_targets():
    canonical_tokens, canonical_targets = canonical_group_words(False)
    lengthened_tokens, lengthened_targets = canonical_group_words(True)
    assert canonical_tokens.shape[0] == 2 * WIDTH
    assert lengthened_tokens.shape[0] == 2 * WIDTH
    assert np.array_equal(canonical_targets, lengthened_targets)
    assert lengthened_tokens.shape[1] > canonical_tokens.shape[1]


def test_candidate_and_diagonal_initial_parameters_match():
    candidate = ScreenModel(3109, True)
    diagonal = ScreenModel(3109, False)
    for left, right in zip(candidate.parameters(), diagonal.parameters(), strict=True):
        assert torch.equal(left, right)
    assert sum(value.numel() for value in candidate.parameters()) == sum(
        value.numel() for value in diagonal.parameters()
    )


def test_known_routes_track_noncommuting_word_exactly():
    model = ScreenModel(3109, True)
    with torch.no_grad():
        model.route_logits.fill_(-10.0)
        model.route_logits[0, ROUTE_CHOICES.index((1, 1))] = 10.0
        model.route_logits[1, ROUTE_CHOICES.index((-1, 0))] = 10.0
    tokens = torch.tensor([[TOKEN_R, TOKEN_F], [TOKEN_F, TOKEN_R]])
    initial = torch.zeros(2, WIDTH)
    initial[:, 0] = 1.0
    initial[:, 1] = 2.0
    observed = model.scan(tokens, initial).detach().numpy()
    signs, rotations = true_group_for_words(tokens.numpy(), np.asarray([2, 2]))
    assert np.array_equal(observed, marker_targets(signs, rotations))


def test_compiler_matches_online_scan_with_known_routes():
    world = build_world()
    model = ScreenModel(3109, True)
    with torch.no_grad():
        model.route_logits.fill_(-10.0)
        model.route_logits[0, ROUTE_CHOICES.index((1, 1))] = 10.0
        model.route_logits[1, ROUTE_CHOICES.index((-1, 0))] = 10.0
    tokens = world.document_tokens[:4]
    groups, offsets = compile_documents(model, tokens)
    online = model.scan(torch.from_numpy(tokens))
    assert np.array_equal(groups[:, 0], np.ones(4, dtype=np.int64))
    assert torch.allclose(offsets, online, atol=1e-6, rtol=0.0)


def test_byte_and_cell_ledger():
    assert WIDTH * 2 + 2 == 220
    assert 2 * WIDTH + 2 == 220
    assert IDENTITY_GROUP == (1, 0)
