import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "self_indexed_sparse_latent_stage0.py"
SPEC = importlib.util.spec_from_file_location("self_indexed_sparse_latent_stage0", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pointer_lookup_and_dense_reference_agree():
    table = MODULE.build_table(nodes=13, degree=4, seed=19)
    labels = [3, 1, 0, 2, 2]
    pointer, pointer_reads = MODULE.pointer_walk(table, 4, 7, labels)
    lookup, lookup_reads = MODULE.lookup_control_walk(table, 4, 7, labels)
    dense, dense_macs = MODULE.dense_one_hot_walk(table, 13, 4, 7, labels)
    assert pointer == lookup == dense
    assert pointer_reads == lookup_reads == len(labels) * 4
    assert dense_macs == len(labels) * 13 * 13


def test_global_reachability_lower_bound():
    assert MODULE.min_global_steps(2**24, 16) == 6
    assert MODULE.reachable_upper_bound(16, 6) >= 2**24
    assert MODULE.reachable_upper_bound(16, 5) < 2**24


def test_frozen_stage0_gates_pass():
    result = MODULE.run(7302026)
    assert result["all_gates_pass"]
    assert result["fixed_2gib_ledger"]["resident_graph_nodes"] == 2**24
    active = {
        row["active_records_per_token"] for row in result["scaling"]
    }
    assert active == {256 * 16 * 32}

