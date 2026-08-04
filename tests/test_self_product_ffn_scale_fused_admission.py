import json
import inspect

import pytest

pytest.importorskip("torch")

from experiments.self_product_ffn_scale_fused_admission import (
    OUTPUT,
    PRECOMMIT,
    PREREGISTRATION,
    TEST_SOURCE,
)
from experiments import self_product_ffn_scale_fused_admission as admission


def test_admission_path_is_precommitted_and_uses_distinct_output():
    manifest = json.loads(PRECOMMIT.read_text())
    assert manifest["admission_source_sha256"] == admission.scale.base.sha256_file(admission.Path(admission.__file__))
    assert manifest["admission_preregistration_sha256"] == admission.scale.base.sha256_file(PREREGISTRATION)
    assert manifest["admission_test_sha256"] == admission.scale.base.sha256_file(TEST_SOURCE)
    assert manifest["confirmation_sha256"] == admission.scale.base.sha256_file(admission.scale.CONFIRMATION)
    assert manifest["data_manifest_sha256"] == admission.scale.base.sha256_file(admission.scale.DATA_MANIFEST)
    assert OUTPUT.name == "self-product-ffn-scale-fused-admission.json"
    source = inspect.getsource(admission.run)
    assert 'fused.get("hashes", {}).get("precommit")' in source
    assert 'fused.get("semantic_equivalence_pass") is True' in source
    assert 'fused.get("fused_serving_pass") is True' in source
