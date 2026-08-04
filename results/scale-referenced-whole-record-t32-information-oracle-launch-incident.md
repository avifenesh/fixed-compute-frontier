# T32 information oracle — pre-execution launcher incident

Status: **PRESERVED; ZERO EXPERIMENT EXECUTION**  
Date: 2026-07-31

The first shell launch invoked the frozen source as a file path. Python placed
the `experiments/` directory rather than the project root on `sys.path`, so the
module's first package import failed with `ModuleNotFoundError: experiments`.

The failure occurred before runtime/model imports, artifact loads, evaluator
reads, prompt construction, model loading, or scoring. The output JSON was zero
bytes and H100 memory/utilization were zero. The traceback is preserved at
`scale-referenced-whole-record-t32-information-oracle-prelaunch-error.log`,
SHA-256
`1dddd6c12c754c2c9f8644b327bc74a54ce5360694225b218974ffb5c26b40fb`.

The frozen source remains byte-identical at SHA-256
`02460eb39889eca0011d4867d4d413ac7b897105bc318a05b26c50d87c59df8a`.
The admitted correction is solely to invoke it through Python's package mode:

```text
python -m experiments.scale_referenced_whole_record_t32_information_oracle
```

This is not a scoring retry because no experimental code executed. There is
still exactly one admitted prompt-scoring run.
