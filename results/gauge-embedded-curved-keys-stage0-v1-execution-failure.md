# GECK-G2 Stage 0 v1 execution failure

The frozen v1 computation passed all five tests and reached result writing, but
`json.dumps` rejected NumPy boolean values inside the atlas edge-case checks.
No Stage-0 result was produced and no mathematical observation was used to
change the gates.

The v2 change converts those two booleans to Python `bool` and adds a direct
JSON-serialization test. The construction, seeds, thresholds, ledgers, and
decision rule are unchanged.

Frozen v1 hashes:

- source: `adc680692d1de6ef866c3b567a33623296aba795e6b64e789b7a1412a8276cd8`
- preregistration: `5a9acf77aad84086bc771b44c84792517308a6f665c360b03949496161739b81`
- test: `66714df203b4d82b54024a51d958973856e461a97094def7e0c4543c8a98542f`
