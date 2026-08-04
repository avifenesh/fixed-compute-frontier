# Address-factored transport attention: Stage-0 decision

## Decision

Advance AFTA to one matched learned structured-task screen. Do not advance it
to language modeling or kernel work yet.

All ten frozen gates passed on the retained H100. The result establishes a
specific capability/cost mechanism: when several payload fields share one
address, a single full-width QK map can retrieve them through distinct
channel-group shifts much more reliably than equal-total-width narrow maps.
The opposite workload also behaved as predicted, so the result does not support
replacing all independent heads.

## Main result

At query noise `1.0`, shared-address four-field exact-record accuracy was:

- AFTA: `99.9878%-100%`, mean `99.9927%`;
- four equal-QK/PV-width narrow transported heads: `12.3779%-12.9395%`, mean
  `12.6318%`;
- one wide ordinary alignment: `0%`;
- one wide common-delay alignment: `0%`.

The mean AFTA advantage over the narrow control was `87.3608` percentage
points. At noise `1.5`, the mean advantage remained `96.4819` points.

On the required independent-address negative control, the result reversed. At
noise `1.0`, the narrow heads' mean per-field advantage over AFTA was `51.3354`
points. AFTA is therefore a record/neighbor specialization: it shares one
address and cannot replace heads that need unrelated addresses.

## Causal and ledger checks

- future-value perturbation error: exactly `0`;
- zero-offset equality with ordinary causal attention: `1.19e-7` maximum error;
- dense effective-map forward error: `1.79e-7`;
- Q/K/V gradient reference error: `4.77e-7`;
- no-renormalization reference error: `1.19e-7`;
- QK MACs, PV MACs, Q/K/V/O parameters, output width, K/V cache width,
  offset metadata, total V scalar loads, and total V load regions matched.

The executor topology is not yet proven free. AFTA places multiple shifted V
load regions under one attention map, whereas the narrow control has one
contiguous region per map. A later fused H100 gate must measure that layout
cost; Stage 0 claims only the exact matrix/cache ledger.

## What the learned screen must test

Use a hybrid layer: retain independent narrow/ordinary heads and allocate only
a bounded fraction of Q/K width to AFTA. It must learn the shared-address gain,
remain noninferior on independent-address and ordinary same-token tasks, beat
temperature and common-delay controls, and causally use the wide transported
path. Only that result can justify a tiny language-model screen.

## Frozen evidence

- source: `03bbb61a04b5aaa009670e4e18ab836685f3032509ce3a0369968ebb98c67d0c`
- preregistration: `01ecafd7a38d27a0e226f6b25b1eafabdb0f7d994f2f6f508d8c07fcd83391e6`
- result: `3efc18684ab8f1af34742c24f97138e60ad7b60b8e3ebc7a9a8949da4e2defbd`

An independent post-run audit reconstructed all ten gates, all 30 run rows,
the denominators, and the reported means and returned `VALID`.
