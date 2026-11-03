# R97W: portable reproduction (derived code)

This is an independently runnable **port** of the R97W synthetic meta-learning
experiment. The computational definitions (teacher, deterministic seeding,
120-parameter models, first-order Reptile, controls, training, evaluation and
qualification gates) were copied from a reviewed and path-redacted historical
source. The original frozen package and its predecessor attestation were not
copied here; their digests remain recorded in the derived spec.

## Method

- ECS: three independent cubic E0 outputs, each with 5 inputs and 8 units:
  `3 * 5 * 8 = 120` trainable weights.
- FFN: GELU network width 13: `13 * 5 + 13 + 3 * 13 + 3 = 120` weights.
- First-order Reptile slow initialization, 64 training episodes per family/seed,
  8 steps of inner learning, then 12 held-out test episodes per group.
- Smooth/conditional targets, seeds 17/83; cold/shuffled/continual controls.
- Explicit fast-state reset; no claim of spontaneous emergence or general
  continual learning.

**Note:** This is not the `CGC120` four-output wrapper. The benchmark uses a
three-output architecture with the same parameter budget.

## Execute from a clean checkout

Install PyTorch according to your system, then run:

```sh
python -m pip install -e .
python research/benchmarks/r97w/portable/test_portable.py
bash research/benchmarks/r97w/portable/RUN_PORTABLE.sh ../cgc-r97w-full
```

The output destination must not exist yet and must be **outside the repository**.
If the original results are recreated numerically within declared tolerances,
`historical_comparison.json` reports `historical_numeric_agreement=true`.
The result is still a **new reproduction**, not the original frozen run.

A smaller mechanics smoke test is also possible:

```sh
python research/benchmarks/r97w/portable/experiment.py --fixture --output ../cgc-r97w-fixture
```

The archived `result.json` and checkpoint are unmodified; the full runner
compares new summary values, controls and gates against archived results.
It does **not** claim byte-for-byte result/checkpoint hashes, original source
attestation, or hardware-independent bitwise numerical equality.

## Provenance

- Original experiment SHA-256: `219898f80448962d3a1952dc5a2d4ba381b3bd4260efdf777b985b50333e080a`
- Original frozen specification SHA-256: `531ab08da88bdb0f872e828b17be05ac4dca570580e22560f350fc8aa1671784`
- Original archival result SHA-256: `576922cb74ba4aa13f824ac399a389bb215e12cc4e4b936315a279386169c0db`
- Pinned E0 code SHA-256: `92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084`

The portable spec is **not** the historical frozen specification. Historical
R97O/R97Q/R97V predecessor digests are recorded for traceability but are not
reattested locally.
