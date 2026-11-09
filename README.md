# CGC — Cubic Geometric Core

**A compact geometric neural operator that competes with conventional feed-forward networks.**

CGC (Cubic Geometric Core) is a trainable nonlinear operator developed as part of the Elpis geometric intelligence research project.

Instead of a conventional stack of affine transformations and activation functions, CGC uses a collection of microscopic cubic transformations to construct a learned input-output mapping.

A CGC core can be evaluated repeatedly, differentiated through, and adapted through recurrent weight updates. The operator does not require an attention mechanism or KV cache for its own forward computation.

The research question is straightforward:

**Can a small geometric operator learn useful transformations as effectively as a conventional FFN?**

In our R97W experiments, a 120-parameter CGC configuration outperformed a parameter-matched GELU FFN across all 48 held-out paired evaluations under a controlled meta-learning and adaptation protocol.

## 1. The geometric operator

The original ECS_G E0 formulation uses the cubic function:

\[
\phi(z)=\frac{z+z^2+z^3}{2}
\]

For microscopic weight vectors \(w_i\) and input \(x\):

\[
f_W(x)=\sum_{i=1}^{N}\phi(x^\top w_i)
\]

Here, \(W\) contains the microscopic trainable weights and \(N\) is the number of microscopic units.

The gradient is analytically defined:

\[
\phi'(z)=\frac12+z+\frac32z^2
\]

This permits direct gradient-based optimization, differentiable unrolling, and repeated applications of the same parameterized operator.

For a scalar E0 cell, parameter count is:

\[
P=dN
\]

Multiple output channels can share the same input while maintaining independent microscopic weights.

### The 120-parameter configurations

Two configurations are represented in this repository:

| Configuration | Outputs | Input dimension | Units/output | Parameters |
|---|---:|---:|---:|---:|
| R97W benchmark CGC | 3 | 5 | 8 | 120 |
| CGC120 reference wrapper | 4 | 5 | 6 | 120 |

The experimental results below belong to the **three-output R97W configuration**.

The four-output `CGC120` PyTorch wrapper is a separate reference implementation.

## 2. CGC versus FFN — experimental results

### R97W: Learning after forgetting

R97W evaluated whether a geometric operator could acquire useful adaptation behavior through repeated task exposure and explicit resetting of episode-specific fast weights.

Both models used **exactly 120 trainable parameters**:

- CGC: three cubic channels, five inputs, eight units per output.
- FFN: GELU network with hidden width 13.

The experiment used first-order Reptile meta-learning, 64 meta-training episodes per group, eight inner adaptation steps, and 12 held-out test episodes per group.

Two synthetic task families were evaluated across two independent seeds.

### Held-out comparison

**Median query mean squared error after eight adaptation steps. Lower is better.**

| Task family | Seed | CGC MSE | FFN MSE | CGC error reduction |
|---|---:|---:|---:|---:|
| Smooth | 17 | 0.042095 | 0.073992 | 43.1% |
| Smooth | 83 | 0.051127 | 0.097771 | 47.7% |
| Conditional | 17 | 0.030286 | 0.058697 | 48.4% |
| Conditional | 83 | 0.044841 | 0.079982 | 43.9% |

**CGC outperformed the matched FFN in 48 of 48 paired held-out evaluations.**

Across the four experimental groups, CGC reduced median final query error by approximately 43–48% relative to the corresponding FFN.

The experiment also evaluated cold-start, shuffled-history, continual-learning and meta-trained controls, alongside a fast-state-reset forgetting probe.

These results establish that CGC is not merely capable of fitting nonlinear functions: under the tested adaptation dynamics, it can outperform a conventional FFN with an equivalent trainable parameter budget.

The comparison is parameter-matched, not a claim of matched FLOPs or universal superiority across neural-network workloads.

### Scientific evidence

The original experiment and its recorded results are available here:

- [R97W results](research/benchmarks/r97w/evidence/run_20261025T191037_af40e7f2/result.json)
- [Reproducible R97W implementation](research/benchmarks/r97w/portable/)
- [Reproduction methodology](research/benchmarks/r97w/portable/README.md)

The frozen historical scientific disposition was:

`SCIENTIFIC_PASS_BOUNDED_POST_FORGETTING_META_LEARNING_ECS_NO_CAUSAL_EMERGENCE_CLAIM`

The meta-learning procedure is explicitly trained; the result is not a claim that sophisticated learning behavior emerged spontaneously.

## 3. Differentiable recurrence and native validation

CGC originates from the ECS_G geometric computation research lineage.

The R97N and R97O qualification experiments established numerical agreement between native and PyTorch implementations of the microscopic E0 mathematics.

R97O validated differentiable multistep computation over 2, 4 and 8 successive updates, with 72/72 numerical qualification cases passing.

The validation included microscopic forward computation, S3 projections, gradient calculations, and differentiation through repeated updates.

[Original R97O evidence](evidence/phase97o_ecsg_e0_differentiable_unroll/run_YwHfIlCG/result.json)

The repository also provides a standalone C11 implementation of the geometric math, state and executor subset, together with native tests.

Additional research on K1 consolidation and sequential retention is documented separately under [research reports](docs/research/). K1 is a distinct state mechanism and is not part of the minimal PyTorch CGC operator.

## 4. Using CGC

### PyTorch

Install from the repository root:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Example using the four-output, 120-parameter wrapper:

```python
import torch
from cgc import CGC120

weights = 0.02 * torch.randn(
    4, 5, 6, dtype=torch.float64
)

model = CGC120(weights)

x = torch.randn(32, 5, dtype=torch.float64)
y = model(x)

print(y.shape)  # torch.Size([32, 4])
print(sum(p.numel() for p in model.parameters()))  # 120
```

### Reproduce CGC versus FFN

Run the complete R97W benchmark:

```bash
bash research/benchmarks/r97w/portable/RUN_PORTABLE.sh ../cgc-r97w-reproduction
```

The destination must not already exist and must be outside the repository.

The runner executes the portable benchmark, compares newly generated results against the archived experimental measurements, and writes a `historical_comparison.json` report.

The archived scientific evidence remains unchanged.

### Native C11

```bash
cmake -S native/ECS_G -B build/cgc-native
cmake --build build/cgc-native
ctest --test-dir build/cgc-native --output-on-failure
```

The native subset builds independently of the original Elpis development repository.

## 5. Research direction

CGC investigates whether compact geometric computation can provide a useful alternative to conventional feed-forward computation, particularly in recurrent learning and adaptation.

Current research directions include geometric transformations inside neural experts, recurrent latent computation, parameter-efficient adaptation, and integration with mixture-of-experts architectures.

The core itself does not use attention or maintain a KV cache. Whether CGC can replace FFN sublayers in production transformers, or contribute to alternative attention and memory architectures, requires further model-level evaluation.

## LAO: optional research learning components

CGC provides two independently tested C11 research learning companions under
[`native/LAO/`](native/LAO/README.md): the bounded **RMF3** feedback reliability
learner and **RRMF3_M**, its relational extension. On synthetic RLB1, relational
coupling improved held-out compositional regret relative to independent filters
with the same readout by −0.000255 (95% CI [−0.000280, −0.000231]).

This portability integration leaves the original 120-parameter geometric
operator unchanged. It does not demonstrate joint CGC+LAO model improvement,
certify a real measurement provider, or authorize production inference. See
[research provenance and nonclaims](docs/research/LAO_PORTABILITY.md).

## 6. Provenance and license

The original E0 PyTorch implementation is preserved under:

`src/cgc/ecsg_e0.py`

Source SHA-256:

`92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084`

The native implementation originates from the Elpis/ECS_G research project. The standalone packaging, CGC120 wrapper and portable benchmark reproduction are distinguished from the historical qualified sources.

Licensed under [Apache License 2.0](LICENSE).
