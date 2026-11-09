# CGC–LAO native portability R1

**Research-only, no runtime admission.** Six native implementation, header and
test files were copied byte-identically from Elpis_DEV revision
`76563a9d43f110a8f4a8bd299b949d2408420844`. Their SHA-256 digests are in
[`native/LAO/SOURCE_SHA256SUMS.txt`](../../native/LAO/SOURCE_SHA256SUMS.txt).

The original CGC/ECS_G source, PyTorch public API and 120-parameter reference
remain unchanged. These independently linkable kernels do not train the CGC
core and do not implement source-authority admission or provider certification.

The frozen RLB1 synthetic relational holdout reported a compositional regret
gain over independent per-edge RMF of -0.000255
(95% CI [-0.000280, -0.000231]); this is not a CGC+LAO joint result.

Qualification records and research limitations remain upstream:
[Relational LAO final report](https://github.com/abraxis717/Elpis_DEV/blob/76563a9d43f110a8f4a8bd299b949d2408420844/research/cgc_lao/relational_lao_r0/FINAL_REPORT_R0.md).

**Next independent experiment:** test LAO-controlled observations of a frozen
CGC geometric operator against matched controls. No persistent inference-time
weight mutation, no unbounded memory and no production integration.
