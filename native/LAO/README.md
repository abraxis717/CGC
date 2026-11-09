# Portable native LAO — research only

Two independently linkable, source-identical C11 learners copied from
[Elpis_DEV commit `76563a9d43f110a8f4a8bd299b949d2408420844`](https://github.com/abraxis717/Elpis_DEV/commit/76563a9d43f110a8f4a8bd299b949d2408420844).

- `rmf3/`: feedback reliability-mixture learner (native state 480 bytes); synthetic SLB2 v4 qualification.
- `relational/`: bounded shared-reliability relational learner (state 1016 bytes, config 128 bytes, 8 slots);
  synthetic RLB1 qualification.

The native code does not include the qualified real-provider admission host, ECS_C
source authority or its revocation policy. The kernels alone are NOT a production
admission, a verified integration with the CGC geometric operator, or evidence
of new learned weight-update dynamics. Frozen weights remain immutable;
learner episode state is volatile and bounded.

The original scientific results, limitations, and post-freeze host correction:
- [SLB2 v4 final report](https://github.com/abraxis717/Elpis_DEV/blob/76563a9d43f110a8f4a8bd299b949d2408420844/research/cgc_lao/solver_lab/FINAL_REPORT.md)
- [RLB1 final report](https://github.com/abraxis717/Elpis_DEV/blob/76563a9d43f110a8f4a8bd299b949d2408420844/research/cgc_lao/relational_lao_r0/FINAL_REPORT_R0.md)
- [RLB1 freeze](https://github.com/abraxis717/Elpis_DEV/blob/76563a9d43f110a8f4a8bd299b949d2408420844/research/cgc_lao/relational_lao_r0/FROZEN_R6.json)
- [Host correction disclosure](https://github.com/abraxis717/Elpis_DEV/blob/76563a9d43f110a8f4a8bd299b949d2408420844/research/cgc_lao/relational_lao_r0/evidence/POSTFREEZE_R6.json)

The RLB1 correlation admission grid supports rho <= 0.25 under its synthetic
conditions; it is not a certified real-provider bound. Its reliability coupling
can hurt under mismatched/noisy regimes. HCA salience projection had no positive
value and is not included.

Build and test with `cmake -S . -B /dev/shm/cgc-lao-build -DBUILD_TESTING=ON`,
`cmake --build /dev/shm/cgc-lao-build`, and
`ctest --test-dir /dev/shm/cgc-lao-build --output-on-failure`.

First-party Elpis native source is covered by Apache-2.0;
preserve the upstream [NOTICE](https://github.com/abraxis717/Elpis_DEV/blob/76563a9d43f110a8f4a8bd299b949d2408420844/NOTICE).
