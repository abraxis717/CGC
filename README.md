# CGC — Cubic Geometric Core

Research implementation of a compact cubic geometric operator originating
from Elpis/ECS_G.

## Portable PyTorch reference

`src/cgc/ecsg_e0.py` preserves the original E0 PyTorch source byte-for-byte
(SHA-256 `92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084`).
A scalar E0 channel of dimension 5 and width 6 has 30 trainable weights.
`src/cgc/four_channel.py` provides a **new** four-channel wrapper, `CGC120`,
with 120 trainable core weights before external embedding/projection layers.

The frozen E0 operator is:

```text
phi(z) = (z + z^2 + z^3)/2
f_W(x) = sum_i phi(x^T w_i)
```

With a Python environment containing PyTorch:

```sh
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Portable native C11 subset

```sh
cmake -S native/ECS_G -B build/cgc-native
cmake --build build/cgc-native
ctest --test-dir build/cgc-native --output-on-failure
```

The CMake definitions in this repository are standalone and require no Elpis
build system or sibling project. Build outputs are ignored by Git.

## Scientific evidence and boundaries

- R97O: original recorded differentiable multistep parity result is included.
- R97W: original result and slow-state checkpoint are included. The historical
  path-dependent source/spec/DEV record are **not** part of this portable release;
  the benchmark cannot yet be rerun from this repository alone.
- Historic native K1 retention reports are included only as context; K1/FMS
  is **not** part of this portable native subset.
- Parity and task-specific learning results do not establish general FFN
  replacement, language capability, or a KV-free attention substitute.

The R97W experiment and qualification phase runners will be ported as new
independent experiments rather than altering frozen historical scientific records.

## Provenance

The native ECS_G implementation originates from the Apache-2.0 licensed
Elpis_DEV project. The E0 PyTorch file is copied unchanged from the original
RALPH R97O research payload. New packaging files and portable CMake definitions
are distinguished from historical qualified sources.
