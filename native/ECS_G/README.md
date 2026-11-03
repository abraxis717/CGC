# Native ECS_G reference subset

This self-contained C11 library contains the original mathematical, state, and
executor sources copied from the Elpis ECS_G lineage. The local CMake build
is a **new portable build definition**, not the historical Elpis build.

```sh
cmake -S native/ECS_G -B build/cgc-native
cmake --build build/cgc-native
ctest --test-dir build/cgc-native --output-on-failure
```

The native K1/FMS runtime and its external substrate/hash dependencies are **not**
part of this portable subset. Historical K1 result reports, where included,
are archival research context and are not presented as reproducible by this subset.
