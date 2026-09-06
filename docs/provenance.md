# Provenance

The initial source snapshot is the Rust/CUDA subset of the private local
`Marshall2HD/parakeet.cpp` research checkout at commit
`2d0390160e7d0d2ad4864c28162a59bb39076844`, retained September 6, 2026.
It was extracted without rewriting or modifying that repository.

Included: Cargo manifest/lockfile, build.rs, Rust modules, five sm_89 kernels,
the pinned NeMo v2 conversion script, and the historical evaluator manifest.
Excluded: legacy C++/ggml, submodules, infrastructure scripts, local research
logs, model weights, transcripts, audio, credentials, and private Git history.
The historical manifest's machine-local locators were removed; numerical
scoring fields were preserved. Its fixtures are not distributed here.
The converter's Python minimum was corrected to 3.12, as required by its
already-pinned NumPy version. Conversion arithmetic and model checks are unchanged.

The code retains the original MIT license and copyright notice for the
parakeet.cpp authors. This repository is independently named and is not an
official NVIDIA or upstream parakeet.cpp release. The initial export preserves
the retained inference code; packaging does not introduce a new kernel candidate.

The checkpoint is NVIDIA's `nvidia/parakeet-tdt-0.6b-v2`, revision
`ae9ad07059c7c739ffaf932226a8fe64ae2620b0`. Its license is separate from this
repository. Downloads and redistribution must follow the model's terms.
