# Container verification — September 6, 2026

The published Dockerfile was built on an existing Linux x86_64 Docker host
without an NVIDIA GPU. This was an actual clean compilation, not a Dockerfile
lint or a copy of the research executable into an image.

- Runtime target: all 29 CPU unit tests passed; CPU tools and CUDA-enabled
  release binary compiled with Rust 1.88.0 and CUDA 13.0.1.
- Runtime container: `--help` and `--version` succeed as UID 65532. cuBLAS,
  cuBLASLt, cuFFT and NVRTC are present; no CUDA compiler is in the runtime.
- Converter target: Python 3.12.13, CPU PyTorch 2.14.0, and the script's pinned
  direct dependencies install successfully. Both conversion and packing CLIs run.
- Full pinned 2,472,222,720-byte checkpoint converted and packed inside the
  converter container as a non-root user, with no GPU, a 10 GiB memory limit
  and four CPU cores.
- Result: 701 tensors, 1,236,264,960 payload bytes, 1,236,383,744 file bytes.
  The rebuilt `.pkl4` has BLAKE3
  `0bed87c0913040a38ccdb303e19ccdfacdd9aaf7d5281b054fed7f49a3b2cb1b`,
  matching the original benchmark artifact. Weights are not published here.
- Local unit tests, formatting and Clippy also passed. The 28 exported
  Rust/build/kernel files match the retained inference source byte-for-byte;
  the only converter source change is its Python-minimum metadata correction.

This verifies the source → image and checkpoint → artifact rebuild paths.
It does **not** constitute a new L4 inference benchmark, sanitizer run, or
quality evaluation of the Docker image. The retained source's earlier L4
measurements and their caveats are in [benchmarks.md](benchmarks.md).
