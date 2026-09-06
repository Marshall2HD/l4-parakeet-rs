# l4-parakeet-rs

This is an NVIDIA L4 (sm_89), Linux x86_64 Rust/CUDA inference engine.
The retained kernels are hardware-specific, not portable CUDA examples.

- Preserve LayerNorm and frontend normalization arithmetic, reduction order,
  and synchronization barriers. Precision changes need model quality tests.
- Do not change benchmarks or quality thresholds to retain an optimization.
- Keep Cargo.lock and use `--locked` for builds and tests.
- No GPU/model is required for unit tests or Docker builds. Actual inference
  must be tested on an L4; a successful CPU build does not validate GPU results.
- Keep model weights, copyrighted benchmark audio/text, credentials, and local
  research state out of Git and the Docker build context.
- Retain the MIT copyright notice. AI-assisted commits use an `Assisted-by:`
  trailer; assistants must not add DCO `Signed-off-by` or AI `Co-Authored-By`.
