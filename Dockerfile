# syntax=docker/dockerfile:1
# Build on linux/amd64. No GPU or model is required to compile the image.
FROM rust:1.88.0-bookworm AS rust

FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS build
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ src/
COPY kernels/ kernels/
RUN cargo test --locked --all-targets \
    && cargo build --locked --release \
    && mkdir /out && cp target/release/parakeet-l4 /out/parakeet-tools \
    && cargo build --locked --release --features cuda \
    && cp target/release/parakeet-l4 /out/parakeet-l4

# Optional CPU-only model preparation image. Weights are never baked in.
FROM ghcr.io/astral-sh/uv:0.11.25 AS uv
FROM python:3.12.13-slim-bookworm AS converter
LABEL org.opencontainers.image.source="https://github.com/Marshall2HD/l4-parakeet-rs"
LABEL org.opencontainers.image.licenses="MIT"
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=build /out/parakeet-tools /usr/local/bin/parakeet-l4
COPY LICENSE /usr/share/doc/l4-parakeet-rs/LICENSE
WORKDIR /app
COPY scripts/convert_nemo_v2.py ./
# Match the converter's pinned PEP 723 dependencies, selecting CPU PyTorch.
RUN uv pip install --system --torch-backend cpu \
    'torch==2.14.0' 'numpy==2.5.2' 'PyYAML==6.0.3' \
    'safetensors==0.8.0' 'blake3==1.0.9' \
    && python convert_nemo_v2.py --help \
    && parakeet-l4 --help
ENTRYPOINT ["python", "/app/convert_nemo_v2.py"]
CMD ["--help"]

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04 AS runtime
LABEL org.opencontainers.image.source="https://github.com/Marshall2HD/l4-parakeet-rs"
LABEL org.opencontainers.image.licenses="MIT"
COPY --from=build /out/parakeet-l4 /usr/local/bin/parakeet-l4
COPY LICENSE /usr/share/doc/l4-parakeet-rs/LICENSE
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY docs/benchmarks/night-circus-1h.json docs/benchmarks/night-circus-1h.json
USER 65532:65532
EXPOSE 8080
RUN parakeet-l4 --help
ENTRYPOINT ["parakeet-l4"]
CMD ["--help"]
