# l4-parakeet-rs

NVIDIA **L4-specific** Parakeet TDT speech recognition in Rust and CUDA.
No Python, PyTorch, NeMo, or ggml is required at inference time. The executable
is named `parakeet-l4`; the GitHub repository is `Marshall2HD/l4-parakeet-rs`.

This standalone extraction retains the best validated L4 implementation from
the parakeet.cpp research branch, including mixed FP16/FP8/INT8/INT4 kernels,
local attention, and activation-sparse FF2. It is **not** an all-FP16 reference
implementation and does not claim numerical parity with stock NeMo.

## Requirements

- **Inference:** Linux x86_64, NVIDIA L4 (24 GB), a CUDA 13-compatible NVIDIA
  driver, Docker, and NVIDIA Container Toolkit. Tested research driver: 595.71.05.
- **Build:** Docker on Linux x86_64; neither a GPU nor model weights is needed.
  Allow roughly 20 GB free disk for build images/caches and conversion.
- **Model:** the pinned English `nvidia/parakeet-tdt-0.6b-v2` checkpoint below.
  V3, arbitrary GPUs, CPU transcription, streaming, and batch inference are not
  supported deployment targets. Requests are processed serially.
- **Audio:** 16 kHz, mono, signed 16-bit PCM WAV. The server does not run ffmpeg
  or accept arbitrary audio codecs. Maximum request duration defaults to one hour.

## Build the container

GitHub Actions builds and smoke-tests both images, then publishes successful
`main` builds to GHCR:

```sh
docker pull ghcr.io/marshall2hd/l4-parakeet-rs:latest
docker pull ghcr.io/marshall2hd/l4-parakeet-rs:converter
```

Runtime builds also receive `sha-<full-commit>` tags; converter builds receive
`converter-sha-<full-commit>` tags. Pin a registry digest for deployments rather
than following `latest`. Images contain code and dependencies, **not model weights**.
The MIT license covers this project's code; bundled dependencies and NVIDIA CUDA
remain subject to their own licenses. Pull requests build/test without publishing.

To build from source instead:

```sh
git clone https://github.com/Marshall2HD/l4-parakeet-rs.git
cd l4-parakeet-rs
docker build --platform linux/amd64 --target runtime -t l4-parakeet-rs:local .
docker run --rm l4-parakeet-rs:local --help
```

The multi-stage build uses Rust 1.88.0, Cargo.lock, and CUDA 13.0.1. It runs the
CPU unit tests, compiles all five CUDA translation units into sm_89 cubins, and
embeds them in the executable. The runtime image includes cuBLAS/cuFFT, not a
compiler. Build on an x86_64 machine; `--platform` alone does not install
emulation on an ARM host. Dependency versions are pinned, but mutable registry
tags and Ubuntu package repositories mean this is not a bit-reproducible image.

## Prepare the model (one time, no GPU)

Model files are not included in Git or the image. Obtain the upstream checkpoint
subject to its license. The converter rejects other revisions or modified bytes.

```sh
mkdir -p models
curl --fail --location --retry 3 \
  'https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/ae9ad07059c7c739ffaf932226a8fe64ae2620b0/parakeet-tdt-0.6b-v2.nemo' \
  -o models/parakeet-tdt-0.6b-v2.nemo

docker build --platform linux/amd64 --target converter -t l4-parakeet-rs:converter .
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models" l4-parakeet-rs:converter \
  --nemo /models/parakeet-tdt-0.6b-v2.nemo --output-dir /models/v2-normalized
docker run --rm --user "$(id -u):$(id -g)" --entrypoint parakeet-l4 \
  -v "$PWD/models:/models" l4-parakeet-rs:converter pack-fp16 \
  --model-dir /models/v2-normalized --output /models/v2-fp16-sm89.pkl4
```

The source file is 2,472,222,720 bytes, BLAKE3
`51929378d4f9a301ca00a964fece355f46dab3ffb093e5e7f516ecd7365e7c84`.
Conversion validates this before deserializing the checkpoint with
`torch.load(weights_only=True)`. The output directory must not already exist.
Allow at least 12 GB RAM and additional temporary disk space for conversion.
The `.pkl4` artifact contains vocabulary and frontend data as well as weights;
only this artifact is needed at inference. Despite the `pack-fp16` name, the
runtime derives its retained mixed-precision layouts when loading the artifact.

Alternatively, `uv run --script scripts/convert_nemo_v2.py --help` provides the
offline conversion CLI without Docker; PyTorch is confined to conversion.

## Transcribe or serve

```sh
# Convert your audio beforehand if necessary:
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav

docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" -v "$PWD/input.wav:/audio.wav:ro" \
  l4-parakeet-rs:local transcribe --artifact /models/v2-fp16-sm89.pkl4 \
  --input /audio.wav --warmup-iterations 0 --measured-trials 1

docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v "$PWD/models:/models:ro" l4-parakeet-rs:local serve \
  --artifact /models/v2-fp16-sm89.pkl4 --host 0.0.0.0 --port 8080
```

From another terminal:

```sh
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'file=@input.wav;type=audio/wav' -F response_format=json
```

The HTTP API implements a subset of OpenAI-compatible transcription requests,
not a full Whisper replacement. There is no built-in authentication or TLS;
keep it loopback-only or put an authenticated reverse proxy in front of it.
The container runs as UID/GID 65532; mounted model/audio files must be readable.

## Development and verification

```sh
cargo test --locked --all-targets
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
# Native CUDA build, on Linux x86_64 with CUDA 13 installed:
LIBRARY_PATH=/usr/local/cuda/lib64 cargo build --locked --release --features cuda
```

CI rebuilds the runtime and converter images without a GPU. That verifies
compilation and container startup, **not** model quality or L4 performance.
The [container verification](docs/container-verification.md) also exercised the
full checkpoint-to-artifact rebuild and matched the benchmark artifact exactly.
See [benchmarks](docs/benchmarks.md) for measured results and limitations, and
[provenance](docs/provenance.md) for the extraction boundary and licensing.

MIT code license; upstream model and benchmark data licenses remain separate.
