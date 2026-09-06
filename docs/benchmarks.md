# Retained L4 results — September 6, 2026

These are historical measurements of the extracted inference source, **not**
a claim that this Docker image has been independently re-benchmarked on L4.

Warm end-to-end loopback multipart-WAV HTTP, batch one, model loaded; three
warmups followed by 31 short or five hour-long requests. Servers ran serially
on NVIDIA L4. The custom runtime was refreshed September 6; the other runtimes
were measured September 5, not freshly interleaved.

| Runtime | 7.435-second clip | One-hour audio | Hour WER |
| --- | ---: | ---: | ---: |
| Last production parakeet.cpp V3 Q4_K | 46.234 ms | 14.982 s | 1.8695% |
| NeMo 2.7.3 V2, FP16 autocast, stock full attention | 46.087 ms | Not run | — |
| NeMo 2.7.3 V2, FP16 autocast, local attention ±128 | 57.371 ms | 10.223 s | 1.4318% |
| Retained L4 Rust/CUDA V2, mixed precision | 9.959 ms | 1.485 s | 1.5608% |

NeMo used PyTorch 2.8.0+cu128, FP32 stored parameters and FP16 autocast, with a
minimal HTTP adapter around `transcribe`. V3 production is a different model,
so the 10.09× hour ratio is a deployment comparison, not a same-model kernel
speedup. Matched-context V2 NeMo's hour ratio is 6.88×. Default full attention
was not measured on an hour; do not silently substitute the local configuration.

Accuracy is a real trade-off: the latest INT4 FF2 expansion increased hour WER
from the previous retained 1.4212% to 1.5608%, below the fixed 1.6200% gate.
The retained clean/other LibriSpeech WERs were 2.5830% and 7.4912%, respectively.
All eight refreshed hour HTTP responses, including warmups, matched the formal
run transcript. Full copyrighted audio and reference text are not distributed.

Separate immutable **CUDA-event** measurements confirmed the latest change
with +39.02 ms and +42.66 ms paired gain relative to the preceding incumbent.
The confirmation hour median was 1119.66 ms and short median 12.05 ms.
Model-plus-workspace accounting was 2,612,764,236 bytes (below 2.5 GiB), not
total process/driver GPU memory. LayerNorm and frontend normalization arithmetic
and barriers were preserved; mixed-precision paths are not globally bit-exact.

Do not mix HTTP and CUDA-event boundaries. Short HTTP p95 was 14.395 ms and
maximum 19.260 ms: the 16 ms correctness gate was a CUDA-event **median**, not
an HTTP tail-latency guarantee. A clean rebuild is not itself a reproduction of
these quality/performance measurements; rerun on an L4 with licensed fixtures.
