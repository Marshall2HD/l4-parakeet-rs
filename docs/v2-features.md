# V2 offline batching and timestamps

These features are local feature-branch work, not a production deployment or a
new registry release. The base retains main's GHCR packaging and the seven
previously unmerged L4 optimization commits. Model weights and precision
policies are unchanged; this is still the L4-specific mixed-precision runtime.

## Checkpoint capabilities versus runtime support

[NVIDIA's V2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
specifies English ASR, punctuation, capitalization, and word timestamps. It
describes a full-attention FastConformer/TDT model and batched offline
transcription. Its leaderboard batch-128 throughput is not an L4 promise.

| Capability | This implementation |
| --- | --- |
| English, punctuation, capitalization, spoken numbers | Retained checkpoint predictions; no extra text-normalization model |
| Explicit batching | Packed GPU encoder, per-input frontend and cooperative TDT decoder |
| Token/word times | Emission frames plus predicted TDT duration, not uniform text spacing |
| SRT / VTT | Word-based display cues using model times, not a whole-clip placeholder |
| Multilingual / language detection / translation | Not implemented; V2 is English, not V3. Non-English language requests are rejected |
| True streaming / live incremental partials | Not implemented; offline chunked computation does not establish causal streaming |
| Diarization / speaker labels / forced alignment | Not provided by this checkpoint/runtime |
| FLAC and arbitrary codecs | Convert externally; the runtime accepts 16 kHz mono PCM16 WAV only |
| Full-attention equivalence to NeMo | Not claimed; the retained runtime uses local attention |

## Batch contract

`POST /v1/audio/transcriptions/batch` is an explicit extension, **not** an
OpenAI-standard endpoint. Send repeated `file` multipart fields. No other fields
are accepted on this endpoint. It returns JSON with an ordered `results` array:
each result includes `text`, `audio_seconds`, `token_ids`, `tokens`, and `words`.
The entire request is validated before inference; invalid inputs or capacity
violations return HTTP 400. The existing upload-byte and per-input duration
limits still apply. A rejected request does not leave a batch mask installed.

```sh
curl --fail http://127.0.0.1:8080/v1/audio/transcriptions/batch \
  -F file=@first.wav -F file=@second.wav

parakeet-l4 transcribe-batch --artifact model.pkl4 \
  --input first.wav second.wav --warmup-iterations 5 --measured-trials 31

# Opt-in strict diagnostic: fail if any measured result differs from singles.
parakeet-l4 transcribe-batch --artifact model.pkl4 \
  --input first.wav second.wav --verify-single
```

The Rust API is `PipelineEngine::transcribe_batch(&[&[f32]])`; samples must be
16 kHz mono. Load the engine with a capacity at least as large as its longest
input. `BatchTranscription` and `PipelineTranscription` are public exports.

Limits for a multi-input batch:

- 2–16 inputs, each using the short-clip precision policy: fewer than 512 padded
  encoder rows (approximately 40 seconds per clip).
- At most 1,008 total packed rows. For each input, compute
  `frames = ceil((floor(samples / 160) + 1) / 8)`, then reserve
  `round_up(frames, 16) + 16` rows. Sum these reservations.
- Sixteen zero guard rows isolate the width-nine convolution. Attention uses
  an explicit per-row start/end interval; padding has an empty interval.
- A singleton uses the existing single-input path, including long-form
  processing. It does not pay the packed-encoder cost.

Frontend normalization remains independent per utterance. All 24 encoder
layers process the packed tensor in shared GPU launches, including GEMMs;
attention cannot attend to another utterance and masked GLU guards prevent
convolution leakage. The short FP16 attention path is retained even when the
combined batch is longer. Long-form fused/quantized kernels are unchanged.
The cooperative decoder still occupies the GPU for **one utterance at a time**;
this is encoder batching, not batched LSTM decoding, multi-stream inference,
or serial queueing disguised as batching. Separate HTTP requests are serialized;
there is no automatic cross-request batching or batch-wait timer.

Results need not be bit-identical to individual inference: larger GEMM shapes
can select different accumulation paths. Check actual quality rather than
assuming mathematical independence guarantees floating-point parity. The
strict `--verify-single` and benchmark harness deliberately report mismatches.

## Timestamp semantics

Each token includes its SentencePiece `piece`, `id`, integer `start_frame` and
`duration_frames`, and `start`/`end` in seconds. Following
[NeMo's TDT offset calculation](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_decoding.py),
the offsets are `start_frame` and `start_frame + duration_frames`, with an
80-ms encoder step (10-ms frontend hop × 8 subsampling). Seconds are clipped to
the clip duration; raw integer predictions are retained. A zero-duration token
remains zero-duration. The next token's start is **not** substituted for its end.

SentencePiece word-boundary markers group subwords; closing punctuation remains
with its word. Standalone separator tokens do not set the next word's start;
its first nonempty subword does, matching NeMo's word-offset convention.
A word ends at its last constituent token.
`verbose_json` includes token metadata and word-based `segments`; add
`timestamp_granularities[]=word` to include `words`. Plain JSON and text retain
their existing response shapes. Native `transcribe` reports also include times.

SRT/VTT display cues merge zero-duration words with adjacent cues without
inventing offsets. A prediction with no positive span at all cannot produce a
timed display cue, but remains available in JSON. These are model predictions
on an 80-ms grid, not sample-accurate forced alignment, character-level acoustic
alignment, or a fresh proof of NVIDIA's timestamp-accuracy claim. No comparison
against manually aligned word references or stock NeMo timestamps was made.

## Measurement boundaries and reproduction

- `cuda_stage_ms`: sum of CUDA-event frontend, packed encoder, and decoder
  intervals. Excludes sample uploads, packing copies between stages, output
  readback, host metadata generation, and HTTP. It can include host dispatch
  gaps on the CUDA stream; it is not a sum of isolated kernel durations.
- `encoder_latency_ms`: the shared encoder interval. Zero for a singleton,
  where the existing whole-pipeline interval is returned instead.
- `wall_latency_ms`: the engine batch call, including its allocation/upload,
  readback, and metadata work, but not WAV parsing or response serialization.
- Per-result `inference_latency_ms` is the **shared whole-batch** CUDA-stage
  value, not an independent per-item time. Do not sum it over results.
- HTTP measurements include the complete request/response. Client WAV reads
  and multipart construction happen before timing. Native single-input CLI
  measurements exclude initial upload and final readback, as before.
- `model_and_workspace_bytes` accounts for explicit resident allocations;
  it is not total NVML process VRAM or a continuously measured allocation peak.

The standard-library harness accepts a JSON list of `id`, `audio` (relative to
the manifest), `duration_seconds`, and optionally `split` and `reference`.
It groups within the row limit, preserves all trials, alternates single/batch
order, and writes raw results including transcripts. **Keep those results
private** when using private or copyrighted material.

```sh
python3 scripts/benchmark_batch.py --url http://127.0.0.1:8080 \
  --manifest corpus/manifest.json --output batch-results.jsonl \
  --batch-size 8 --shuffle-seed 20260908
python3 scripts/test_batch_api.py --url http://127.0.0.1:8080 --speech speech.wav
```

The API test uses the public 7.435-second parakeet.cpp `speech.wav`, or a
similarly short fixture that fits eight times in the packed workspace. It
checks singleton parity, mixed lengths, silence, sub-hop inputs, frame/tile
boundaries, result order, replacement of neighboring speech with silence,
dirty-buffer reuse, the 16-item path, and rejected requests.

## L4 validation — September 8, 2026

Reference: main's packaging plus the seven retained optimization commits from
`autoresearch/fable-l4-20260906`, **not** the older deployed main binary, V3, or
stock NeMo. Candidate: this feature implementation. Both were built with Rust
1.88.0 and CUDA 13.0.1 using separate Cargo target directories and the same
pinned artifact. No image was published or production deployment changed.

Hardware: NVIDIA L4, driver 595.71.05, stock dynamic clocks and 72 W power limit,
AMD Ryzen 9 9955HX host. The temporary pod had four CPUs and 12 GiB host memory.
Production remained resident but logged no inference during validation. Only
one benchmark inference workload was driven at a time. Both benchmark servers
were resident for HTTP measurements and stopped before native measurements.
This was a shared host, not an isolated latency laboratory.

### Quality and independence

Complete LibriSpeech test-clean and dev-other: 5,484 utterances, 10.52465 hours,
three outputs per utterance (reference, candidate single, candidate batch), no
missing/duplicated IDs or failed HTTP requests in the final corpus run. Scoring
lowercases, deletes ASCII punctuation, splits on whitespace, and sums word
Levenshtein edits (RapidFuzz); no hypothesis-dependent reference selection.

| Split | Reference words | Reference / candidate single edits | Candidate batch edits | Single WER | Batch WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| test-clean, 2,620 utterances | 52,576 | 991 | 991 | 1.8849% | 1.8849% |
| dev-other, 2,864 utterances | 50,948 | 1,608 | 1,607 | 3.1562% | 3.1542% |

Every candidate **single** transcript matched the reference. **91 batched
token sequences differed** from singles (45 clean, 46 other); the strict parity
harness correctly exited nonzero. Similar corpus WER is not bitwise parity or
proof of accuracy on other domains. Replacing every neighboring utterance with
same-length silence preserved all token IDs and predicted times in each of
those 91 cases. The dedicated order/padding/dirty-buffer API tests also passed.
No thresholds were relaxed. These corpora overlap earlier optimization work;
they are not a new blind holdout.

The shuffled corpus formed 717 mixed-length batches: 549 × 8 items, 98 × 7,
57 × 6, 12 × 5, and 1 × 4. Summed timed batch HTTP intervals were 34.560 seconds
versus 73.558 seconds for separate candidate verbose-JSON requests: **2.13×
throughput** in those timed intervals. This excludes client preparation,
inter-request gaps, and waiting to accumulate a batch; it is not campaign wall
time or production queue latency.

### HTTP latency versus throughput

Loopback in the same pod. Three blocks, reference/candidate order AB/BA/AB.
Three warmups per fixture/block, 31 measured trials per short fixture/block,
seven per hour/block. All trials retained. P95 uses nearest rank.

| Single input | Reference median / p95 | Candidate median / p95 |
| --- | ---: | ---: |
| 7.435 seconds | 13.45 / 18.24 ms | 13.61 / 20.59 ms |
| 30 seconds | 27.18 / 34.49 ms | 27.48 / 34.54 ms |
| 1 hour | 1.136 / 1.144 s | 1.134 / 1.162 s |

These observed medians differ by about +1.1%, +1.1%, and −0.2%. Short p95 was
higher, and block-to-block variation was substantial; do not claim a proven
zero regression or a tail-latency guarantee. The earlier reported ~1.02-second
hour result was not reproduced under these measurement conditions.

Each item in the next table is the same 7.435-second fixture; 93 trials per
batch size, with full token/word metadata in the response.

| Items | Batch HTTP median / p95 | Audio seconds per HTTP second |
| ---: | ---: | ---: |
| 1 | 13.69 / 18.05 ms | 543× |
| 2 | 19.00 / 25.84 ms | 783× |
| 4 | 31.25 / 42.81 ms | 952× |
| 8 | 54.91 / 61.78 ms | 1,083× |

Eight items approximately double throughput versus separate single requests,
but each result waits for the whole ~55-ms batch. Dividing that by eight is
amortized work per item, **not per-request latency**. Corpus quality above uses
independent mixed-length utterances rather than this repeated timing fixture.

### Separate CUDA-event measurements and memory

Three native invocations per implementation/fixture, five warmups, then 31
trials each (seven for the hour): 93 short/30-second and 21 hour intervals.
Final native token IDs matched between implementations on all three fixtures.

| Input | Reference event median / p95 | Candidate event median / p95 |
| --- | ---: | ---: |
| 7.435 seconds | 10.47 / 16.16 ms | 10.10 / 16.25 ms |
| 30 seconds | 25.30 / 31.45 ms | 25.14 / 29.99 ms |
| 1 hour | 1.084 / 1.104 s | 1.092 / 1.110 s |

Independent native batch calls (31 trials/size, five warmups) had CUDA-stage
sum medians of **13.70, 16.68, 27.42, 47.44 ms** for sizes 1/2/4/8; engine wall
medians were **14.13, 17.68, 28.54, 49.41 ms**. All repeated-fixture batch token
parity checks passed. These are not HTTP measurements or pure kernel timings.

NVML sampling (1,421 samples, 250-ms pause plus query overhead) observed a
**2,740 MiB process peak for both implementations**. This is a sampled peak,
not proof that no shorter allocation spike occurred. Native hour explicit
model/workspace accounting rose from 2,629,541,452 to 2,633,141,452 bytes
(+3,600,000 bytes for emission/duration planes); it remains below 2.5 GiB, but
**total process VRAM does not**. The short native workspace reserves packed
encoder capacity, adding about 13.6 MiB versus the reference; the batch staging
buffer adds a further 2,064,384 bytes on first batch use.

Verification: 38 CPU tests, 44 CUDA-feature build/unit tests, formatting and CPU
Clippy passed, alongside live L4 API/CLI tests and the corpus/latency work above.
Strict CUDA-feature Clippy still reports 11 pre-existing lints in the retained
kernel wrappers/frontend (argument counts, tuple complexity, slice sizing, and
a redundant mutable reference); those unrelated APIs were not refactored.
Final metadata
fixes merge zero-duration subtitle cues and skip standalone separators when
choosing word starts; these do not change inference/token predictions. Final
SRT/VTT positive-span checks and all 91 isolation checks passed after rebuilding.
No new Docker image was built/published, no multilingual/streaming behavior
was inferred, and no manually aligned timestamp-accuracy evaluation was done.
Audio, references, transcripts, raw timings, and telemetry remain private.
