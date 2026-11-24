# MiniRT

A from-scratch GPU inference engine for vision models, written in CUDA C++.

MiniRT loads a trained model (ResNet-18 first, a small Vision Transformer later), optimizes its computation graph, plans memory statically, and executes it with hand-written CUDA kernels. It supports FP32, FP16 and INT8 execution. Every component is benchmarked against PyTorch, cuBLAS/cuDNN and TensorRT.

The project is a learning and engineering exercise in inference infrastructure. It is not a replacement for TensorRT. The goal is to understand, implement and measure each layer of an inference stack: kernels, operator fusion, memory planning, asynchronous execution, quantization and attention.

Status: work in progress. See the [Project status](#project-status) section. Numbers in the results tables are filled in as each project completes; entries marked TBD have not been measured yet.

---

## Table of contents

1. [Goals and non-goals](#goals-and-non-goals)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Build](#build)
6. [Quick start](#quick-start)
7. [Component design](#component-design)
8. [Numerical conventions and testing](#numerical-conventions-and-testing)
9. [Benchmark methodology](#benchmark-methodology)
10. [Results](#results)
11. [Project status](#project-status)
12. [Supported operators and models](#supported-operators-and-models)
13. [Limitations](#limitations)
14. [Reproducing results](#reproducing-results)
15. [Profiling workflow](#profiling-workflow)
16. [Contributing and code style](#contributing-and-code-style)
17. [References](#references)
18. [License](#license)

---

## Goals and non-goals

### Goals

- Implement the core operators of CNN and Vision Transformer inference as standalone, tested CUDA kernels.
- Build a graph intermediate representation (IR) with shape inference and a pattern-based operator fusion pass.
- Implement a static, liveness-based memory planner that allocates all activation memory from a single arena.
- Reduce launch overhead with CUDA Graphs and overlap host-device transfers with compute using streams.
- Implement INT8 post-training quantization with calibration, per-channel weight scales and a fused requantization epilogue.
- Implement a FlashAttention-style fused attention kernel using an online softmax.
- Measure everything: correctness against a reference, latency percentiles, throughput, memory and accuracy.
- Document each optimization with before and after numbers and profiler evidence.

### Non-goals

- Training. MiniRT runs forward passes only.
- Full ONNX operator coverage. Only the operator subset needed by the supported models is implemented.
- Dynamic shapes. Shapes are static per compiled engine, which is what enables static memory planning and CUDA Graph capture.
- Competing with TensorRT across all hardware and models. The aim is to reach a stated fraction of vendor-library performance and to explain the gap.
- Portability beyond NVIDIA GPUs.

---

## Architecture

```
                  +------------------------------+
  PyTorch model   |  scripts/export_*.py         |
  (trained)  ---> |  weights + graph + reference |
                  +--------------+---------------+
                                 |
                                 v
                  +------------------------------+
                  |  graph/loader                |
                  |  ONNX or JSON -> Graph IR    |
                  +--------------+---------------+
                                 |
                                 v
                  +------------------------------+
                  |  graph/passes                |
                  |  shape inference             |
                  |  BN folding                  |
                  |  operator fusion             |
                  |  quantize/dequantize insert  |
                  +--------------+---------------+
                                 |
                                 v
                  +------------------------------+
                  |  graph/planner               |
                  |  liveness analysis           |
                  |  arena offset assignment     |
                  +--------------+---------------+
                                 |
                                 v
                  +------------------------------+
                  |  runtime                     |
                  |  kernel dispatch             |
                  |  streams, events             |
                  |  CUDA Graph capture/replay   |
                  +--------------+---------------+
                                 |
                                 v
                  +------------------------------+
                  |  kernels                     |
                  |  elementwise, reduce, softmax|
                  |  GEMM, conv, pooling         |
                  |  LayerNorm, GELU, attention  |
                  |  INT8 GEMM / conv            |
                  +------------------------------+
```

### Execution pipeline

1. **Export.** A Python script exports weights, the graph description and reference input and output tensors from a PyTorch model.
2. **Load.** The loader parses the graph into the IR and reads weights into device memory.
3. **Optimize.** Passes run in a fixed order: shape inference, constant folding of batch normalization into convolution weights, pattern-based operator fusion, and (for INT8) quantize and dequantize insertion.
4. **Plan.** The memory planner computes tensor lifetimes over the topological order and assigns each activation tensor an offset in one device arena.
5. **Build.** The runtime selects a kernel and launch configuration for each node and records the schedule.
6. **Run.** The runtime executes the schedule either kernel by kernel or by replaying a captured CUDA Graph. Input and output transfers use pinned host memory and separate streams.

---

## Repository layout

```
minirt/
  CMakeLists.txt
  README.md
  LICENSE
  core/          Device memory, tensors, weight loader, CUDA error checking
  kernels/       One kernel per file: elementwise, reduce, softmax, gemm, conv,
                 pooling, layernorm, gelu, attention, int8 variants
  graph/         IR, loaders, shape inference, fusion passes, memory planner
  runtime/       Executor, streams, CUDA Graph wrapper, serving (optional)
  quant/         Calibration, quantizers, INT8 helpers
  models/        Hardcoded reference networks (ResNet-18, ViT) used as baselines
  bench/         Benchmark harness and per-project benchmark programs
  tests/         GoogleTest unit tests, one file per kernel or module
  scripts/       Model export, reference generation, plotting, requirements.txt
  docs/          One write-up per project (p1.md ... p12.md) and images/
  .github/       CI workflow
```

Conventions:

- Each kernel has a header, a `.cu` implementation and a test file of the same base name.
- Each optimization stage of a kernel (for example, naive, tiled, register-blocked GEMM) lives in its own file so the progression stays benchmarkable.
- Public headers expose host-side launch functions only. Kernels themselves are internal.

---

## Requirements

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA GPU. FP16 tensor core paths need compute capability 7.0 or newer. INT8 tensor core paths need 7.5 or newer. |
| CUDA Toolkit | 12.x (the version used for published results is recorded next to each table) |
| Compiler | C++17 host compiler supported by the installed CUDA Toolkit |
| CMake | 3.24 or newer |
| Python | 3.10 or newer, for export and reference scripts |
| Python packages | torch, torchvision, onnx, onnxruntime, numpy, matplotlib (see `scripts/requirements.txt`) |
| Profilers | Nsight Systems and Nsight Compute |
| Optional | TensorRT, for the vendor baseline in end-to-end comparisons |

Check your GPU's compute capability:

```
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

---

## Build

```
git clone <repository-url> minirt
cd minirt

python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake options:

| Option | Default | Meaning |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Target GPU architecture. Set explicitly (for example `80`, `86`, `89`) when building for a different machine. |
| `MINIRT_BUILD_TESTS` | `ON` | Build the GoogleTest suite. |
| `MINIRT_BUILD_BENCH` | `ON` | Build benchmark programs. |
| `MINIRT_WITH_TENSORRT` | `OFF` | Enable the TensorRT baseline in end-to-end benchmarks. Requires a TensorRT install. |
| `MINIRT_LINEINFO` | `OFF` | Compile with `-lineinfo` so Nsight Compute can map metrics to source lines. |

Release builds are required for any benchmark. Debug builds are for `compute-sanitizer` and `cuda-gdb`.

---

## Quick start

These commands assume the relevant projects are complete. See [Project status](#project-status).

```
# 1. Export weights, graph and reference tensors from PyTorch
python scripts/export_resnet18.py --out data/resnet18

# 2. Run the correctness check against the PyTorch reference
./build/bench/resnet18_check --model data/resnet18 --images data/val_subset

# 3. Benchmark latency and throughput
./build/bench/resnet18_bench --model data/resnet18 --batch 1 --iters 200 --warmup 50

# 4. Run with the graph optimizer and CUDA Graphs enabled
./build/bench/engine_bench --model data/resnet18 --fuse --plan --cuda-graph --precision fp16

# 5. INT8 with calibration
python scripts/calibrate.py --model data/resnet18 --calib-images data/calib --out data/resnet18/calib.json
./build/bench/engine_bench --model data/resnet18 --precision int8 --calib data/resnet18/calib.json
```

Model weights and datasets are not committed. The `scripts/` directory contains everything needed to regenerate them.

---

## Component design

### core: memory and tensors

- `DeviceBuffer`: move-only RAII owner of a `cudaMalloc` allocation. Copying is disabled so ownership is explicit.
- `PinnedBuffer`: move-only RAII owner of page-locked host memory, used for asynchronous transfers.
- `Tensor`: shape, strides, data type (FP32, FP16, INT8, INT32) and layout tag (NCHW or NHWC), pointing into a buffer or a region of the planner's arena. `reshape` and slicing return views and never copy.
- Weight file format: a small header per tensor (name, dtype, shape, byte offset) followed by raw little-endian data. A Python exporter writes it and a C++ loader reads it. Round-trip tests verify bit-exact transfer.
- `CUDA_CHECK`: wraps every runtime call and reports file, line and the error string. Kernel launches are followed by a launch-error check, and debug builds also synchronize.

### kernels: memory-bound operators

- Elementwise add, multiply, bias and ReLU with scalar and `float4`-vectorized variants, grid-stride loops and scalar tail handling.
- Reductions (sum, max) implemented in stages: atomic baseline, shared-memory tree, warp-shuffle, and a two-pass version for large inputs.
- Row-wise softmax in the numerically stable form (subtract row maximum before exponentiation).
- These kernels are benchmarked as a percentage of the GPU's peak memory bandwidth, computed from device properties.

### kernels: GEMM

GEMM is implemented as a ladder of versions, each in its own file and benchmarked against cuBLAS:

1. Naive: one thread per output element.
2. Coalesced global access.
3. Shared-memory tiling.
4. Register blocking: each thread computes a small output tile.
5. Vectorized global loads with a layout that avoids shared-memory bank conflicts.
6. Double buffering to overlap the next tile's load with the current tile's compute.
7. Tensor core version (FP16 inputs, FP32 accumulation) using WMMA and then MMA intrinsics.

The GEMM supports a fused epilogue: bias add, optional residual add and activation applied before the final store, so these operations add no extra global memory traffic.

### kernels: convolution and pooling

- Direct convolution kernel as a reference implementation.
- `im2col` plus GEMM, using the GEMM from the previous section.
- NHWC layout support, which maps naturally onto tensor core GEMM.
- Max pooling and global average pooling.
- Batch normalization is not executed at inference time. It is folded into the preceding convolution's weights and bias offline, since at inference it is a per-channel affine transform.

### graph: IR and passes

- `Graph` owns `Node` objects (operator type, attributes, input and output tensor ids) and tensor metadata (shape, dtype, layout).
- Shape inference computes every output shape from operator semantics, so no shape is hardcoded.
- Loaders: a simple JSON format exported from PyTorch, and an ONNX loader for the supported operator subset.
- Topological ordering drives execution and memory planning.
- A pattern-matching rewrite framework replaces subgraphs with fused nodes. Implemented patterns:
  - Convolution with batch normalization (folded)
  - Convolution with ReLU
  - Convolution, residual add and ReLU
  - GEMM with bias and activation
- A fusion is applied only when every intermediate tensor removed by the fusion has exactly one consumer.
- A graph printer (text and Graphviz) shows the graph before and after each pass.

### graph: static memory planner

- Liveness analysis records, for each activation tensor, the step that produces it and the step of its last consumer.
- The planner assigns offsets inside one arena so that tensors with overlapping lifetimes never overlap in memory. The baseline strategy is greedy by size; alternative strategies are compared on peak arena size.
- Offsets are aligned to 256 bytes.
- One device allocation is made when the engine is built. There are no `cudaMalloc` or `cudaFree` calls on the inference path, which also makes CUDA Graph capture possible.
- A debug mode fills freed regions with a poison pattern and checks that no kernel reads a dead tensor.
- The planner reports the naive total (one buffer per tensor), the planned arena size and the percentage saved.

### runtime: execution

- The executor walks the planned schedule and dispatches each node to its kernel with a precomputed launch configuration.
- Input and output buffers are pinned host memory. Host-to-device copy, compute and device-to-host copy run on separate streams, ordered with events, so consecutive requests overlap.
- A forward pass can be captured once into a CUDA Graph with stream capture and replayed per request. This removes per-kernel launch overhead, which dominates latency at small batch sizes. Capture requires static shapes, static addresses and no synchronous calls, all of which the planner guarantees.

### quant: INT8

- Affine quantization with scale and zero point; symmetric quantization for weights.
- Per-channel weight scales (one per output channel) and per-tensor activation scales.
- A calibration pass runs representative inputs through the FP32 model and records activation statistics at each quantized tensor. Supported methods: min and max, percentile, and entropy (KL divergence) minimization.
- INT8 GEMM and convolution use integer tensor core MMA with INT32 accumulation.
- A fused requantization epilogue converts the INT32 accumulator to INT8: add bias, multiply by the combined scale, apply the activation, round, clamp and store.
- Layers found to be sensitive to quantization (typically the first and last) can be kept in higher precision. Sensitivity is measured by per-layer error and final task accuracy.

### kernels: transformer support

- LayerNorm with one block per row and a numerically stable variance computation.
- GELU, fused into the GEMM epilogue where possible.
- Fused attention forward kernel in the FlashAttention style: tiles of Q are held on chip, tiles of K and V are streamed through, and softmax is computed online with a running maximum and running sum. The full sequence-by-sequence score matrix is never written to global memory.
- A naive three-kernel attention (GEMM, softmax, GEMM) is kept as a correctness reference and a performance baseline.
- Graph operators for patch embedding, LayerNorm, GELU and multi-head attention allow a small pretrained ViT to run through the same loader and executor as the CNN.

---

## Numerical conventions and testing

### Tolerances

| Precision | Comparison | Tolerance (relative unless noted) |
| --- | --- | --- |
| FP32 kernels | vs CPU or PyTorch reference | about 1e-5, with a small absolute floor near zero |
| FP16 kernels | vs FP32 reference | about 1e-2 |
| INT8 integer arithmetic | vs integer reference | exact |
| INT8 end-to-end | top-1 accuracy on the validation set | within a documented drop from FP32 |

Tolerances are stated in each test and in each project write-up. When a kernel exceeds its tolerance, the cause is investigated and documented rather than the tolerance being widened.

### Test suite

- One GoogleTest file per kernel or module under `tests/`.
- Kernel tests check multiple shapes, including sizes that are not multiples of the block or tile size, odd dimensions and non-square inputs.
- Tests include adversarial numeric inputs for softmax and attention (very large and very small values) to exercise the stable formulations.
- The `compare` utility reports maximum absolute error, maximum relative error, the index of the worst element and the mismatch count, and treats NaN and Inf as failures explicitly.
- Run `compute-sanitizer` with memcheck and racecheck on every new kernel before benchmarking it.

```
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck ./build/tests/test_gemm
compute-sanitizer --tool racecheck ./build/tests/test_reduce
```

---

## Benchmark methodology

All published numbers follow the same procedure.

- Timing uses CUDA events on the executing stream. Host wall-clock time alone is never used for kernel timing.
- Each measurement runs warmup iterations first, to absorb context creation, lazy module loading and clock ramp-up.
- Each measurement records at least 100 timed iterations and reports minimum, p50, p90, p99 and maximum, not only a mean.
- For memory-bound kernels, input sizes are chosen so the data does not fit in L2 cache, or the cache is flushed between iterations.
- Achieved memory bandwidth and achieved FLOP/s are reported as a percentage of the GPU's theoretical peak, computed from device properties. The formulas are documented in `docs/p1.md`.
- Every results table records GPU model, driver version, CUDA version, compiler version and clock state (locked or default).
- Comparisons with cuBLAS use identical precision and math mode. TF32 is disabled when comparing FP32 GEMM.
- End-to-end comparisons use the same model weights, input preprocessing and batch size for MiniRT, PyTorch and TensorRT.

---

## Results

All tables below are placeholders until the corresponding project is complete. Each table will state the hardware and software environment next to it.

### Environment

| Item | Value |
| --- | --- |
| GPU | TBD |
| Compute capability | TBD |
| Driver | TBD |
| CUDA Toolkit | TBD |
| Clock state | TBD |

### Memory-bound kernels (P3)

| Kernel | Size | Achieved bandwidth (GB/s) | Percent of peak |
| --- | --- | --- | --- |
| Vectorized add | TBD | TBD | TBD |
| Softmax | TBD | TBD | TBD |
| Sum reduction | TBD | TBD | TBD |

### GEMM ladder (P4)

| Version | GFLOP/s at 2048 x 2048 | Percent of cuBLAS | Limiting factor |
| --- | --- | --- | --- |
| Naive | TBD | TBD | TBD |
| Shared-memory tiling | TBD | TBD | TBD |
| Register blocking | TBD | TBD | TBD |
| Vectorized and double buffered | TBD | TBD | TBD |
| Tensor core (FP16) | TBD | TBD | TBD |

### End-to-end ResNet-18 (P7 onward)

| Engine | Precision | Batch | Latency p50 (ms) | Latency p99 (ms) | Throughput (img/s) | Peak memory (MB) | Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | FP32 | 1 | TBD | TBD | TBD | TBD | TBD |
| PyTorch | FP16 | 1 | TBD | TBD | TBD | TBD | TBD |
| TensorRT | FP16 | 1 | TBD | TBD | TBD | TBD | TBD |
| MiniRT | FP32 | 1 | TBD | TBD | TBD | TBD | TBD |
| MiniRT | FP16 | 1 | TBD | TBD | TBD | TBD | TBD |
| MiniRT | INT8 | 1 | TBD | TBD | TBD | TBD | TBD |

### Engine optimizations (P8 to P10)

| Optimization | Metric | Before | After |
| --- | --- | --- | --- |
| Operator fusion | Kernel launches per forward pass | TBD | TBD |
| Static memory planning | Activation memory (MB) | TBD | TBD |
| CUDA Graphs | Batch-1 latency p50 (ms) | TBD | TBD |
| Stream overlap | Throughput (img/s) | TBD | TBD |

### Fused attention (P12)

| Sequence length | Naive time (ms) | Fused time (ms) | Naive peak memory (MB) | Fused peak memory (MB) |
| --- | --- | --- | --- | --- |
| 256 | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD |
| 4096 | TBD | TBD | TBD | TBD |
| 8192 | TBD | TBD | TBD | TBD |

---

## Project status

The work is organized as 12 small projects. Each ships independently, becomes a module of the engine, and has a write-up in `docs/`.

| Project | Description | Status | Write-up |
| --- | --- | --- | --- |
| P1 | Benchmark and test harness | Planned | docs/p1.md |
| P2 | Tensor library and weight loader | Planned | docs/p2.md |
| P3 | Memory-bound kernel pack | Planned | docs/p3.md |
| P4 | GEMM ladder | Planned | docs/p4.md |
| P5 | Convolution engine | Planned | docs/p5.md |
| P6 | Fused Conv + BatchNorm + ReLU | Planned | docs/p6.md |
| P7 | ResNet-18 end to end (hardcoded) | Planned | docs/p7.md |
| P8 | Graph IR and fusion pass | Planned | docs/p8.md |
| P9 | Static memory planner | Planned | docs/p9.md |
| P10 | Async runtime and CUDA Graphs | Planned | docs/p10.md |
| P11 | INT8 quantization toolkit | Planned | docs/p11.md |
| P12 | Fused attention and ViT support | Planned | docs/p12.md |

Milestones:

- Milestone 1 (after P7): a tested, benchmarked CUDA inference library running ResNet-18.
- Milestone 2 (after P10): a graph-driven engine with fusion, static memory planning and low-overhead execution.
- Milestone 3 (after P12): the full engine running CNN and ViT models in FP32, FP16 and INT8.

Optional extensions after Milestone 3: dynamic batching server, kernel autotuner, multi-GPU with NCCL.

---

## Supported operators and models

Operator coverage grows with the projects. The final target set:

| Category | Operators |
| --- | --- |
| Linear algebra | Gemm, MatMul |
| Convolution | Conv (1x1, 3x3, 5x5, 7x7; stride and padding; dilation 1) |
| Normalization | BatchNormalization (folded at load time), LayerNormalization |
| Activation | Relu, Gelu, Softmax |
| Pooling | MaxPool, GlobalAveragePool |
| Elementwise | Add, Mul |
| Shape | Flatten, Reshape, Transpose |
| Attention | Fused multi-head attention |

Target models: ResNet-18 (image classification), a small Vision Transformer (image classification). Additional CNN classifiers that use only the operators above should load without changes.

---

## Limitations

- Static shapes only. Changing the input size requires building a new engine.
- Forward pass only; no training and no gradients.
- Single GPU unless the multi-GPU extension is completed.
- Operator set is limited to what the target models require.
- Kernels are tuned for a small set of GPU generations. Performance on other architectures is not guaranteed and is not tuned for.
- Performance is expected to remain below TensorRT in general. Each results table explains the gap where it exists.
- The ONNX loader does not implement the full ONNX specification.
- INT8 accuracy depends on calibration data quality and is validated only on the documented evaluation set.

---

## Reproducing results

1. Record the environment: `nvidia-smi`, `nvcc --version`, compiler and CMake versions.
2. Build in Release mode with the architecture set explicitly.
3. Regenerate weights and reference tensors with the export scripts. Do not rely on cached files from another machine.
4. Run the benchmark programs with the documented warmup and iteration counts.
5. If possible, lock GPU clocks before benchmarking and record the setting. Otherwise, record observed clocks and expect run-to-run variation.
6. Run each benchmark at least three times and compare. If run-to-run variation exceeds the effect being measured, increase iterations or fix the environment before drawing conclusions.

Raw benchmark output is written to `bench/results/`, which is not tracked. Summary tables are copied into `docs/pN.md` with the environment recorded.

---

## Profiling workflow

Use Nsight Systems first to see the whole timeline, then Nsight Compute to inspect single kernels.

```
# Timeline: launch gaps, copies, stream overlap, CUDA Graph behavior
nsys profile -o p10_timeline --trace=cuda,nvtx ./build/bench/engine_bench --model data/resnet18 --cuda-graph

# Kernel analysis: bandwidth, occupancy, bank conflicts, stall reasons
ncu --set full -o p4_gemm ./build/bench/gemm_bench --version register_blocked --size 2048
```

Guidelines:

- Build with `MINIRT_LINEINFO=ON` to map metrics to source lines.
- Change one thing at a time and record the effect.
- Keep screenshots of the relevant profiler views in `docs/images/` and reference them from the write-ups.
- Profiler output files (`.nsys-rep`, `.ncu-rep`) are large and are not committed.

---

## Contributing and code style

This is primarily a personal learning project, but issues and suggestions are welcome.

- C++17 and CUDA C++17. Formatting is enforced with `clang-format` using the configuration in the repository.
- One kernel per file, with a header, an implementation and a test.
- Every CUDA call is checked. No raw `cudaMalloc` or `cudaFree` outside the buffer classes.
- No kernel is merged without a correctness test and a benchmark entry.
- Commit messages use a prefix for the area, for example `kernels:`, `core:`, `graph:`, `runtime:`, `quant:`, `bench:`, `docs:`.
- Each project ends with a git tag and a release containing the write-up.

---

## References

- NVIDIA, CUDA C++ Programming Guide and CUDA C++ Best Practices Guide.
- NVIDIA, Nsight Systems and Nsight Compute documentation.
- Hwu, Kirk and Hajj, Programming Massively Parallel Processors.
- S. Boehm, How to Optimize a CUDA Matmul Kernel.
- NVIDIA CUTLASS repository and documentation.
- K. He et al., Deep Residual Learning for Image Recognition, 2015.
- B. Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, 2018.
- H. Wu et al., Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation, 2020.
- A. Vaswani et al., Attention Is All You Need, 2017.
- A. Dosovitskiy et al., An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale, 2020.
- T. Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022.

---

## License

MIT License. See `LICENSE`.
