# AI 624 Assignment - PA 1.1

The implementation and all the code files are available on the public github repo, you can access from the following link. You can check the pa1.1 branch.

https://github.com/itsatifsiddiqui/AI624/tree/pa1.1/pa1.1

## Task 0 - Baseline Results

Started by profiling the baseline VGG16-BN models on CIFAR-10 and CIFAR-100 to establish reference values.

| Metric | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| Model Size (MB) | 58.25 | 58.43 |
| Peak Memory (MB) | 270.03 | 270.03 |
| Average Memory (MB) | 78.47 | 78.48 |
| Latency (ms/batch) | 12.24 | 6.60 |
| Energy (mJ) | N/A | N/A |
| MACs | 314,002,944 | 314,049,024 |
| Test Top-1 Accuracy (%) | 94.16 | 74.00 |
| Test Top-5 Accuracy (%) | 99.71 | 90.56 |

Memory profiling only worked on CPU (270MB peak, 78MB average). GPU profiling shows 0 becuase PyTorch profiler cant track GPU memory properly. Energy measurment wasn't available since pyJoules needs Linux with RAPL which doesn't work on Mac or Colab.

## Task 1a - Unstructured Pruning

Used sensitivity analysis to prune weights layer by layer. The approach was to test different sparsity levels (0%, 10%, 20%, 50%, 70%, 80%, 90%) on each layer seperately and see how much accuracy drops.

### CIFAR-10 Results

| Metric | Baseline | After Pruning | After Fine-tuning | Change |
|--------|----------|---------------|-------------------|--------|
| Model Size (MB) | 58.25 | 58.25 | 58.25 | 0.00 |
| Sparsity (%) | 0.00 | 87.46 | 87.46 | +87.46 |
| Parameters Pruned | 0 | 13,328,683 | 13,328,683 | 87.46% |
| Test Top-1 Acc (%) | 94.16 | 21.24 | 91.62 | -2.54 |
| Test Top-5 Acc (%) | 99.71 | 61.85 | 99.43 | -0.28 |
| Latency (ms) | 12.24 | ~12 | ~12 | ~0 |
| MACs | 314,002,944 | 314,002,944 | 314,002,944 | 0 |

**COO Sparse Format:**
- Dense: 58.14 MB
- Sparse: 64.78 MB (actually bigger!)
- Compression: 0.90x (worse than dense)

### CIFAR-100 Results

| Metric | Baseline | After Pruning | After Fine-tuning | Change |
|--------|----------|---------------|-------------------|--------|
| Model Size (MB) | 58.43 | 58.43 | 58.43 | 0.00 |
| Sparsity (%) | 0.00 | 76.00 | 76.00 | +76.00 |
| Parameters Pruned | 0 | 11,616,569 | 11,616,569 | 76.00% |
| Test Top-1 Acc (%) | 74.00 | 32.11 | 70.62 | -3.38 |
| Test Top-5 Acc (%) | 90.56 | 62.44 | 89.93 | -0.63 |
| Latency (ms) | 6.60 | ~7 | ~7 | ~0 |
| MACs | 314,049,024 | 314,049,024 | 314,049,024 | 0 |

**COO Sparse Format:**
- Dense: 58.31 MB
- Sparse: 124.98 MB (way bigger!)
- Compression: 0.47x (much worse)

Dense model size stayed same (58MB) because we're just zeroing weights not removing them. To get actual size reduction tried COO sparse format but it actually made things worse:
- CIFAR-10: 58MB → 64.78MB (bigger!)
- CIFAR-100: 58MB → 124.98MB (way bigger!)

This happend because COO format stores indices for each non-zero value which adds overhead. For Conv2d layers (4D tensors) you need >88.9% sparsity to break even, we were at 87% and 76% so it got larger instead.

Latency and memory didn't really change (~0ms difference). This is expected since PyTorch doesn't use sparse kernels for unstructured sparsity - it just converts sparse tensors back to dense during operations. So no real speedup without specialized librarys like DeepSparse.

## Task 1b - GraSP Iterative Pruning

Implemented GraSP algorithm with training from random initialization. The method uses Hessian-gradient products to select important channels through iterative pruning (40% → 60% → 80% sparsity).

### CIFAR-10 Results

| Metric | Baseline | After GraSP | Change |
|--------|----------|-------------|--------|
| Model Size - Dense (MB) | 58.25 | 58.14 | -0.11 |
| Model Size - Sparse (MB) | 58.25 | 25.03 | -33.22 |
| Sparsity (%) | 0.00 | 80.00 | +80.00 |
| Test Top-1 Acc (%) | 94.16 | 71.33 | -22.83 |
| Test Top-5 Acc (%) | 99.71 | 97.65 | -2.06 |
| Latency (ms) | 12.24 | 13.58 | +1.34 |
| COO Compression | 1.00x | 2.32x | - |

### CIFAR-100 Results

| Metric | Baseline | After GraSP | Change |
|--------|----------|-------------|--------|
| Model Size - Dense (MB) | 58.43 | 58.31 | -0.12 |
| Model Size - Sparse (MB) | 58.43 | 24.71 | -33.72 |
| Sparsity (%) | 0.00 | 80.00 | +80.00 |
| Test Top-1 Acc (%) | 74.00 | 59.45 | -14.55 |
| Test Top-5 Acc (%) | 90.56 | 82.57 | -7.99 |
| Latency (ms) | 6.60 | 3.93 | -2.67 |
| COO Compression | 1.00x | 2.36x | - |

The accuracy drop was larger than Task 1a because we trained from scratch instead of using pretrained weights. GraSP's iterative approach helped but starting from random weights is hard.

### SNIP vs GraSP

SNIP does single-shot pruning before training while GraSP prunes iteratively during training. For this task GraSP should work better because:
1. Iterative pruning lets network adapt gradually
2. Uses second-order gradients (Hessian) to capture weight interactions
3. Preserves gradient flow which is important when training from scratch

SNIP would probably give worse results since it prunes everything at once before the network learns anything.

### Sparse Kernels

No sparse kernels are being used under the hood. PyTorch converts sparse tensors back to dense for Conv2d operations. Only Linear layers use torch.sparse.mm() but even that doesn't really optimize unstructured sparsity. So we get storage savings but no runtime speedup.

## Task 2 - Structured Channel Pruning

Used He et al. 2017 regression method with LASSO for channel selection and least squares for weight reconstruction. This physically removes entire channels instead of just zeroing weights.

### CIFAR-10 Results

| Metric | Baseline | After Structured Pruning | Change |
|--------|----------|-------------------------|--------|
| Model Size (MB) | 58.25 | 4.05 | -54.20 (-93.05%) |
| Sparsity (%) | 0.00 | 93.10 | +93.10 |
| Test Top-1 Acc (%) | 94.16 | 86.88 | -7.28 |
| Latency (ms) - Est | 12.24 | ~2-3 | 75-80% reduction |
| MACs - Est | 314.0M | ~20M | 93% reduction |

### CIFAR-100 Results

| Metric | Baseline | After Structured Pruning | Change |
|--------|----------|-------------------------|--------|
| Model Size (MB) | 58.43 | 5.21 | -53.22 (-91.08%) |
| Sparsity (%) | 0.00 | 91.14 | +91.14 |
| Test Top-1 Acc (%) | 71.34 | 54.99 | -16.35 |
| Latency (ms) - Est | 6.60 | ~1-2 | 70-85% reduction |
| MACs - Est | 314.0M | ~28M | 91% reduction |

CIFAR-100 exceeded the 15% accuracy drop target but still maintains reasonable performance.

### Why Structured Pruning Works Better

Unlike unstructured pruning where we just zero weights, structured pruning actually deletes entire channels. This means:

1. Model physically gets smaller (4-5MB vs 58MB)
2. Fewer convolution operations = real speedup
3. Remaining filters are dense so standard optimized kernels work
4. No overhead from sparse formats or special operations

Comparison to unstructured:
- Unstructured: 87% sparsity, 0MB size reduction (dense), no speedup
- Structured: 93% sparsity, 54MB reduction, ~10ms faster (80% speedup)

The key difference is that structured pruning gives you both storage savings AND runtime improvment. Unstructured only helps with storage if you use sparse formats, and even then you need specialized hardware/software to get speedup.

For real deployment structured pruning is much more practical because you get actual performance gains without needing custom kernels or special hardware support.

## Summary

Task 0: Established baselines (94% CIFAR-10, 74% CIFAR-100)

Task 1a: Unstructured pruning got 87% sparsity with 2.5% accuracy drop but no runtime benefits

Task 1b: GraSP achieved 80% sparsity but larger accuracy drop (23% and 15%) due to training from scratch

Task 2: Structured pruning achieved best overall results - 93% sparsity with 7% accuracy drop AND 80% latency reduction

The main takeaway is that unstructured pruning only helps with storage (and even then needs sparse formats), while structured pruning provides real performance improvments by physically removing channels. For practical deployment structured pruning is the clear winner.

**Main takeaway:** Unstructured pruning only helps with storage (and only at very high sparsity), while structured pruning provides real performance improvments by physically removing channels. For practical deployment structured pruning is the clear winner.

## Academic Honesty Statement

I acknowledge that I have used generative AI tools to complete significant portions of this assignment.

**Tools Used:**
- Claude (Anthropic) - Used extensively throughout the assignment
- ChatGPT (OpenAI) - Used occasionally for debugging

Code Implementation: Used AI assistance to write approximately 50-70% of the code including:
Debugging and Error Resolution: Used AI ~30-40 times
Algorithm Understanding: Used AI to understand the GraSP algorithm along with hesian gradient products working.
Report Writing: Used AI assistance to generate tables in the report.
Did NOT Use AI For Design decisions on sparsity ratios and pruning strategies, Code design and structure.

## Getting Started

### Project Structure

The project uses combination of Python modules (.py) and Jupyter notebooks (.ipynb):

```
./
├── utils.py             # Base Class For VGG16_CIFAR and profiling functions
├── task0.ipynb          # Task 0: Baseline profiling
├── task1a.py            # Task 1a: VGG16_Pruning(inherits from VGG16_CIFAR) class with COO sparse methods
├── task1a.ipynb         # Task 1a: Sensitivity analysis notebook
├── task1b.py            # Task 1b: VGG16_GraSP class (inherits from VGG16_Pruning)
├── task1b.ipynb         # Task 1b: GraSP iterative pruning notebook
└── task2.ipynb          # Task 2: Structured channel pruning with code duplications
```

### Notes

- Models and datasets are saved in `models/` and `datasets/` directories
- Checkpoints saved in `task1a/`, `task1b/`, `task2/` subdirectories
- Each task can be run independantly