Assignments for AI 624 (AI For Edge Devices)

## PA 1.1 - Neural Network Pruning

Implementation of various pruning techniques on VGG16-BN models for CIFAR-10 and CIFAR-100:

- **Task 0**: Baseline profiling (94.16% CIFAR-10, 74.00% CIFAR-100 accuracy)
- **Task 1a**: Unstructured pruning with sensitivity analysis (87.46% sparsity, -2.54% accuracy drop)
- **Task 1b**: GraSP iterative pruning from scratch (80% sparsity, structured channel removal)
- **Task 2**: Structured channel pruning with LASSO regression (93.10% sparsity, 93% size reduction, ~80% latency reduction)

**Key Results**: Structured pruning achieved 93% model size reduction (58MB → 4MB) with 7.28% accuracy drop on CIFAR-10, demonstrating significant practical advantages over unstructured approaches for deployment.

[Full details →](pa1.1/README.md)