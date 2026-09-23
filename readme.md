# ResNet Paper Implementation

Educational implementation of the central idea from He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016): why deeper plain networks can become harder to optimize and how residual connections address that formulation.

## Current architecture

The main baseline is `PlainCIFARNet(depth=20)`, a paper-inspired CIFAR-10 network using the `depth = 6n + 2` structure. It has three stages with 16, 32, and 64 channels, stride-2 convolution at stage transitions, and no skip connections.

## Current experiment

The training entry point uses CIFAR-10, cross-entropy loss, SGD with momentum, batch size 128, and 50 epochs. Results are saved to `experiments/results/plain_cifar20.csv`.

## Project history

The original custom 6-layer vs 13-layer plain CNN experiment did not clearly show degradation. Its code and result files remain as preliminary history; the main experiment now uses the more paper-aligned CIFAR architecture. This is not a claim of exact reproduction of every paper training detail.

## Repository structure

- `src/models.py`: preliminary models and configurable plain CIFAR model
- `src/data.py`: CIFAR-10 loaders and preprocessing
- `src/train.py`: training and evaluation loop
- `experiments/results/`: experiment CSV files
- `src/models.md`: architecture revision notes
- `src/train.md`: experiment and training revision notes

## Status

The plain depth-20 baseline is implemented and verified for `[8, 10]` output on synthetic input. Residual models and deeper comparisons are future experiment steps.