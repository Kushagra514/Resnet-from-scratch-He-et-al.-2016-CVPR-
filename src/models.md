# Model Architecture

## 1. Why the architecture changed

The preliminary custom 6-layer vs 13-layer plain CNN experiment did not clearly reproduce the degradation problem. It remains useful history, but it was not closely aligned with the CIFAR architecture family used in the ResNet paper.

The main experiment now uses `PlainCIFARNet`, a paper-inspired `6n + 2` plain network. This makes depth a clearer controlled variable and prepares a later comparison with a residual network of similar structure. It does not guarantee that degradation will appear.

## 2. CIFAR architecture

```text
Input: 3 x 32 x 32
        |
3 x 3 convolution -> 16 channels
        |
Stage 1: n blocks, 16 channels, 32 x 32
        |
Stage 2: n blocks, 32 channels, 16 x 16
        |
Stage 3: n blocks, 64 channels, 8 x 8
        |
Global average pooling -> 64
        |
Linear -> 10 class logits
```

The network increases channels from 16 to 32 to 64 while reducing spatial resolution from 32 to 16 to 8. This trades some spatial detail for richer channel-wise feature representations and lower spatial computation. Stage 2 and Stage 3 begin with stride-2 convolutions; there is no max-pooling layer.

The current main model is `PlainCIFARNet(depth=20)`. The implementation also supports valid depths such as 32 and 44.

## 3. Depth convention

```text
3 stages x n blocks x 2 convolutions = 6n convolutions
1 initial convolution + 1 final fully connected layer

paper depth = 6n + 2
```

Therefore:

```text
depth 20 -> n = 3
depth 32 -> n = 5
depth 44 -> n = 7
```

`depth=30` is not used because it does not satisfy `(depth - 2) % 6 == 0`. The code validates this condition and reports the number of convolutional and paper-depth layers in its smoke test.

## 4. Plain block

Each block is:

```text
Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
```

Convolutions use `bias=False` because BatchNorm follows them immediately. There are no shortcut connections. The plain network learns the complete transformation directly:

```text
Plain:    y = H(x)
Residual: y = F(x) + x
```

A later residual model will preserve `x` through a shortcut and learn the residual modification `F(x)`. This file documents the plain baseline only.

## 5. Core layers

- **Convolution:** learns local feature detectors from neighboring pixels.
- **BatchNorm:** normalizes activations and learns scale and shift parameters, helping optimization.
- **ReLU:** adds non-linearity: `ReLU(x) = max(0, x)`.
- **Global average pooling:** converts each final feature map to one value, producing 64 features without a large fully connected spatial vector.

## 6. Interview checks

**Why change 64/128/256 to 16/32/64?**  The new widths match the paper-inspired CIFAR architecture family and make depth comparisons more controlled.

**Why does depth equal `6n + 2`?**  There are three stages, `n` blocks per stage, two convolutions per block, one initial convolution, and one final fully connected layer.

**Why increase channels as resolution decreases?**  Fewer spatial locations reduce computation, while more channels preserve richer learned representations.

**Why no MaxPool?**  Downsampling is performed by stride-2 convolutions at the start of Stages 2 and 3.

**Is this an exact reproduction?**  No. It is an educational, paper-inspired implementation, not a claim to reproduce every architecture and training detail of the original paper.

## 7. Preliminary baseline record

The original plain CNN baseline was trained for 50 epochs. Its recorded final training accuracy was 92.85% and final test accuracy was 71.28%; its best observed test accuracy was 81.30% at epoch 37. These results are retained as preliminary history and are not silently relabeled as results from `PlainCIFARNet`.
