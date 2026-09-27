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

The implementation provides both `PlainCIFARNet` and `ResNetCIFAR` with the
same CIFAR-style stage layout. The recorded residual experiments use depths
20, 32, and 50.

## 3. Depth convention

```text
3 stages x n blocks x 2 convolutions = 6n convolutions
1 initial convolution + 1 final fully connected layer

paper depth = 6n + 2
```

Therefore:

```text
ResNet-20 -> n = 3 -> 3 residual blocks per stage
ResNet-32 -> n = 5 -> 5 residual blocks per stage
ResNet-50 -> n = 8 -> 8 residual blocks per stage
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

The residual implementation preserves the input through a shortcut and learns
the residual modification:

```text
Residual: y = F(x) + shortcut(x)
```

When the input and output shapes match, `shortcut(x)` is `nn.Identity()`. At
Stage 2 and Stage 3 transitions, where channels and spatial resolution
change, the shortcut is a 1 x 1 convolution with the same stride followed by
BatchNorm so that it can be added to the residual branch.

## 5. Core layers

- **Convolution:** learns local feature detectors from neighboring pixels.
- **BatchNorm:** normalizes activations and learns scale and shift parameters, helping optimization.
- **ReLU:** adds non-linearity: `ReLU(x) = max(0, x)`.
- **Global average pooling:** converts each final feature map to one value, producing 64 features without a large fully connected spatial vector.

## 6. Insights at a glance

**Why change 64/128/256 to 16/32/64?**  The new widths match the paper-inspired CIFAR architecture family and make depth comparisons more controlled.

**Why does depth equal `6n + 2`?**  There are three stages, `n` blocks per stage, two convolutions per block, one initial convolution, and one final fully connected layer.

**Why increase channels as resolution decreases?**  Fewer spatial locations reduce computation, while more channels preserve richer learned representations.

**Why no MaxPool?**  Downsampling is performed by stride-2 convolutions at the start of Stages 2 and 3.

**Is this an exact reproduction?**  No. It is an educational, paper-inspired implementation, not a claim to reproduce every architecture and training detail of the original paper.

## 7. Preliminary baseline record

The original plain CNN baseline was trained for 50 epochs. Its recorded final training accuracy was 92.85% and final test accuracy was 71.28%; its best observed test accuracy was 81.30% at epoch 37. These results are retained as preliminary history and are not silently relabeled as results from `PlainCIFARNet`.

## Understanding Tensor Dimensions

A CNN tensor has the form:

    [batch, channels, height, width]

For example:

    [8, 64, 32, 32]

means:
- 8 images in the batch
- 64 feature channels per image
- each feature map is 32×32

### Changing channels vs changing spatial resolution

A convolution such as:

    Conv2d(64, 128, ...)

changes:

    64 channels → 128 channels

but may keep:

    32×32 → 32×32

if stride=1 and padding is appropriate.

A pooling operation such as:

    MaxPool2d(2)

changes:

    32×32 → 16×16

but keeps:

    64 channels → 64 channels

Therefore:

    Before pooling:
    [batch, 64, 32, 32]

    After pooling:
    [batch, 64, 16, 16]

    After Conv2d(64,128):
    [batch, 128, 16, 16]

The channel dimension and spatial dimensions are separate concepts.

## Global Average Pooling

At the end of the CNN we have:

    [batch, 64, 8, 8]

AdaptiveAvgPool2d((1,1)) operates independently on every image
and every channel.

For each channel, it averages the 8×8 spatial values:

    8×8 = 64 values
         ↓
    average
         ↓
    1 value

Therefore:

    [batch, 64, 8, 8]
             ↓
    [batch, 64, 1, 1]

The batch dimension is NOT mixed together.

For a batch of 8 images:

    8 images
    ×
    64 channels
    ×
    8×8 spatial values

becomes:

    8 images
    ×
    64 values

After flattening:

    [8, 64]

This can then be passed to:

    Linear(64, 10)

## From Feature Maps to Class Predictions

After the three convolutional stages, a batch of 8 images has the shape:

    [8, 64, 8, 8]

This means:

    8  = number of images in the batch
    64 = feature channels for each image
    8 × 8 = spatial size of each feature map

### Global Average Pooling

We apply:

    AdaptiveAvgPool2d((1, 1))

This operates independently on every image and every channel.

For one image:

    Channel 1: 8 × 8 → 1 value
    Channel 2: 8 × 8 → 1 value
    ...
    Channel 64: 8 × 8 → 1 value

For each channel, the 8 × 8 values are averaged:

    64 spatial values
          ↓
       average
          ↓
       1 value

Therefore:

    [8, 64, 8, 8]
            ↓
    [8, 64, 1, 1]

The batch dimension is NOT mixed together. Each image is processed
independently.

### Flattening

We then use:

    torch.flatten(x, 1)

The `1` means:

    Keep dimension 0 (the batch)
    Flatten dimensions 1 and beyond

Therefore:

    [8, 64, 1, 1]
            ↓
    [8, 64]

Now each image is represented by 64 numbers.

Conceptually:

    Image 1 → [64 features]
    Image 2 → [64 features]
    Image 3 → [64 features]
    ...
    Image 8 → [64 features]

So `[8, 64]` means:

    8 separate images
    ×
    64 features describing each image

### Final Classifier

The final layer is:

    Linear(64, 10)

This takes the 64 features from EACH image and produces 10 numbers:

    [8, 64]
       ↓
    Linear(64, 10)
       ↓
    [8, 10]

Therefore each image gets 10 output values, one corresponding to each
CIFAR-10 class.

For example:

    Image 1 → 64 features → 10 class scores
    Image 2 → 64 features → 10 class scores
    ...
    Image 8 → 64 features → 10 class scores

The 10 outputs are logits, not probabilities. CrossEntropyLoss uses these
logits during training.

The complete final transformation is:

    [batch, 64, 8, 8]
             ↓
    Global Average Pooling
             ↓
    [batch, 64, 1, 1]
             ↓
    Flatten
             ↓
    [batch, 64]
             ↓
    Linear(64, 10)
             ↓
    [batch, 10]

    