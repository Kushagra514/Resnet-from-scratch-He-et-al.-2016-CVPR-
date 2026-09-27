# Deep Residual Learning for Image Recognition — From Scratch

An educational implementation and experimental study of the ideas introduced
in the paper:

> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.  
> **Deep Residual Learning for Image Recognition.**  
> CVPR 2016.  
> arXiv:1512.03385

This project implements plain and residual convolutional networks for
CIFAR-10 and investigates the **degradation problem** and the motivation
behind residual learning.

The goal is not to reproduce every experimental detail of the original paper,
but to understand the underlying ideas by implementing them and comparing
plain and residual networks under a controlled training setup.

---

## 1. Motivation

A natural assumption is that adding more layers to a neural network should
make it more powerful.

However, the ResNet paper highlighted a surprising problem: sufficiently
deep **plain networks** can become harder to optimize, resulting in higher
training error than shallower networks.

This is called the **degradation problem**.

The important point is that degradation is not simply the same thing as
overfitting.

A deeper network may have enough capacity to represent a good solution, but
the optimization process can have difficulty finding that solution.

ResNet addresses this by changing how a block represents a transformation.

Instead of directly learning:

\[
H(x)
\]

a residual block learns:

\[
F(x) = H(x) - x
\]

and produces:

\[
\boxed{H(x) = F(x) + x}
\]

The input is passed through a shortcut connection while the convolutional
branch learns the residual transformation.

---

## 2. Project Objectives

This project aims to:

- Understand the degradation problem in deep plain networks.
- Implement a controlled CIFAR-10 plain CNN architecture.
- Investigate how increasing depth affects training behavior.
- Implement residual blocks with identity and projection shortcuts.
- Compare plain and residual networks at matched depths.
- Understand why residual learning can make deeper networks easier to
  optimize.
- Record experimental results rather than assuming the theoretical claims
  will automatically appear under every training setup.

---

## 3. Plain vs Residual Learning

### Plain Network

A plain block directly learns a transformation:

\[
y = H(x)
\]

Conceptually:

```text
x
│
▼
Conv → BN → ReLU → Conv → BN → ReLU
│
▼
y
```

The convolutional layers must learn the complete transformation required

from the input to the output.

### Residual Network

A residual block separates the transformation into two paths:

```
				 ┌────────────── x ──────────────┐
				 │                               │
x → Conv → BN → ReLU → Conv → BN → F(x) ────────┼→ +
												 │
												 ▼
												ReLU
												 │
												 ▼
												y
```

The output is:

y=F(x)+xy = F(x) + x
The shortcut provides the input directly, while the convolutional branch

learns the modification required by the block.

If the desired transformation is close to the identity:

H(x)≈xH(x) \approx x
the residual branch can approach:

F(x)≈0F(x) \approx 0
and the block becomes approximately:

y≈xy \approx x
The important idea is that the convolutional layers still learn their

weights using backpropagation. The difference is the **parameterization of

the block** and the addition of the shortcut.

---

## 4. CIFAR-10 Architecture

The implementation uses a CIFAR-style architecture inspired by the

architecture family used in the original ResNet work.

The network follows:

```
Input
3 × 32 × 32
	│
	▼
Initial 3×3 Conv
16 channels
	│
	▼
Stage 1
16 channels
32 × 32
	│
	▼
Stage 2
32 channels
16 × 16
	│
	▼
Stage 3
64 channels
8 × 8
	│
	▼
Global Average Pooling
	│
	▼
64-dimensional representation
	│
	▼
Linear Layer
10 class logits
```

### Why do channels increase?

The network changes:

```
16 → 32 → 64 channels
```

As spatial resolution decreases, the network can maintain a larger number

of learned feature channels.

The spatial dimensions change:

```
32 × 32 → 16 × 16 → 8 × 8
```

This creates progressively more compact representations while increasing

the number of feature channels.

---

## 5. Tensor Dimensions

CNN tensors use the format:

```
[batch, channels, height, width]
```

For example:

```
[8, 64, 32, 32]
```

means:

- 8 images
- 64 feature channels per image
- each channel has a 32 × 32 spatial grid

Channels and spatial dimensions are different concepts.

For example:

```
[8, 64, 32, 32]
		│
		│ MaxPool2d(2)
		▼
[8, 64, 16, 16]
```

The pooling operation changes the spatial resolution but does not change

the number of channels.

A convolution can instead change the number of channels:

```
[8, 64, 16, 16]
		│
		│ Conv2d(64, 128, ...)
		▼
[8, 128, 16, 16]
```

---

## 6. From Feature Maps to Class Predictions

After the convolutional stages, suppose the tensor is:

```
[8, 64, 8, 8]
```

Each of the 8 images has 64 feature maps, each containing an 8 × 8 grid.

Global average pooling:

```
[8, 64, 8, 8]
		│
		▼
AdaptiveAvgPool2d(1,1)
		│
		▼
[8, 64, 1, 1]
```

Each 8 × 8 feature map is reduced to a single value.

Then:

```
[8, 64, 1, 1]
		│
		▼
Flatten
		│
		▼
[8, 64]
```

The batch dimension is preserved.

Therefore, `[8, 64]` means:

```
8 images
each represented by 64 numbers
```

The final linear layer performs:

```
[8, 64]
	│
	▼
Linear(64, 10)
	│
	▼
[8, 10]
```

Each image receives 10 output values, one for each CIFAR-10 class.

These outputs are **logits**, not probabilities.

`CrossEntropyLoss` operates directly on these logits.

---

## 7. Depth

The CIFAR architecture uses the depth relationship:

depth=6n+2\boxed{depth = 6n + 2}
where nn is the number of blocks in each of the three stages.

Each block contains two convolutional layers.

There is also:

- one initial convolution
- one final linear classification layer

Therefore:

depth=1+3(2n)+1depth = 1 + 3(2n) + 1
which simplifies to:

depth=6n+2depth = 6n + 2
Examples used in this project include:

```
Depth 20 → n = 3
Depth 32 → n = 5
Depth 44 → n = 7
```

Other depths satisfying the formula are also possible.

---

## 8. Plain Blocks

The plain implementation uses blocks of the form:

```
Conv
 ↓
BatchNorm
 ↓
ReLU
 ↓
Conv
 ↓
BatchNorm
 ↓
ReLU
```

There is no shortcut connection.

Mathematically, the block directly represents:

y=H(x)y = H(x)
The classes implemented in `src/models.py` include:

```
PlainBlock
PlainCIFARNet
```

---

## 9. Residual Blocks

The residual implementation replaces the plain block with:

```
				 ┌────────────── shortcut ──────────────┐
				 │                                      │
x → Conv → BN → ReLU → Conv → BN ─────────────────────┼→ +
														│
														▼
													   ReLU
														│
														▼
														y
```

The residual branch produces:

F(x)F(x)
and the shortcut provides either:

xx
or a projected version of xx.

The final operation is:

y=F(x)+shortcut(x)y = F(x) + shortcut(x)

### Identity Shortcut

When the input and output dimensions match:

```
self.shortcut = nn.Identity()
```

Therefore:

shortcut(x)=xshortcut(x)=x
No additional learnable parameters are required.

### Projection Shortcut

When dimensions change, the original input cannot be directly added to

the residual branch.

For example:

```
Input:
[8, 16, 32, 32]

Residual branch:
[8, 32, 16, 16]
```

The tensors cannot be added because their shapes differ.

A 1×1 convolution with the appropriate stride is therefore used:

```
[8, 16, 32, 32]
		│
		▼
1×1 Conv
16 → 32 channels
stride = 2
		│
		▼
[8, 32, 16, 16]
```

The shortcut and residual branch can then be added element-wise.

---

## 10. Training

The training process for every batch is:

```
Input batch
	│
	▼
Forward pass
	│
	▼
Calculate loss
	│
	▼
Backpropagation
	│
	▼
Calculate gradients
	│
	▼
Optimizer updates weights
```

The core training sequence is:

```
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

The test set is used only for evaluation.

During evaluation:

```
model.eval()
```

is used and gradients are disabled.

---

## 11. Experimental Setup

The following configuration is kept constant across the main experiments:

ParameterValueDatasetCIFAR-10Batch size128OptimizerSGDLearning rate0.1Momentum0.9Weight decay5e-4Loss functionCrossEntropyLossTraining epochs50Input normalizationCIFAR-10 mean/stdData augmentationNone currently

The main variable being investigated is **network depth and the presence

of residual connections**.

Keeping the training configuration constant makes comparisons between

architectures more meaningful.

---

## 12. Experiments

### Experiment 1 — Initial Custom CNN

An initial custom CNN was implemented with a smaller number of convolutional

layers.

A deeper version was also tested.

This experiment did not clearly reproduce the degradation behavior.

The architecture was therefore replaced with a more controlled CIFAR-style

6n+26n+2 architecture for the main experiments.

---

### Experiment 2 — Plain Network Depth

The paper-inspired plain network is tested at increasing depths:

```
Plain-20
Plain-32
Plain-44
```

The primary metric for investigating degradation is **training behavior**.

Training loss and training accuracy are more directly relevant to the

degradation question than test accuracy alone.

---

### Plain-20 Result

After 50 epochs:

```
Training loss:       ~0.431
Training accuracy:   ~85.1%
Final test accuracy: ~72.5%
Best test accuracy:  ~77.5%
```

The model successfully optimized the training set.

---

### Plain-32 Result

After 50 epochs:

```
Training loss:       ~0.580
Training accuracy:   ~80.3%
Final test accuracy: ~64.7%
Best test accuracy:  ~75.1%
```

Under the current training setup, the 32-layer plain network achieved

lower training accuracy than the 20-layer network after the same 50-epoch

training budget.

This is consistent with the degradation behavior being investigated.

However, these results alone are not treated as definitive proof because

training behavior can also be affected by optimization settings,

initialization, and other experimental factors.

---

## 13. Planned Residual Comparison

The next experiment is to train residual networks under the same setup:

```
Plain-20   ↔   ResNet-20
Plain-32   ↔   ResNet-32
Plain-44   ↔   ResNet-44
```

The goal is to determine whether residual connections change the training

behavior observed in the deeper plain networks.

The comparison will focus on:

- Training loss
- Training accuracy
- Test loss
- Test accuracy
- Optimization behavior as depth increases

The experiment will not assume that residual networks must outperform

plain networks. The conclusion will be based on the measured results.

---

## 14. Metrics

### Training Loss

Measures how well the network is fitting the training examples according

to the chosen loss function.

### Training Accuracy

Percentage of training examples classified correctly.

This is particularly important for investigating degradation because the

original problem concerns optimization and training error.

### Test Loss

Measures the loss on unseen CIFAR-10 test examples.

### Test Accuracy

Percentage of test examples classified correctly.

Test accuracy measures generalization, but a change in test accuracy by

itself does not establish the degradation problem.

---

## 15. Degradation vs Overfitting

These concepts are related but different.

### Degradation

A deeper plain network has difficulty achieving low training error compared

with a shallower network.

The key evidence is therefore:

```
Deeper model
	  ↓
higher training error
```

under a comparable training setup.

### Overfitting

A model performs substantially better on the training data than on unseen

data.

For example:

```
High training accuracy
		+
Lower test accuracy
```

The experiments record both training and test metrics so that these

phenomena can be distinguished.

---

## 16. Repository Structure

```
resnet-from-scratch/
│
├── README.md
├── model.md
├── train.md
├── math.md
├── ablations.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data.py
│   ├── models.py
│   └── train.py
│
└── experiments/
	└── results/
		├── plain_cifar20.csv
		├── plain_cifar32.csv
		├── plain_cifar44.csv
		├── resnet_cifar20.csv
		├── resnet_cifar32.csv
		└── resnet_cifar44.csv
```

---

## 17. Implementation Details

### `src/data.py`

Responsible for:

- Downloading CIFAR-10
- Applying preprocessing
- Creating training and test datasets
- Creating PyTorch `DataLoader`s

### `src/models.py`

Contains:

- `PlainBlock`
- `PlainCIFARNet`
- `ResidualBlock`
- `ResNetCIFAR`

### `src/train.py`

Contains:

- Training loop
- Evaluation loop
- Loss calculation
- Accuracy calculation
- Optimizer setup
- CSV result logging

### `model.md`

Contains the detailed explanation of:

- Architecture
- Tensor dimensions
- Plain blocks
- Residual blocks
- Skip connections
- Projection shortcuts
- Global average pooling
- Depth calculation

### `train.md`

Contains the detailed explanation of:

- Training procedure
- Experimental controls
- Metrics
- Results
- Interpretation of experiments

### `math.md`

Contains the mathematical background for:

- Forward propagation
- Loss
- Gradients
- Backpropagation
- Gradient descent
- Residual formulation

### `ablations.md`

Contains experimental comparisons and observations.

---

## 18. Limitations

This project is an **educational, paper-inspired implementation** rather

than an exact reproduction of the original ResNet experiments.

Important differences include:

- simplified training procedure
- no data augmentation in the current pipeline
- limited experimental runs
- simplified implementation
- results depend on initialization and training conditions

Therefore, the numerical results should not be directly compared with the

results reported in the original paper.

The purpose of this project is to understand the underlying mechanism of

residual learning and investigate its optimization behavior through a

controlled implementation.

---

## 19. Key Takeaways

The main concepts demonstrated by this project are:

1. Increasing depth does not automatically make optimization easier.
2. The degradation problem concerns training error in deeper plain

networks.
3. A plain block directly learns a transformation H(x)H(x).
4. A residual block learns a residual transformation F(x)F(x).
5. The shortcut provides the input directly to the output addition.
6. When dimensions change, a projection shortcut makes the tensors

compatible.
7. The residual formulation is:

H(x)=F(x)+x\boxed{H(x)=F(x)+x}

1. Residual learning provides an alternative parameterization that can make

deep networks easier to optimize.
2. Experimental conclusions should be based on controlled measurements

rather than assuming that deeper or residual networks must perform

better.

---

## 20. ResNet Depth Experiments

The completed CIFAR-style ResNet-20, ResNet-32, and ResNet-50 results and
figures are documented in [ablations.md](ablations.md). The reproducible plot
script is [experiments/plot_results.py](experiments/plot_results.py), and the
figures are stored under `experiments/results/`.

