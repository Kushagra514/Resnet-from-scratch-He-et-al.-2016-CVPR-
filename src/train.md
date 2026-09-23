# Training and Experiment

## 1. Research question

The central experiment asks whether increasing depth in a plain network creates optimization or degradation difficulties, and later whether residual connections change this behavior.

We are not claiming that deeper networks are always worse. The target phenomenon is degradation: a deeper plain network can show higher training error even though it has greater representational capacity. This is distinct from ordinary overfitting and is investigated through training behavior, not test accuracy alone.

## 2. Current experiment

```text
Dataset: CIFAR-10
Model: PlainCIFARNet(depth=20)
Loss: CrossEntropyLoss
Optimizer: SGD
Batch size: 128
Momentum: 0.9
Learning rate: 0.1
Weight decay: 5e-4
Epochs: 50
Results: experiments/results/plain_cifar20.csv
```

The model accepts `[N, 3, 32, 32]` images and produces `[N, 10]` class logits. The existing `plain_baseline.csv` and `deep_plain.csv` files are preserved as preliminary experiment history.

## 3. Controlled variables

When comparing depths, and later comparing Plain vs ResNet, keep these as constant as practical:

```text
dataset and preprocessing
batch size
optimizer and loss
training duration
evaluation procedure
```

The architectural variables of interest are depth and, later, the presence of residual connections.

## 4. Training loop

```text
batch
  -> forward pass
  -> loss
  -> backward pass
  -> optimizer.step()
```

For each training batch, the code moves images and labels to the selected device, clears accumulated gradients, computes logits, evaluates cross-entropy loss, backpropagates, and updates the weights.

`model.train()` enables training behavior, including BatchNorm updates. `model.eval()` switches to evaluation behavior. The evaluation function uses `torch.no_grad()` because gradients are not needed for measurement.

## 5. Metrics

- **Training loss:** the objective minimized during training.
- **Training accuracy:** the percentage of training examples classified correctly.
- **Test loss:** cross-entropy loss on held-out examples.
- **Test accuracy:** classification performance on held-out examples.

Test accuracy alone cannot establish degradation. Compare training loss and training accuracy across depths, alongside test metrics, to determine whether a deeper plain model is harder to optimize.

## 6. Why the first experiment was insufficient

The preliminary custom 6-layer vs 13-layer plain CNN experiment did not clearly reproduce degradation. That result is not being artificially forced or discarded. The project instead moves to the paper-inspired `6n + 2` CIFAR architecture, uses valid controlled depths, and establishes a stronger plain-network baseline before adding residual connections.

## 7. Current next step

Train:

```text
PlainCIFARNet(depth=20)
```

Save results to:

```text
experiments/results/plain_cifar20.csv
```

The next comparison will use a deeper plain model and then a residual counterpart with the same general CIFAR structure.

## 8. Training history note

The old training code and CSV outputs remain useful for understanding the preliminary custom architecture. They should not be compared as though they came from the new `PlainCIFARNet` setup without accounting for the architecture change.
