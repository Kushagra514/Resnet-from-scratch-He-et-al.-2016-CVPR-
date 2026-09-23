# train.py — Training Pipeline Documentation

## Overview

This file trains a CNN on CIFAR-10 and records the results.
It implements the standard supervised training loop:

    forward pass
    → loss
    → backpropagation
    → gradient computation
    → optimizer update
    → repeat for next batch

repeated for a fixed number of epochs, with evaluation
after each epoch.

The training code is **model-agnostic**. It works for any
`nn.Module` — PlainCNN today, ResNet later — as long as the
model takes `[N, 3, 32, 32]` images and outputs `[N, 10]`
logits.

---

## The training loop in one picture

    ┌─────────────────────┐
    │   CIFAR-10 images   │
    └──────────┬──────────┘
               ↓
          Forward pass
               ↓
          Predictions
               ↓
      Compare with true labels
               ↓
             Loss
               ↓
        Backpropagation
               ↓
       Gradients of weights
               ↓
        Update the weights
               ↓
          Next batch

Repeated for 50 epochs. Each epoch also runs an evaluation
pass on the test set.

---

## Project structure

    data.py     → provides train_loader, test_loader
    models.py   → defines PlainCNN
    train.py    → trains the model

Separation of concerns:
- data.py knows nothing about models or training.
- models.py knows nothing about data or training.
- train.py orchestrates.

---

## Imports

    import csv, os
    import torch, torch.nn as nn, torch.optim as optim
    from data import get_cifar10_loaders
    from models import PlainCNN

- `csv`  — writes per-epoch results to a CSV file.
- `os`   — creates the results directory.
- `torch` — tensors, device management (CPU/GPU).
- `torch.nn` — layers and loss functions (`Conv2d`, `BatchNorm2d`,
  `Linear`, `CrossEntropyLoss`, etc.).
- `torch.optim` — optimizers (`SGD`).
- `data.py` — gives us the loaders.
- `models.py` — gives us the network.

---

## `train_one_epoch(model, loader, criterion, optimizer, device)`

Trains the model for one full pass over the training set.

### Setup

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

- `model.train()` — puts the model in training mode.
  This matters for BatchNorm (uses batch statistics and
  updates running stats) and Dropout (active).
- `running_loss` — accumulates total loss across batches.
- `correct` — accumulates number of correct predictions.
- `total` — accumulates number of examples seen.

### Batch loop

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)                  # [128, 10]
        loss = criterion(outputs, labels)        # scalar

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)      # [128]
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

### Why each line

- **`images.to(device)` / `labels.to(device)`**
  Model and data must be on the same device (GPU or CPU).

- **`optimizer.zero_grad()`**
  PyTorch accumulates gradients by default. Without
  zeroing, batch N's gradients would include batch N-1's.
  Must clear before each backward pass.

- **`outputs = model(images)`**
  Forward pass. Produces `[batch, 10]` logits — 10 scores
  per image, one per class.

- **`loss = criterion(outputs, labels)`**
  CrossEntropyLoss. Averages over the batch, so `loss` is
  the mean per-image loss for this batch.

- **`loss.backward()`**
  Backpropagation. Computes ∂Loss/∂w for every learnable
  parameter. Gradients stored in `param.grad`.

- **`optimizer.step()`**
  Applies the update: `w ← w − lr · ∂Loss/∂w`
  (with momentum and weight decay included for SGD).

- **`running_loss += loss.item() * images.size(0)`**
  `loss` is a batch average, so multiply back by the batch
  size to recover the sum. This makes the accumulation a
  true total across all images — correct even when the
  final batch is smaller.

- **`predictions = outputs.argmax(dim=1)`**
  `dim=1` collapses the class axis. `[128, 10] → [128]`.
  One predicted class per image.

- **`correct += (predictions == labels).sum().item()`**
  Elementwise integer comparison. Counts matches.

- **`total += labels.size(0)`**
  Counts examples seen this epoch.

### Return

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy

One average loss and one accuracy for the whole epoch,
built from batch-level accumulation.

---

## `evaluate(model, loader, criterion, device)`

Same structure as `train_one_epoch`, but:

    @torch.no_grad()
    def evaluate(...):
        model.eval()
        ...

Differences from training:

| Aspect                   | train_one_epoch | evaluate |
|--------------------------|-----------------|----------|
| Purpose                  | Improve model   | Measure  |
| `model.train()`/`eval()` | train()         | eval()   |
| `@torch.no_grad()`       | no              | yes      |
| `zero_grad()`            | yes             | no       |
| `loss.backward()`        | yes             | no       |
| `optimizer.step()`       | yes             | no       |
| Weights                  | change          | frozen   |
| Data                     | training set    | test set |

- `@torch.no_grad()` — disables graph construction. Saves
  memory and time; gradients aren't needed for measurement.

- `model.eval()` — BatchNorm uses learned running statistics
  instead of current-batch statistics, and does not update
  them. Dropout is disabled. Evaluating in train mode would
  produce noisy, incorrect results and corrupt the model.

The test set is **never** used to update weights. It is the
only honest measure of generalization.

---

## `main()`

### Device selection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Data

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

- Training set: 50,000 images.
- Test set: 10,000 images.
- Batch size: 128.
- Training batches per epoch: 50,000 / 128 ≈ 391.

### Model

    model = PlainCNN().to(device)

### Loss

    criterion = nn.CrossEntropyLoss()

Standard for multi-class single-label classification.

### Optimizer

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

- `lr=0.1` — learning rate; step size for updates.
- `momentum=0.9` — retains 90% of previous velocity.
  v ← 0.9·v + g; w ← w − lr·v. Reduces oscillation
  and speeds convergence.
- `weight_decay=5e-4` — L2 regularization. Penalizes
  large weights; helps generalization.

### Epoch loop

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(...)
        test_loss,  test_acc  = evaluate(...)
        writer.writerow([epoch, train_loss, train_acc,
                         test_loss, test_acc])
        print(...)

Each epoch:
1. Train over 50,000 images (weights change).
2. Evaluate over 10,000 images (weights frozen).
3. Log results.

### Output

    experiments/results/plain_baseline.csv

Columns:

    epoch, train_loss, train_accuracy, test_loss, test_accuracy

### Entry point

    if __name__ == "__main__":
        main()

Runs `main()` only when executed as a script, not when
imported.

---

## Key concepts

### Batch

A subset of the training set processed together. Batch
size 128 means 128 images per forward/backward/update.
Batching is forced by GPU memory and enables parallelism.

### Epoch

One full pass over the training set. 50 epochs means the
model sees the 50,000 training images 50 times.

### Update frequency

Weights are updated **once per batch**, not once per image
and not once per epoch.

- Full-batch gradient descent: 1 update per epoch (stable,
  slow).
- Mini-batch SGD (this code): ~391 updates per epoch
  (good balance).
- Pure SGD: 50,000 updates per epoch (noisy, slow on GPU).

### Gradient averaging

`CrossEntropyLoss` averages over the batch, so
`loss.backward()` computes the batch-averaged gradient.
That is why one `step()` per batch is correct.

### Training vs evaluation

    TRAIN:  forward → loss → backward → step   (weights change)
    EVAL:   forward → loss → accuracy          (weights frozen)

The test set answers: "how well does the model generalize
to data it has never seen?" Training accuracy cannot
answer this, because the model is being fit to that data.

### Overfitting

    Train Acc: 92.85%
    Test  Acc: 71.28%
    Gap:      21.57%   ← overfitting

A large train/test gap signals the model has memorized
the training set rather than learned general features.
Regularization (weight decay, dropout, augmentation,
early stopping) reduces this gap.

---

## Training pipeline

    data
    → forward pass
    → loss
    → backpropagation
    → gradients
    → optimizer update
    → next batch
    → end of epoch
    → evaluate on test set
    → log results
    → next epoch

---

## How this extends to ResNet

The training code does **not** change when swapping
PlainCNN for ResNet. The optimizer, loss, loops, and
evaluation remain identical.

Only the model's forward mapping changes:

Plain block:

    x → Conv → BN → ReLU → Conv → BN → y

Residual block:

             ┌──────────── x ────────────┐
             │                           │
    x → Conv → BN → ReLU → Conv → BN     │
             │                           │
             └────────── + ──────────────┘
                         ↓
                       ReLU
                         ↓
                         y

Key realization: **ResNet changes the architecture, not
the training algorithm.** The same loss → backpropagation
→ optimizer process trains it.

---

## Summary

> For each batch: the model makes predictions,
> CrossEntropyLoss measures the error, backpropagation
> computes gradients with respect to every learnable
> parameter, and SGD updates those parameters. After each
> epoch, the unchanged model is evaluated on the test set
> to measure generalization.