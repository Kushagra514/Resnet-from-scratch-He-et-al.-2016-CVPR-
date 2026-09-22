## PREREQUISITES

### Channels

A channel is one feature plane.

A CIFAR-10 RGB image has 3 channels:

- Red
- Green
- Blue

A convolution can produce many learned feature channels.
These are feature maps produced by different learned filters.

For example:

3 input channels -> 64 learned feature maps

does not mean the image has 64 colors. It means the layer
learns 64 different feature detectors.

### Spatial Resolution

Spatial resolution refers to the height and width of a
feature map.

For example:

32 x 32 -> 16 x 16 -> 8 x 8

means the spatial representation is being downsampled.

### Why Increase Channels?

As the spatial resolution becomes smaller, the network can
use more feature channels to represent different kinds of
learned patterns.

The architecture therefore roughly follows:

spatial resolution ↓
feature channels ↑

This is a common CNN design pattern.

### Why Reduce Spatial Resolution?

Downsampling:

- reduces computation,
- reduces the number of spatial locations,
- allows later layers to operate on larger effective regions
  of the input.

The network trades some exact spatial detail for a more compact
and feature-rich representation.

### Our Plain CNN

Input:

3 x 32 x 32

Then approximately:

3 x 32 x 32
-> 64 x 32 x 32
-> 64 x 16 x 16
-> 128 x 16 x 16
-> 128 x 8 x 8
-> 256 x 8 x 8
-> 256 x 1 x 1
-> 256
-> 10 class logits




## PLAIN CNN BASELINE 

### Purpose

The plain CNN is the control model for the ResNet experiment.
It does not contain skip connections or residual blocks, its transformations are applied sequentially:
x -> H_1(x) -> H_2(x) -> H_3(x) -> ...

Each H_i is implemented using learnable layers such as convolution, batch normalization, and a non linear activation.

### What the network learns?

The network is trained only using the final classification loss.
Backpropogation computes gradients of the loss with respect to the learnable parameters, and an optimizer updates those parameters.
The intermediate activations are not seperately supervised.

### Convolution

A convolution applies learned filters to local regions of the input.
The convolution weights determine which local patterns the network corresponds to.

### Multiple Convolution Layers

Stacking convolutional layers allows later layers to operate on features produced by earlier layers, and thus learn more complex representation.
The network therefore builds incresingly useful feature representations as a consequence of learning its parameters.

### ReLU

ReLU is:

ReLU(x) = max(0,x)

It introduces non linearity into the network.

### Batch Normalization

Batch Normalization helps keep activations numerically well-behaved during training and contains learnable scale and shift parameters.

### Max Pooling 

Max pooling reduces spatial resolution.
For eg : 32 x 32 -> 16 x 16 
This reduces computation and gives later layers a larger effective receptive field.

### Increasing Channels

The network increases the number of feature channels while reducing spatial resolution:
3 x 32 x 32 
->  64 x 32 x 32
->  64 x 16 x 16
->  128 x 16 x 16
->  128 x 8 x 8
->  256 x 8 x 8 
This allows the network to represent more feature type while using a more compact spatial resolution 


### Final Classification 

Adaptive average pooling converts:

256 x 8 x 8 -> 256 x 1 x 1 

After flattening:
256 x 1 x 1 -> 256

A linear layer maps the 256 features into 10 class logits:
256 -> 10

### Plain vs Residual Formulation

Plain network:

x_(l+1) = H_l(x_l)

Residual network:

x_(l+1) = x_l + F_l(x_l)

The main experiment changes this formulation while keeping the
classification task and dataset fixed.

## Plain CNN Baseline

The initial plain CNN baseline was trained for 50 epochs on CIFAR-10.

Final training accuracy: 92.85%
Final test accuracy: 71.28%

Best observed test accuracy: 81.30% at epoch 37.

The model continued improving on the training set while
test performance became unstable and generally stopped
improving consistently, indicating overfitting.

This baseline establishes that the CIFAR-10 training pipeline
and plain CNN implementation are functioning before introducing
deeper architectures and residual connections.