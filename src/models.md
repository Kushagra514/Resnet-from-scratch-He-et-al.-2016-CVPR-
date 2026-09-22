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