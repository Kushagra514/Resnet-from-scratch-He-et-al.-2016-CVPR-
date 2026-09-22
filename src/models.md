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

### 
