# Mathematics and Concepts

## 1. THE RESEARCH PROBLEM:
Increasing network depth should theoretically provide a neural network with greater representational power.

However, simply adding layers to a plain network does not necessarily make optimization easier.

The ResNet paper identifies a degradation problem:
as a plain network becomes deeper, its training error can actually increase.

This is different from ordinary overfitting because the problem is observed in the training error itself.

### Theroetical identity argument
suppose a shalow network has learned a desired mapping: H(x)

If additional layers could simply implement an identity mapping: G(x) = x, in that case the deeper network could actually behave exactly like the shallower network. Therefore, theoretically , increasing depth should not make the best achievable training solution worse.

The practical problem is that ordinary optimization can have difficulty finding this identity mapping in a plain stack of layers. This motivated a different formulation of the learning problem.

## DEGRADATION Vs. VANISHING GRADIENTS

The degradation problem should not be reduced to the vanishing-gradient problem.

Vanishing gradients refer to gradients becoming very small as they propogate backward through a deep network.

The degradation problem described by ResNet is the observation that increasing the depth of a plain network can cause its training error to increase.

The important distinction is:
-> Vanishing gradient: gradient becomes difficult to propogate
-> Degradation: optimization of the deeper plain network becomes harder, and training error increases.

The ResNet paper motivates residual learning as way to make the deeper network easier to optimize.

## 3. IDENTITY MAPPING 
An identity mapping is simply:
    G(x) = x
It leaves its input unchanged.
    For eg-> x    = [2, 5, 9]
             G(x) = [2, 5, 9]

In the deeper-network argument, the newly added layers would ideally learn an identity mapping so that the deeper network could reproduce the shallower network's behaviour.

## MOTIVATION FOR RESIDUAL LEARNING 

A deeper plain network should theoretically be able to represent the shallower solution, but learning that solution through ordinary layers can be difficult.
ResNet changes the formulation so that the added layers learn a residual mapping rather than directly learning the complete desired mapping 


## 4. RESIDUAL LEARNING 

A plain network attempts to directly learn a desired mapping:
H(x)

Instead, residual learning reformulates the problem so that the stacked layers learn the residual
F(x) = H(x) - x
Here: 
-H(x) is the desired underlying mapping
-F(x) is the residual mapping
-x is the original input 
-F(x) + x is the output of the residual block.

### Identity mapping

If the desired mapping is simply:
H(x) = x

then:
F(x) = H(x) -  x
     = x - x
     = 0
    Therefore, an identity mapping corresponds to learning a zero residual.
    This is important for the degradation problem because a deeper network should theoretically be able to preserve the mapping learned by a shallower network 

### Numerical example

Suppose:
    x = 5
    H(x) = 8

    Then:

    F(x) = H(x) - x
         = 8 - 5
         = 3

    The residual block produces: F(x) + x = 3 + 5 = 8, which is the desired mapping.
    The residual thus represents the change that must be applied to the input.

### Key Intution
Plain network :  Learn the complete tranformation H(x) 
Residual network : Learn the required modification F(x) and add the original input x 

## 5. RESIDUAL BLOCK ARCHITECTURE

The residual function F(x) is implemented using learnable layers, such as convolution and batch normalization.

A basic residual block can be represented as:

x
│
├──────────────────────────────┐
│                              │
▼                              │
Conv → BN → ReLU → Conv → BN   │
│                              │
└────────────────────────────► (+)
                                │
                               ReLU
                                │
                                ▼
                                y

The mathematical form is:

y = ReLU(F(x) + x)
The convolutional layers contain learnable parameters.
The identity shortcut contains no learnable parameters whenn the input and output dimensions already match.

When dimensions change, a projection shortcut  can be used:
y = F(x) + W_s * x
where W_s is a learned projection that makes the dimensions compatible.

## Why Residual Learning Helps Optimization

Plain and residual formulations can represent the same
functions, but solve different optimization problems.

When a block should behave like the identity (H(x) = x):
  Plain:     must learn weights that reproduce x → hard,
             because identity is a specific configuration.
  Residual:  must learn F(x) = 0 → easy, because weights
             are initialized near zero and only need to be
             shrunk.

In deep networks, most blocks should make only small changes,
so H(x) ≈ x for most blocks. The residual formulation makes
this common case cheap to learn.

This is why deeper residual networks do not degrade: the
worst-case behavior of a residual block (doing nothing) is
easy to achieve, so adding depth cannot increase training error.