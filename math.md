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

