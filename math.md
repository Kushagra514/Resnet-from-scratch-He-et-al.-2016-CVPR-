# Mathematics and Concepts

## 1. THE RESEARCH PROBLEM:
Increasing network depth should theoretically provide a neural network with greater representational power.

However, simply adding layers to a plain network does not necessarily make optimization easier.

The ResNet paper identifies a degradation problem:
as a plain network becomes deeper, its training error can actually increase.

This is different from ordinary overfitting because the problem is observed in the training error itself.

### Theroetical identity argument
suppose a shalow network has learned a desired mapping: H(x)


## DEGRADATION Vs. VANISHING GRADIENTS

The degradation problem should not be reduced to the vanishing-gradient problem.

Vanishing gradients refer to gradients becoming very small as they propogate backward through a deep network.

The degradation problem described by ResNet is the observation that increasing the depth of a plain network can cause its training error to increase.

The important distinction is:
-> Vanishing gradient: gradient becomes difficult to propogate
-> Degradation: optimization of the deeper plain network becomes harder, and training error increases.

The ResNet paper motivates residual learning as way to make the deeper network easier to optimize.