## DEGRADATION Vs. VANISHING GRADIENTS

The degradation problem should not be reduced to the vanishing-gradient problem.

Vanishing gradients refer to gradients becoming very small as they propogate backward through a deep network.

The degradation problem described by ResNet is the observation that increasing the depth of a plain network can cause its training error to increase.

The important distinction is:
-> Vanishing gradient: gradient becomes difficult to propogate
-> Degradation: optimization of the deeper plain network becomes harder, and training error increases.

The ResNet paper motivates residual learning as way to make the deeper network easier to optimize.