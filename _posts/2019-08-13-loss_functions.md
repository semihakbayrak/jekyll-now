---
layout: post
title: Wanna See Where the Loss Functions of Deep Learning Come From?
---

One common way of interpreting Deep Neural Netowrks (DNNs) is that the first layers work as feature extractors and the middle 
layers map extracted features to a new low dimensional manifold such that a Generalized Linear Model can make the discremenation.
Thus for a classification task, we want to map the instances to a manifold that the classes are linearly seperated. From a probabilistic
point of view, minimization of negative log-likelihood leads to the loss functions that are being used in Deep Learning. In order to
see this connection more detailly, I invite you to take a look at my medium article on [Generalized Linear Models](https://towardsdatascience.com/generalized-linear-models-8738ae0fb97d).
