Scalable Gaussian process classification with additive noise for various likelihoods
====

This is the python implementation of scalable Gaussian process classification (GPC) with **additive noise** for various likelihoods.

As a statistical model, GPC provides a flexible and powerful framework describing joint distributions over function space. Conventional GPCs however suffer from two prominent weaknesses: 

1. the poor scalability for big data due to the full kernel matrix; and 
2. the intractable inference due to the non-Gaussian likelihoods. 

To address the two issues, various scalable GPCs have been proposed through 

1. the sparse approximation which employs a small inducing set to distill the entire training data in order to reduce the time complexity; and 
2. the approximate inference to derive analytical evidence lower bound (ELBO).  

However, these scalable GPCs equipped with analytical ELBO are limited to *specific likelihoods* or *additional assumptions*.

In this work, we present **a unifying framework which accommodates scalable GPCs with various likelihoods**. Analogous to GP regression (GPR), we introduce additive noises to augment the probability space for (i) the GPCs with step and (multinomial) probit/logit likelihoods via the internal variables; and *particularly*, (ii) the GPC using softmax likelihood via the noise variables themselves, resulting in **scalable** models with **analytical ELBO** by using variational inference.

The model is implemented based on [GPflow 1.3.0](https://github.com/GPflow/GPflow) and tested using Tensorflow 1.13.0. 

The illustration examples are provided in

```
demo_binary.ipynb
```
and
```
demo_multiclass.ipynb
```
