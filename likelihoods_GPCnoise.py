# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews, Alexis Boukouvalas
# Copyright 2017 Artem Artemev @awav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Haitao Liu (htliu@ntu.edu.sg)


import numpy as np
import tensorflow as tf

from gpflow import logdensities
from gpflow import priors
from gpflow import settings
from gpflow import transforms
from gpflow.decors import params_as_tensors
from gpflow.decors import params_as_tensors_for
from gpflow.params import ParamList
from gpflow.params import Parameter
from gpflow.params import Parameterized
from gpflow.quadrature import hermgauss
from gpflow.quadrature import ndiagquad, ndiag_mc


class Likelihood(Parameterized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                              self.num_gauss_hermite_points,
                              Fmu, Fvar)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            \log \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, logspace=True, Y=Y)

    def variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, Y=Y)

class RobustMax(Parameterized):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.
    """

    def __init__(self, num_classes, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = Parameter(epsilon, transforms.Logistic(), trainable=False, dtype=settings.float_type,
                                 prior=priors.Beta(0.2, 5.))
        self.num_classes = num_classes

    @params_as_tensors
    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, tf.squeeze(1. - self.epsilon), tf.squeeze(self._eps_K1))

    @property
    @params_as_tensors
    def _eps_K1(self):
        return self.epsilon / (self.num_classes - 1.)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = tf.cast(Y, tf.int64)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.float_type) # (lht): one-hot encode
        mu_selected = tf.reduce_sum(oh_on * mu, 1) # (lht): size (N,) oh_on * mu: element-wsie dot of on_on and mu
        var_selected = tf.reduce_sum(oh_on * var, 1) # size (N,)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1)) # N x S, where S is the size of gh_x

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2) # N x P x S
        cdfs = 0.5 * (1.0 + tf.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.float_type)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2) # N x P x S

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, reduction_indices=[1]), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1))) # N x 1

		
class MultiClass_Unifying(Likelihood):
    def __init__(self, num_classes, a=0., **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        invlink = RobustMax(self.num_classes)
        self.invlink = invlink
        self.a = Parameter(a, transforms.positive, trainable=False, dtype=settings.float_type)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        with params_as_tensors_for(self.invlink):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            Fvar = Fvar + self.a
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            ve = p * tf.log(1. - self.invlink.epsilon) + (1. - p) * tf.log(self.invlink._eps_K1)
        return ve

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        Fvar = Fvar + self.a
        possible_outputs = [tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in
                            range(self.num_classes)]
        ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
        ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
        return ps, ps - tf.square(ps)

    @params_as_tensors
    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        with params_as_tensors_for(self.invlink):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            den = p * (1. - self.invlink.epsilon) + (1. - p) * (self.invlink._eps_K1)
        return den


class Binary_Unifying(Likelihood):
    def __init__(self, delta=1e-3, a=0., **kwargs):
        super().__init__(**kwargs)
        self.delta = Parameter(delta, transforms.Logistic(), trainable=False, dtype=settings.float_type, prior=priors.Beta(0.2, 5.))
        self.a = Parameter(a, transforms.positive, trainable=False, dtype=settings.float_type)
    
    @params_as_tensors
    def norm_cdf(self, x):
        return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0)))

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        ve_all = (tf.log(1 - self.delta) - tf.log(self.delta)) * self.norm_cdf(Y * Fmu / tf.sqrt(self.a + Fvar)) + tf.log(self.delta) # N x 1
        ve = tf.reduce_sum(ve_all)

        return ve

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        mu = (1 - 2 * self.delta) * self.norm_cdf(Fmu / tf.sqrt(self.a + Fvar)) + self.delta
        var = mu - tf.square(mu)

        return mu, var


class MultiClass_SoftMax_Aug(Likelihood):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    
    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        Y = tf.cast(Y, tf.int64)
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.float_type) # (lht): one-hot encode
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.float_type)
        Fmu_selected = tf.reduce_sum(oh_on * Fmu, 1) # (lht): oh_on * mu: element-wsie dot of on_on and mu
        Fvar_selected = tf.reduce_sum(oh_on * Fvar, 1)

        P = tf.exp(0.5*Fvar_selected - Fmu_selected) * tf.reduce_sum(tf.exp(0.5*Fvar + Fmu)*oh_off, 1) # (N,)
        ve = - tf.log(1. + P)
        
        return ve

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        # MC solution
        np.random.seed(1)
        N_sample = 1000
        u = np.random.randn(N_sample, self.num_classes) # N_sample x C
        u_3D = tf.tile(tf.expand_dims(u, 1), [1, tf.shape(Fmu)[0], 1]) # N_sample x N* x C 
        Fmu_3D = tf.tile(tf.expand_dims(Fmu, 0), [N_sample, 1, 1]) # N_sample x N* x C 
        Fvar_3D = tf.tile(tf.expand_dims(Fvar, 0), [N_sample, 1, 1]) # N_sample x N* x C 
        exp_term = tf.exp(Fmu_3D + tf.sqrt(Fvar) * u_3D) # mu_3D + tf.sqrt(Fvar) * u_3D are samples from Gaussian distribution
        exp_sum_term = tf.tile(tf.expand_dims(tf.reduce_sum(exp_term, -1), 2), [1, 1, self.num_classes])
        ps = tf.reduce_sum(exp_term / exp_sum_term, 0) / N_sample
        vs = tf.reduce_sum(tf.square(exp_term / exp_sum_term), 0) / N_sample - tf.square(ps)

        return ps, vs

    @params_as_tensors
    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        Y = tf.cast(Y, tf.int64)
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.float_type) # (lht): one-hot encode
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.float_type) 

        Fmu_selected = tf.reduce_sum(oh_on * Fmu, 1) # (lht): size (N,) oh_on * mu: element-wsie dot of on_on and mu
        Fmu_selected_expand = tf.tile(tf.reshape(Fmu_selected, (-1,1)), [1, self.num_classes]) # N* x C
        Fmu_expSum = tf.reduce_sum(tf.exp(Fmu), 1) # N*
        Fmu_expSum_expand = tf.tile(tf.reshape(Fmu_expSum, (-1,1)), [1, self.num_classes]) # N* x C

        D2piD2f_same = tf.exp(Fmu_selected_expand)*(tf.square(Fmu_expSum_expand) - 3*tf.exp(Fmu_selected_expand)*Fmu_expSum_expand + 2*tf.exp(2*Fmu_selected_expand)) / tf.pow(Fmu_expSum_expand, 3)
        D2piD2f_diff = (2*tf.exp(Fmu_selected_expand + 2*Fmu) - tf.exp(Fmu_selected_expand + Fmu)*Fmu_expSum_expand) / tf.pow(Fmu_expSum_expand, 3)        
        D2piD2f = oh_on*D2piD2f_same + oh_off*D2piD2f_diff

        mu = tf.reshape(0.5*tf.reduce_sum(Fvar*D2piD2f, 1), (-1,1)) # N* x 1

        return mu
