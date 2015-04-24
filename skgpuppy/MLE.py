# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
from scipy.optimize import fmin
from scipy.misc import derivative
from scipy.stats import norm
from .Utilities import integrate
from numpy import Inf
from math import fabs, log

class MLE(object):

	def __init__(self, density, theta0, support= None, dims = None, fisher_matrix=None):
		"""
		A class for numerical maximum likelihood estimation

		:param density: lambda x,theta with x and theta being vectors
		:param theta0: the initial parameters of the density
		:param dims: Number of dimensions of x
		:param support: The support of the density
		:param fisher_matrix: Fisher Information Matrix for the density (containts functions of theta)

		.. note:: Either support or dims has to be supplied (support is recommended for estimating the fisher information)
		"""
		assert(dims is not None or support is not None)


		self.theta0 = theta0
		self.fisher_min = None
		if support is not None:
			self.support = support #TODO: Support should be functions of theta
		else:
			self.support = [(-Inf,Inf) for i in range(dims)]

		self.density = density

		self.fisher_matrix = fisher_matrix

	def _get_nll_func(self, observations):
		"""
		negative loglikelihood

		:return: the negative log likelihood function
		"""

		def nll_func(theta):
			for p in theta:
				if p <= 0:
					return 1.0e+20
			sum = 0.0
			for x in observations:
				sum -= np.log(self.density(x, theta))

			return sum

		return nll_func

	def mle(self, observations):
		"""

		:param observations: vector of x vectors
		:return: theta (estimated using maximum likelihood estimation)
		"""
		theta_start = self.theta0
		func = self._get_nll_func(observations)
		theta_min = fmin(func, theta_start)#,xtol=1e-6,ftol=1e-6)

		return theta_min



	def get_fisher_function(self,order=1):
		"""
		Calculate the fisher information matrix

		:param order: using derivates of this order (1 or 2)
		:return: function (w.r.t. theta) calculating the fisher information matrix

		.. note:: If the fisher information matrix was provided to the constructor, then this is used instead of the numerical methods.

		"""

		assert(order == 1 or order == 2)

		def fisher_matrix_function(theta,i,j = None):
			if j is None:
				j = i
			return self.fisher_matrix[i][j](theta)

		def fisher(theta, i, j = None):
			"""
			Fisher information using the first order derivative

			:param theta: the theta of the density
			:param i: The ith component of the diagonal of the fisher information matrix will be returned (if j is None)
			:param j: The i,j th component of the fisher information matrix will be returned
			"""

			#Bring it in a form that we can derive
			fh = lambda ti, t0, tn, x: np.log(self.density(x, list(t0) + [ti] + list(tn)))

			# The derivative
			f_d_theta_i = lambda x: derivative(fh, theta[i], dx=1e-5, n=1, args=(theta[0:i], theta[i + 1:], x))

			if j is not None:
				f_d_theta_j = lambda x: derivative(fh, theta[j], dx=1e-5, n=1, args=(theta[0:j], theta[j + 1:], x))
				f = lambda x: np.float128(0) if fabs(self.density(x, theta)) < 1e-5 else f_d_theta_i(x) * f_d_theta_j(x) * self.density(x, theta)
			else:
				# The function to integrate
				f = lambda x: np.float128(0) if fabs(self.density(x, theta)) < 1e-5 else f_d_theta_i(x) ** 2 * self.density(x, theta)


			#First order
			result = integrate(f, self.support)
			return result

		def fisher_2nd(theta,i, j = None):
			"""
			Fisher information using the second order derivative

			:param theta: the theta of the density
			:param i: The ith component of the diagonal of the fisher information matrix will be returned (if j is None)
			:param j: The i,j th component of the fisher information matrix will be returned
			"""

			# The second order derivate version

			fh = lambda ti, t0, tn, x: np.log(self.density(x, list(t0) + [ti] + list(tn)))
			if j is not None:
				raise NotImplementedError()
			else:
				f_dd_theta_i = lambda x : derivative(fh, theta[i], dx = 1e-5, n=2, args=(theta[0:i],theta[i+1:],x))
				f2 = lambda x: 0 if fabs(self.density(x,theta)) < 1e-5 else f_dd_theta_i(x) * self.density(x,theta)

			result = -integrate(f2,self.support)

			return result

		if self.fisher_matrix is not None:
			return fisher_matrix_function

		if order == 1:
			return fisher
		elif order == 2:
			return fisher_2nd




	def sigma(self, theta, observations=None, n=1):
		"""
		Estimate the quality of the MLE.

		:param theta: The parameters theta of the density
		:param observations: A list of observation vectors
		:param n: Number of observations
		:return: The variances corresponding to the maximum likelihood estimates of theta (quality of the estimation) as 1-d array (i.e. diagonal of the cov matrix)

		.. note:: Either the observations vector or n have to be provided.
		"""
		l2d = []
		if observations is not None:
			n = 1
			func = self._get_nll_func(observations)
			for i in range(len(theta)):
				#Bring it in a form that we can derive
				f = lambda ti, t0, tn: func(list(t0) + [ti] + list(tn))
				l2d.append(derivative(f, theta[i], dx=1e-5, n=2, args=(theta[0:i], theta[i + 1:])))
		else:
			#Fisher Information
			for i in range(len(theta)):
				fisher = self.get_fisher_function()
				result = fisher(theta, i)

				l2d.append(result)

		return 1.0 / np.sqrt(np.array(l2d) * n)

	def mle_ci(self, observations, alpha=0.05):
		"""
		95% CI (if alpha is not given)

		:return: lower bound, upper bound
		"""

		theta = np.array(self.mle(observations))
		sigma = self.sigma(theta, observations)
		return theta - norm.ppf(1-alpha/2) * sigma, theta + norm.ppf(1-alpha/2) * sigma
