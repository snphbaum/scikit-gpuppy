# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np


class GaussianProcess(object):
	"""
	A Gaussian Process implementation based on Girard's work.
	(Girard, A. Approximate Methods for Propagation of Uncertainty with Gaussian Process Models University of Glasgow, 2004)
	The main work is done in the Covariance class

	"""

	def __init__(self, x, t, cov, theta_min=None):
		"""

		:param x: inputs shape (n,d)
		:param t: noisy responses
		:param cov: covariance function
		:param theta_min: use this vector of hyperparameters, otherwise use maximum likelihood estimation ot estimate it

		"""

		self.x = x
		self.n, self.d = np.shape(x)
		self.meant= np.mean(t)
		self.t = t - self.meant


		self.cov = cov
		if theta_min is not None:
			self.theta_min = theta_min
		else:
			self.theta_min = self.cov.ml_estimate(self.x,self.t)

		self.Kinv = self.cov.inv_cov_matrix(self.x,self.theta_min)


	@staticmethod
	def get_realisation(x, cov, theta):
		"""
		Generates a realisation of a gaussian process with the given parameters for the covariance function.

		:param x: shape (n,d) => points where to get the realisation
		:param cov: the covariance function
		:param theta: parameter vector for cov
		:return: Realisation of the gaussian process
		"""
		n,d = np.shape(x)
		K = cov.cov_matrix(x,theta)
		mean = np.zeros(n)
		return np.random.multivariate_normal(mean, K)

	def __call__(self, x_star):
		"""
		Returns an estimate of the mean and the code variance at a given point

		:param x_star: shape (d)
		:return: mean, variance of the GP at x_star
		"""
		return self.estimate(x_star)

	def estimate_many(self, x_stars):

		#TODO Optimize for the SPGP covariance function
		vt = self._get_vt()
		Kinv = self.Kinv

		x_star = np.array(x_stars)
		k = self.cov.cov_matrix(x_star, self.theta_min)
		kv = self.cov.cov_matrix_ij(x_star,self.x,self.theta_min) #np.array([self.covariance(x_star, self.x[i], v, w) for i in range(len(self.x))])
		mean = np.dot(kv, np.dot(Kinv, self.t))
		variance = k - np.dot(kv, np.dot(Kinv, kv.T))#+ vt #code variance + aleatory variance
		#print variance, variance-vt
		return mean+self.meant, np.diag(variance)

		# means = []
		# variances = []
		#
		# for x_star in x_stars:
		# 	mean, variance = self.estimate(x_star)
		# 	means.append(mean)
		# 	variances.append(variance)
		#
		# return np.array(means), np.array(variances)



	def estimate(self, x_star):
		"""
		Returns an estimate of the mean and the code variance at a given point

		:param x_star: input vector of shape (d)
		:return: Mean and Variance at x_star
		"""

		#TODO Optimize for the SPGP covariance function
		vt = self._get_vt()
		Kinv = self.Kinv

		x_star = np.array(x_star)
		k = self.cov(x_star, x_star, self.theta_min) #+ vt  #code variance + aleatory variance
		kv = self.cov.cov_matrix_ij(np.atleast_2d(x_star),self.x,self.theta_min) #np.array([self.covariance(x_star, self.x[i], v, w) for i in range(len(self.x))])
		mean = np.dot(kv, np.dot(Kinv, self.t))
		variance = k - np.dot(kv, np.dot(Kinv, kv.T))
		return mean[0]+self.meant, variance[0,0]


	def _get_beta(self):
		"""

		:return: :math:`\beta= K^{-1}t`
		"""
		return np.dot(self.Kinv, self.t)

	def _get_W_inv(self):
		w = np.exp(self.theta_min[2:self.d+2])
		Winv = np.diag(w)
		return Winv

	def _get_v(self):
		return np.exp(self.theta_min[0])

	def _get_vt(self):
		return np.exp(self.theta_min[1])


	def _covariance(self, xi, xj, v=None, w=None):
		"""
		:param xi: N-dimensional vector
		:param xj: N-dimensional vector
		:return: covariance
		"""


		theta = self.theta_min

		if v is not None:
			theta[0] = np.log(v)

		if w is not None:
			theta[2:] = np.log(w)

		return self.cov(xi,xj,theta)


	def _inv_cov_matrix(self):
		"""

		:return: inverse of the covariance matrix
		"""

		# if self.Kinv is None:
		# 	theta = self.theta_min
		# 	vt = self.get_vt()
		# 	theta[1] = np.log(vt)
		# 	self.Kinv = self.cov.inv_cov_matrix(self.x,theta)

		return self.Kinv

	def _get_mean_t(self):
		"""

		:return: the mean of the original data
		"""
		return self.meant

	def _get_Hessian(self,u,xi):
		"""
		Wrapper for the get_Hessian method of the covariance

		:param u: Mean
		:param xi: Point
		:return: Hessian of the covariance between u and xi (derivative with respect to u)
		"""
		return self.cov.get_Hessian(u,xi,self.theta_min)

	def _get_Jacobian(self,u,xi):
		"""
		Wrapper for the get_Jacobion method of the covariance

		:param u: Mean
		:param xi: Point
		:return: Jacobian of the covariance between u and xi (derivative with respect to u)
		"""
		return self.cov.get_Jacobian(u,xi,self.theta_min)
