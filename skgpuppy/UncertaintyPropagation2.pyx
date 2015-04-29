# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
cimport numpy as np
from skgpuppy.Covariance import tracedot
from numpy.linalg import inv, det

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class UncertaintyPropagationGA:
	"""
	Superclass for all UncertaintyPropagationGA Classes
	"""

	def __init__(self, object gp):
		"""

		:param gp: callable gaussian process that returns mean and variance for a given input vector x
		"""
		self.gp = gp

	cpdef double propagate_mean(self,np.ndarray[DTYPE_t,ndim=1] u,np.ndarray[DTYPE_t,ndim=2] Sigma_x):
		"""
		Propagates the mean using the gaussian approximation from Girard2004

		:param u: vector of means
		:param Sigma_x: covariance Matrix of the input
		:return: mean of the output
		"""
		return 0

	cpdef tuple propagate_GA(self,np.ndarray[DTYPE_t,ndim=1] u,np.ndarray[DTYPE_t,ndim=2] Sigma_x):
		"""
		Propagates the uncertainty using the gaussian approximation from Girard2004

		:param u: vector of means
		:param Sigma_x: covariance Matrix of the input
		:return: mean, variance of the output
		"""
		return 0, 0




cdef class UncertaintyPropagationExact(UncertaintyPropagationGA):
	def __init__(self, object gp):
		"""

		:param gp: callable gaussian process that returns mean and variance for a given input vector x
		"""
		self.gp = gp

	cdef _prepare_C_corr(self,int D):
		"""
		Prepares the constants for C_corr

		:param D: Number of dimensions of the input vectors
		"""
		#D = len(xi)
		cdef np.ndarray[DTYPE_t,ndim=2] I = np.eye(D)
		self.Deltainv = self.Winv - np.diag(
			np.array([self.Winv[i][i] / (1 + self.Winv[i][i] * self.Sigma_x[i][i]) for i in range(D)]))
		self.normalize_C_corr = 1 / np.sqrt(det(I + self.Winv * self.Sigma_x))

	cdef double _get_C_corr(self,np.ndarray[DTYPE_t,ndim=1] u,np.ndarray[DTYPE_t,ndim=1] xi):
		"""
		:param u: N-dimensional vector
		:param xi: N-dimensional vector
		:return: covariance correction factor
		"""
		diff = u - xi
		cdef double normalize_C_corr = self.self.normalize_C_corr
		cdef np.ndarray[DTYPE_t,ndim=2] Deltainv = self.Deltainv
		return  normalize_C_corr * np.exp(0.5 * (np.dot(diff.T, np.dot(Deltainv, diff))))


	cpdef double propagate_mean(self, np.ndarray[DTYPE_t,ndim=1] u,np.ndarray[DTYPE_t,ndim=2] Sigma_x):
		cdef np.ndarray[DTYPE_t,ndim=1] x = self.gp.x
		cdef int N
		cdef int d
		cdef list C_ux_t
		N = x.shape[0]
		d = x.shape[1]
		cdef np.ndarray[DTYPE_t,ndim=1] C_ux = self.C_ux
		if C_ux is None:
			C_ux_t = []
			for i in range(N):
				C_ux_t.append(self.gp._covariance(u,x[i]))
			C_ux = np.array(C_ux_t)

		cdef np.ndarray[DTYPE_t,ndim=1] beta = self.gp._get_beta()
		self.Winv = self.gp._get_W_inv()
		self.Sigma_x = Sigma_x

		assert(N == len(beta))
		self._prepare_C_corr(len(u))

		cdef double sum = 0.0
		for i in range(N):
			sum += beta[i] * C_ux[i] * self._get_C_corr(u, x[i])

		return sum

	cdef _prepare_C_corr2(self, D):
		"""
		Prepares the constants for C_corr2

		:param D: Number of dimensions of the input vectors
		"""

		#D = len(x)
		I = np.eye(D)
		W = np.diag(np.array([1 / self.Winv[i][i] for i in range(D)]))
		self.LambdaInv = 2 * self.Winv - inv(0.5 * W + self.Sigma_x)
		self.normalize_C_corr2 = 1 / np.sqrt(det(2 * self.Winv * self.Sigma_x + I))


	cdef _get_C_corr2(self, u, x):
		"""
		This function is alternatively calculated by weave inline C code

		:param u: N-dimensional vector
		:param xi: N-dimensional vector
		:return: covariance correction factor 2
		"""

		#D = len(x)
		#I = np.eye(D)
		#W = np.diag(np.array([1/self.Winv[i][i] for i in range(D)]))
		#LambdaInv = 2* self.Winv - inv(0.5*W+self.Sigma_x)
		diff = u - x

		return self.normalize_C_corr2 * np.exp(0.5 * (np.dot(diff.T, np.dot(self.LambdaInv, diff))))

	cpdef tuple propagate_GA(self,np.ndarray[DTYPE_t,ndim=1] u,np.ndarray[DTYPE_t,ndim=2] Sigma_x):
		beta = self.gp._get_beta()
		x = self.gp.x
		self.Winv = self.gp._get_W_inv()
		self.Sigma_x = Sigma_x
		Kinv = self.gp._inv_cov_matrix()
		#Kinv = inv(self.gp.cov_matrix())
		N,d = x.shape
		assert(N == len(beta))


		self._prepare_C_corr2(len(u))

		C_ux = []
		for i in range(N):
			C_ux.append(self.gp._covariance(u,x[i]))
		C_ux = np.array(C_ux)
		self.C_ux = C_ux
		mu = self.propagate_mean(u, Sigma_x)



			# Old Version
		sum = 0.0
		for i in range(N):
			for j in range(N):
				x_ = (x[i] + x[j]) / 2.0

				l_ij = C_ux[i] * \
					   C_ux[j] * \
					   self._get_C_corr2(u, x_)

				sum += (Kinv[i][j] - beta[i] * beta[j]) * l_ij

		variance = self.gp._covariance(u, u) - sum - mu ** 2

		return mu + self.gp._get_mean_t(), variance




cdef class UncertaintyPropagationApprox(UncertaintyPropagationGA):
	def __init__(self, gp):
		"""

		:param gp: callable gaussian process that returns mean and variance for a given input vector x
		"""
		UncertaintyPropagationGA.__init__(self, gp)
		self.v = self.gp._get_v()
		self.Winv = self.gp._get_W_inv()
		self.u = None

	cpdef double propagate_mean(self,np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=2] Sigma_x):
		cdef np.ndarray[DTYPE_t,ndim=1] beta = self.gp._get_beta()
		cdef np.ndarray[DTYPE_t,ndim=1] x = self.gp.x
		cdef int n = len(x)
		cdef np.ndarray[DTYPE_t,ndim=1] mu = sum([beta[i]*self.C_ux[i] for i in range(n)])
		#print "Approx mean"
		#print mu
		#print 0.5 * sum([beta[i] * np.trace(np.dot(self.get_Hessian(u,x[i]),Sigma_x)) for i in range(n)])

		#Time complexity d**3 because of dot

		return mu + 0.5 * sum([beta[i] * np.trace(np.dot(self.H_ux[i],Sigma_x)) for i in range(n)])

	#@profile

	cdef double _get_sigma2(self,np.ndarray[DTYPE_t,ndim=1] u,Kinv,x,C_ux,J_ux,H_ux):
		n = len(x)

		sum_ = sum([Kinv[i][j]*C_ux[i]*C_ux[j] for i in range(n) for j in range(n)])

		sigma2 = self.gp._covariance(u,u) - sum_#


		return sigma2

	cdef double _get_variance_rest(self,u,Sigma_x,Kinv,x,beta,C_ux,J_ux,H_ux):
		n,d = x.shape


		S = np.atleast_2d(np.diag(Sigma_x)).T


		variance2 =- sum([(Kinv[i][j]-beta[i]*beta[j]) * (J_ux[i]*J_ux[j]*S).sum() for i in range(n) for j in range(n)])

		trace = np.array([tracedot(H_ux[i],Sigma_x) for i in range(n)])

		variance3 = - 0.5* sum([Kinv[i][j]*(C_ux[i]*trace[j]+C_ux[j]*trace[i]) for i in range(n) for j in range(n)])

		return variance2+variance3

	cdef tuple _get_sigma2_and_variance_rest(self,u,Sigma_x,Kinv,x,beta):
		n = len(x)
		sigma2 = self._get_sigma2(u,Kinv,x,self.C_ux,self.J_ux,self.H_ux)
		variance_rest = self._get_variance_rest(u,Sigma_x,Kinv,x,beta,self.C_ux,self.J_ux,self.H_ux)

		return sigma2, variance_rest

	cpdef tuple propagate_GA(self,np.ndarray[DTYPE_t,ndim=1] u,np.ndarray[DTYPE_t,ndim=2] Sigma_x):
		beta = self.gp._get_beta()
		x = self.gp.x
		n = len(x)
		Kinv = self.gp._inv_cov_matrix()
		#Kinv = inv(self.gp.cov_matrix())

		#We store the values because we can reuse them
		if self.u is None or (self.u != u).any():

			self.u = u
			self.C_ux = []
			self.J_ux = []
			self.H_ux = []
			for i in range(n):
				self.C_ux.append(self.gp._covariance(u,x[i]))
				self.J_ux.append(self.gp._get_Jacobian(u,x[i]))
				self.H_ux.append(self.gp._get_Hessian(u,x[i]))
			self.C_ux = np.array(self.C_ux)
			self.J_ux = np.array(self.J_ux)
			self.H_ux = np.array(self.H_ux)

		mean = self.propagate_mean(u,Sigma_x)
		sigma2, variance_rest = self._get_sigma2_and_variance_rest(u,Sigma_x,Kinv,x,beta)



		#TODO!!!: Only if C''(u,u) == 0
		variance = sigma2 + variance_rest




		return mean + self.gp._get_mean_t(), variance


	cpdef double _getFactor(self,u,Sigma_x,v):
		"""

		:param u: the mean
		:param Sigma_x: The weight_vector based cov matrix
		:param v: the limiting output variance
		:return: the lambda to multiply Sigma_x with
		"""
		#TODO!!!: consider predefined variances and divide into block matrices

		beta = self.gp._get_beta()
		x = self.gp.x
		n = len(x)

		Kinv = self.gp._inv_cov_matrix() #inv(self.gp.cov_matrix())
		#We store the values because we can reuse them
		if self.u is None or (self.u != u).any():

			self.u = u
			self.C_ux = []
			self.J_ux = []
			self.H_ux = []
			for i in range(n):
				self.C_ux.append(self.gp._covariance(u,x[i]))
				self.J_ux.append(self.gp._get_Jacobian(u,x[i]))
				self.H_ux.append(self.gp._get_Hessian(u,x[i]))
			self.C_ux = np.array(self.C_ux)
			self.J_ux = np.array(self.J_ux)
			self.H_ux = np.array(self.H_ux)

		sigma2, variance_rest = self._get_sigma2_and_variance_rest(u,Sigma_x,Kinv,x,beta)



		return (v-sigma2)/(variance_rest)



	cpdef double _get_variance_dv_h(self,u,h):
		beta = self.gp._get_beta()
		Kinv = self.gp._inv_cov_matrix()
		x = self.gp.x
		n,d = x.shape



		#We store the values because we can reuse them
		if self.u is None or (self.u != u).any():

			self.u = u
			self.C_ux = []
			self.J_ux = []
			self.H_ux = []
			for i in range(n):
				self.C_ux.append(self.gp._covariance(u,x[i]))
				self.J_ux.append(self.gp._get_Jacobian(u,x[i]))
				self.H_ux.append(self.gp._get_Hessian(u,x[i]))
			self.C_ux = np.array(self.C_ux)
			self.J_ux = np.array(self.J_ux)
			self.H_ux = np.array(self.H_ux)




		variance2 =	- sum([(Kinv[i][j]-beta[i]*beta[j]) * self.J_ux[i,h,0]*self.J_ux[j,h,0] for i in range(n) for j in range(n)])



		C_ux = self.C_ux
		H_ux = self.H_ux

		variance3 = - 0.5* sum([Kinv[i][j]*(C_ux[i]*H_ux[j,h,h]
										+C_ux[j]*H_ux[i,h,h]) for i in range(n) for j in range(n)])


		return variance2+variance3
