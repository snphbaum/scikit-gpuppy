# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
from skgpuppy.Covariance import tracedot
cython = False
try:
	from scipy import weave
	weaving = True
except ImportError:
	weaving = False
if not weaving:
	try:
		from skgpuppy.UncertaintyPropagation2 import UncertaintyPropagationExact, UncertaintyPropagationApprox
		cython = True
	except ImportError:
		cython = False

from .Utilities import mvnorm, integrate, expected_value_monte_carlo, integrate_hermgauss_nd
from numpy.linalg import inv, det
#from KernelDensityEstimator import KDE, KDEUniform


class UncertaintyPropagation(object):
	def propagate(self, y, u, Sigma_x):
		"""
		Propagates the uncertain density Girard2004 (page 32)

		:param y: point to estimate the output density at
		:param u: vector of means
		:param Sigma_x: covariance Matrix of the input
		:return: output density
		"""

	def propagate_many(self, yvec, u, Sigma_x):
		"""
		Propagates the uncertain density Girard2004 (page 32)

		:param yvec: vector of points to estimate the output density at
		:param u: vector of means
		:param Sigma_x: covariance Matrix of the input
		:return: output densities
		"""
		ret = []
		for y in yvec:
			ret.append(self.propagate(y, u, Sigma_x))

		return np.array(ret)


class UncertaintyPropagationGA(object):
	"""
	Superclass for all UncertaintyPropagationGA Classes
	"""

	def __init__(self, gp):
		"""

		:param gp: callable gaussian process that returns mean and variance for a given input vector x
		"""
		self.gp = gp

	def propagate_mean(self, u, Sigma_x):
		"""
		Propagates the mean using the gaussian approximation from Girard2004

		:param u: vector of means
		:param Sigma_x: covariance Matrix of the input
		:return: mean of the output
		"""
		return 0

	def propagate_GA(self, u, Sigma_x):
		"""
		Propagates the uncertainty using the gaussian approximation from Girard2004

		:param u: vector of means
		:param Sigma_x: covariance Matrix of the input
		:return: mean, variance of the output
		"""
		return 0, 0




class UncertaintyPropagationMC(UncertaintyPropagationGA, UncertaintyPropagation):
	"""
	Using Monte Carlo Integration -> Very inefficient but very stable
	"""
	def __init__(self, gp, n=1000):
		"""

		:param gp: callable gaussian process that returns mean and variance for a given input vector x
		:param n: number of samples
		"""
		UncertaintyPropagationGA.__init__(self,gp)
		self.n = n

	def propagate_mean(self, u, Sigma_x):

		func = lambda x: (self.gp(x))[0] #* mvnorm(x, u, Sigma_x)

		return expected_value_monte_carlo(func, u, Sigma_x,self.n)


	def propagate_GA(self, u, Sigma_x):
		mu = self.propagate_mean(u, Sigma_x)


		func1 = lambda x: (self.gp(x))[1] #* mvnorm(x, u, Sigma_x)
		func2 = lambda x: (self.gp(x))[0] ** 2 #* mvnorm(x, u, Sigma_x)
		#variance = integrate(func1, bounds) + integrate(func2, bounds) - mu ** 2
		variance = expected_value_monte_carlo(func1, u, Sigma_x,self.n) \
				   + expected_value_monte_carlo(func2, u, Sigma_x,self.n) - mu**2



		return mu, variance


	def propagate(self, y, u, Sigma_x):
		def py(x, y):
			mux, varx = self.gp(x)
			return 1 / (np.sqrt(2 * np.pi * varx)) * np.exp(-0.5 * (y - mux) ** 2 / varx)


		func = lambda x: py(x, y) #* mvnorm(x, u, Sigma_x)
		return expected_value_monte_carlo(func, u, Sigma_x,self.n)


class UncertaintyPropagationNumericalHG(UncertaintyPropagationGA, UncertaintyPropagation):
	"""
	The numerical propagation works fine if we want predictions for the noisy f
	"""


	def propagate_mean(self, u, Sigma_x):

		func = lambda x: (self.gp(x))[0]
		return integrate_hermgauss_nd(func,u,Sigma_x,4)


	def propagate_GA(self, u, Sigma_x):
		mu = self.propagate_mean(u, Sigma_x)
		func1 = lambda x: (self.gp(x))[1]
		func2 = lambda x: (self.gp(x))[0] ** 2
		variance = integrate_hermgauss_nd(func1,u,Sigma_x,4) + integrate_hermgauss_nd(func2,u,Sigma_x,4) - mu ** 2
		return mu, variance


	def propagate(self, y, u, Sigma_x):
		def py(x, y):
			mux, varx = self.gp(x)
			return 1 / (np.sqrt(2 * np.pi * varx)) * np.exp(-0.5 * (y - mux) ** 2 / varx)


		func = lambda x: py(x, y)
		return integrate_hermgauss_nd(func,u,Sigma_x,4)

class UncertaintyPropagationNumerical(UncertaintyPropagationGA, UncertaintyPropagation):
	"""
	The numerical propagation works fine if we want predictions for the noisy f
	But it is unstable for small variances

	.. deprecated:: Use UncertaintyPropagationNumericalHG instead

	"""
	#TODO should be replaced permanently by above HG implementation

	def propagate_mean(self, u, Sigma_x):

		bounds = []
		for i in range(len(u)):
			bounds.append((u[i] - 6 * Sigma_x[i][i], u[i] + 6 * Sigma_x[i][i]))

		func = lambda x: (self.gp(x))[0] * mvnorm(x, u, Sigma_x)

		return integrate(func, bounds)


	def propagate_GA(self, u, Sigma_x):
		mu = self.propagate_mean(u, Sigma_x)

		bounds = []
		for i in range(len(u)):
			bounds.append((u[i] - 6 * Sigma_x[i][i], u[i] + 6 * Sigma_x[i][i]))

		func1 = lambda x: (self.gp(x))[1] * mvnorm(x, u, Sigma_x)
		func2 = lambda x: (self.gp(x))[0] ** 2 * mvnorm(x, u, Sigma_x)
		variance = integrate(func1, bounds) + integrate(func2, bounds) - mu ** 2


		return mu, variance


	def propagate(self, y, u, Sigma_x):
		def py(x, y):
			mux, varx = self.gp(x)
			return 1 / (np.sqrt(2 * np.pi * varx)) * np.exp(-0.5 * (y - mux) ** 2 / varx)

		bounds = []
		for i in range(len(u)):
			bounds.append((u[i] - 6 * Sigma_x[i][i], u[i] + 6 * Sigma_x[i][i]))

		func = lambda x: py(x, y) * mvnorm(x, u, Sigma_x)
		return integrate(func, bounds)


class UncertaintyPropagationLinear(UncertaintyPropagationGA):
	def _derivative(self, x, i, d=1e-5):
		"""
		Calculates the derivate of the gaussian process at a given position in a given dimension.

		:param x: vector
		:param i: dimension in which to derive
		:return:
		"""
		xnew1 = list(x)
		xnew2 = list(x)

		xnew1[i] -= d
		xnew2[i] += d
		return (self.gp(xnew2)[0] - self.gp(xnew1)[0]) / (2.0 * d)

	def propagate_mean(self, u, Sigma_x):
		mu = self.gp(u)[0]
		return mu

	def propagate_GA(self, u, Sigma_x):
		mu = self.gp(u)[0]
		steigungen = [self._derivative(u, i, d=1e-5) for i in range(len(u))]
		variance = 0.0
		for i in range(len(u)):
			variance += steigungen[i] ** 2 * Sigma_x[i][i]

		variance += self.gp._get_vt()

		return mu, variance

if not cython:

	class UncertaintyPropagationExact(UncertaintyPropagationGA):
		def _prepare_C_corr(self, D):
			"""
			Prepares the constants for C_corr

			:param D: Number of dimensions of the input vectors
			"""
			#D = len(xi)
			I = np.eye(D)
			self.Deltainv = self.Winv - np.diag(
				np.array([self.Winv[i][i] / (1 + self.Winv[i][i] * self.Sigma_x[i][i]) for i in range(D)]))
			self.normalize_C_corr = 1 / np.sqrt(det(I + self.Winv * self.Sigma_x))

		def _get_C_corr(self, u, xi):
			"""
			:param u: N-dimensional vector
			:param xi: N-dimensional vector
			:return: covariance correction factor
			"""
			diff = u - xi
			return  self.normalize_C_corr * np.exp(0.5 * (np.dot(diff.T, np.dot(self.Deltainv, diff))))


		def propagate_mean(self, u, Sigma_x,C_ux=None):
			x = self.gp.x
			N,d = x.shape

			if C_ux is None:
				C_ux = []
				for i in range(N):
					C_ux.append(self.gp._covariance(u,x[i]))
				C_ux = np.array(C_ux)

			beta = self.gp._get_beta()
			self.Winv = self.gp._get_W_inv()
			self.Sigma_x = Sigma_x

			assert(N == len(beta))
			self._prepare_C_corr(len(u))

			sum = 0.0
			for i in range(N):
				sum += beta[i] * C_ux[i] * self._get_C_corr(u, x[i])

			return sum

		def _prepare_C_corr2(self, D):
			"""
			Prepares the constants for C_corr2

			:param D: Number of dimensions of the input vectors
			"""

			#D = len(x)
			I = np.eye(D)
			W = np.diag(np.array([1 / self.Winv[i][i] for i in range(D)]))
			self.LambdaInv = 2 * self.Winv - inv(0.5 * W + self.Sigma_x)
			self.normalize_C_corr2 = 1 / np.sqrt(det(2 * self.Winv * self.Sigma_x + I))


		def _get_C_corr2(self, u, x):
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

		def propagate_GA(self, u, Sigma_x):
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
			mu = self.propagate_mean(u, Sigma_x,C_ux)


			if weaving:

				code = """
				double sum = 0;
				for(int i = 0;i < N;i++){
					for(int j = 0;j < N;j++){
						double dot = 0;
						for(int i2 = 0;i2<d;i2++){
							for(int j2 = 0;j2<d;j2++){
								dot += (U1(i2)-(X2(i,i2)+X2(j,i2))/2.0) * (U1(j2)-(X2(i,j2)+X2(j,j2))/2.0) *L2(i2,j2);
							}
						}
						sum += (KINV2(i,j)-BETA1(i)*BETA1(j)) * C_UX1(i)*C_UX1(j) * nc * exp(0.5 * dot);
					}
				}
				return_val = sum;
				"""

				nc = float(self.normalize_C_corr2)
				L = self.LambdaInv
				sum = weave.inline(code,['Kinv','C_ux','N','nc','x','beta','d','L','u'],headers = ["<math.h>"])
			else:
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






	class UncertaintyPropagationApprox(UncertaintyPropagationGA):
		def __init__(self, gp):
			"""

			:param gp: callable gaussian process that returns mean and variance for a given input vector x
			"""
			UncertaintyPropagationGA.__init__(self, gp)
			self.v = self.gp._get_v()
			self.Winv = self.gp._get_W_inv()
			self.u = None

		def propagate_mean(self, u, Sigma_x):
			beta = self.gp._get_beta()
			x = self.gp.x
			n = len(x)
			mu = sum([beta[i]*self.C_ux[i] for i in range(n)])
			#print "Approx mean"
			#print mu
			#print 0.5 * sum([beta[i] * np.trace(np.dot(self.get_Hessian(u,x[i]),Sigma_x)) for i in range(n)])

			#Time complexity d**3 because of dot

			return mu + 0.5 * sum([beta[i] * np.trace(np.dot(self.H_ux[i],Sigma_x)) for i in range(n)])

		#@profile

		def _get_sigma2(self,u,Kinv,x,C_ux,J_ux,H_ux):
			n = len(x)


			if weaving:
				code = """
				double sum = 0;
				for(int i = 0; i< n;i++){
					for(int j = 0; j< n;j++){
						sum += KINV2(i,j)*C_UX1(i)*C_UX1(j);
					}
				}
				return_val = sum;
				"""
				sum_ = weave.inline(code,['Kinv','C_ux','n'])
			else:
				sum_ = sum([Kinv[i][j]*C_ux[i]*C_ux[j] for i in range(n) for j in range(n)])

			sigma2 = self.gp._covariance(u,u) - sum_#


			return sigma2

		def _get_variance_rest(self,u,Sigma_x,Kinv,x,beta,C_ux,J_ux,H_ux):
			n,d = x.shape


			S = np.atleast_2d(np.diag(Sigma_x)).T

			if weaving:
				J = J_ux
				code = """
				double sum = 0;
				for(int i = 0; i< n;i++){
					for(int j = 0; j< n;j++){
						double trace = 0;
						for (int k = 0;k < d;k++)
							trace += J3(i,k,0)*J3(j,k,0)*S2(k,0);
						sum += (KINV2(i,j)-BETA1(i)*BETA1(j))*trace;
					}
				}
				return_val = -1*sum;
				"""
				variance2 = weave.inline(code,['Kinv','beta','J','S','n','d'])
			else:
			#Old Versions
			#variance2_1 =	- sum([(Kinv[i][j]-beta[i]*beta[j]) * np.trace(np.dot(np.dot(J_ux[i],J_ux[j].T), Sigma_x)) for i in range(n) for j in range(n)])
			#variance2_2 =	- sum([(Kinv[i][j]-beta[i]*beta[j]) * tracedot(np.dot(J_ux[i],J_ux[j].T), Sigma_x) for i in range(n) for j in range(n)])
				variance2 =	- sum([(Kinv[i][j]-beta[i]*beta[j]) * (J_ux[i]*J_ux[j]*S).sum() for i in range(n) for j in range(n)])



			trace = np.array([tracedot(H_ux[i],Sigma_x) for i in range(n)])

			if weaving:
				code = """
				double sum = 0;
				for(int i = 0; i< n;i++){
					for(int j = 0; j< n;j++){
						sum += KINV2(i,j)*(C_UX1(i)*TRACE1(j) + C_UX1(j)*TRACE1(i));
					}
				}
				return_val = -0.5*sum;
				"""
				variance3 = weave.inline(code,['Kinv','C_ux','trace','n'])
			else:
				variance3 = - 0.5* sum([Kinv[i][j]*(C_ux[i]*trace[j]
												+C_ux[j]*trace[i]) for i in range(n) for j in range(n)])

			return variance2+variance3

		def _get_sigma2_and_variance_rest(self,u,Sigma_x,Kinv,x,beta):
			n = len(x)
			sigma2 = self._get_sigma2(u,Kinv,x,self.C_ux,self.J_ux,self.H_ux)
			variance_rest = self._get_variance_rest(u,Sigma_x,Kinv,x,beta,self.C_ux,self.J_ux,self.H_ux)

			return sigma2, variance_rest

		def propagate_GA(self, u, Sigma_x):
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


		def _getFactor(self,u,Sigma_x,v):
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



		def _get_variance_dv_h(self,u,h):
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




			if weaving:
				J = self.J_ux

				code = """
				double sum = 0;
				for(int i = 0; i< n;i++){
					for(int j = 0; j< n;j++){
						double trace = 0;
						trace = J3(i,h,0)*J3(j,h,0);
						sum += (KINV2(i,j)-BETA1(i)*BETA1(j))*trace;
					}
				}
				return_val = -1*sum;
				"""

				variance2 = weave.inline(code,['Kinv','beta','J','n','h'])
			else:
				variance2 =	- sum([(Kinv[i][j]-beta[i]*beta[j]) * self.J_ux[i,h,0]*self.J_ux[j,h,0] for i in range(n) for j in range(n)])



			C_ux = self.C_ux
			H_ux = self.H_ux
			if weaving:
				code = """
				double sum = 0;
				for(int i = 0; i< n;i++){
					for(int j = 0; j< n;j++){
						sum += KINV2(i,j)*(C_UX1(i)*H_UX3(j,h,h) + C_UX1(j)*H_UX3(i,h,h));
					}
				}
				return_val = -0.5*sum;
				"""

				variance3 = weave.inline(code,['Kinv','C_ux','n','H_ux','h'])
			else:
				variance3 = - 0.5* sum([Kinv[i][j]*(C_ux[i]*H_ux[j,h,h]
											+C_ux[j]*H_ux[i,h,h]) for i in range(n) for j in range(n)])


			return variance2+variance3