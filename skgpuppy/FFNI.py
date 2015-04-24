# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


from .Utilities import integrate, mvnorm
import numpy as np
from .Utilities import integrate_hermgauss_nd, cache_wrapper

class PropagateMoments(object):
	"""
	Superclass for uncertainty propagation through deterministic functions
	"""

	def __init__(self,func,mean):
		"""

		:param func: (n-d) function to approximate
		:param mean: propagate uncertainty around this mean vector
		"""
		self.mean = mean
		self.func = cache_wrapper(func)

	def _exn(self,n,Sigma_x):
		"""
		Generates the n-th moment (not centralized!) of the output distribution

		:param n: order of the moment
		:param Sigma_x: Covariance matrix
		:return: That moment
		"""
		pass

	def propagate(self,Sigma_x, skew=False, kurtosis=False):
		"""
		Propagates a normal distributed uncertainty around self.mean through the deterministic function self.func.

		:param Sigma_x: Covariance matrix (assumed to be diagonal)
		:param skew: Return the skewness of the resulting distribution
		:param kurtosis: Return the kurtosis of the resulting distribution
		:return: mean, variance, [skewness, [kurtosis]]
		"""

		mean = self._exn(1,Sigma_x)
		ex2 = self._exn(2,Sigma_x)
		print("Function calls: ", self.func.calls)
		# import matplotlib.pyplot as plt
		# xs = np.array(self.func.cache.keys())
		# plt.scatter(xs.T[0],xs.T[1])
		# plt.title('Output PDF')
		# plt.show()
		variance = ex2 - mean**2

		if not skew:
			return mean, variance
		else:
			ex3 = self._exn(3,Sigma_x)
			skewness = (ex3 - 3 * mean * variance - mean**3)/np.sqrt(variance)**3

			if not kurtosis:
				return mean, variance, skewness
			else:
				ex4 = self._exn(4,Sigma_x)
				kurtosis = (ex4 - 4 * mean * ex3 + 6*mean**2*ex2 - 3 * mean**4)/(variance**2)
				return mean, variance, skewness, kurtosis



class FullFactorialNumericalIntegrationHermGauss(PropagateMoments):
	"""
	Class to perform error propagation using Gauss-Hermite quadrature
	"""
	def __init__(self,func,mean,order):
		"""

		:param func: (n-d) function to approximate
		:param mean: propagate uncertainty around this mean vector
		:param order: order of the integration series
		"""
		PropagateMoments.__init__(self,func,mean)
		self.order = order




	def _exn(self,n,Sigma_x):
		h = lambda x: self.func(x)**n
		return integrate_hermgauss_nd(h,self.mean,Sigma_x,self.order)



class FullFactorialNumericalIntegrationEvans(PropagateMoments):
	"""
	Class to perform error propagation using Evans Method (1967).

	.. warning::
		This method is very inaccurate (especially for skewness and kurtosis)

	"""


	def _exn(self,n,Sigma_x):
		d = len(self.mean)
		a = np.sqrt(3.0)
		C = 1+d*(d-7)/18.0
		H = -(d-4)/18.0
		P = 1/36.0

		#C + H*2*n+P*2*n*(n-1) = 1

		sigma = np.array([np.sqrt(Sigma_x[i][i]) for i in range(d)])
		h = lambda x: self.func(x)**n
		Q = C*h(self.mean)

		for i in range(d):
			m = np.copy(self.mean)
			m[i] += a * sigma[i]
			Q += H*h(m)

			m = np.copy(self.mean)
			m[i] -= a * sigma[i]
			Q += H*h(m)

		c = 0
		for i in range(d):
			for j in range(i+1,d):
				if i != j:
					c+=4
					m = np.copy(self.mean)
					m[i] += a * sigma[i]
					m[j] += a * sigma[j]
					Q += P*h(m)

					m = np.copy(self.mean)
					m[i] += a * sigma[i]
					m[j] -= a * sigma[j]
					Q += P*h(m)

					m = np.copy(self.mean)
					m[i] -= a * sigma[i]
					m[j] -= a * sigma[j]
					Q += P*h(m)

					m = np.copy(self.mean)
					m[i] -= a * sigma[i]
					m[j] += a * sigma[j]
					Q += P*h(m)

		#print c, 2*n*(n-1)
		return Q



class FullFactorialNumericalIntegrationNaive(PropagateMoments):
	"""
	Class to perform error propagation using Scipy numerical integration
	"""


	def _exn(self,n,Sigma_x):
		bounds = []
		for i,m in enumerate(self.mean):
			bounds.append((self.mean[i]-4*np.sqrt(Sigma_x[i][i]),self.mean[i]+4*np.sqrt(Sigma_x[i][i])))

		fn = lambda x: self.func(x)**n * mvnorm(x,self.mean,Sigma_x)
		return integrate(fn,bounds)

# Download fwtpts from http://www.math.wsu.edu/faculty/genz/software/software.html and uncomment
# class FullFactorialNumericalIntegrationGenzKeister(PropagateMoments):
# 	"""
# 	Class to perform error propagation using Genz-Keister quadrature
# 	"""
#
# 	def __init__(self,func,mean,order):
# 		PropagateMoments.__init__(self,func,mean)
# 		self.order = order
#
#
# 	def _exn(self,n,Sigma_x):
# 		from oct2py import octave
# 		dim = len(self.mean)
# 		#http://www.math.wsu.edu/faculty/genz/software/software.html
# 		w,p,nump = octave.fwtpts(dim,self.order)
#
# 		mean = np.array(self.mean)
# 		sigma = np.array([np.sqrt(Sigma_x[i][i]) for i in range(dim)])
#
# 		h = lambda x: self.func(x)**n
# 		p = np.array(p).T
# 		w = w[0]
# 		y = []
# 		nump = int(nump)
# 		for i in range(nump):
# 			#print p[i]*sigma+mean
# 			y.append(h(p[i]*sigma+mean))
#
# 		y = np.array(y)
# 		return (y*w).sum() #/ sigma.prod()


