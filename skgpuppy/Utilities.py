# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
from numpy.linalg import det, inv
from scipy.integrate import quad, romberg
from math import sqrt, exp, pi
from scipy.optimize import fmin, fmin_tnc, fmin_l_bfgs_b, fmin_cobyla, fmin_slsqp, fmin_bfgs, fmin_cg, fmin_powell, fmin_ncg
from numpy import inf
from scipy.integrate import nquad
from numpy.polynomial.hermite import hermgauss

SQRT2PI = sqrt(2*pi)

class cache_wrapper(object):
	"""
	Wrapper to cache the function values
	"""
	def __init__(self,func):
		self.calls = 0
		self.cache = {} # => von 284 auf 37 function calls runter (even better?)
		self.func = func
	def __call__(self,x):
		tx = tuple(x)
		if tx not in self.cache:
			self.calls += 1
			self.cache[tx] =  self.func(x)
		return self.cache[tx]


def mvnorm(x, mean, K):
	"""

	:param x: input vector
	:param mean: vector of means
	:param K: Covariance Matrix
	:return: density at x
	"""
	n = len(x)
	diff = x - mean
	return 1.0 / (np.sqrt((2 * np.pi) ** n * det(K))) * np.exp(-0.5 * np.dot(diff.T, np.dot(inv(K), diff)))

def norm(x,mean,sigma):
	"""
	Density function of the normal distribution

	:param x:
	:param mean:
	:param sigma:
	:return: density at x
	"""
	return 1/(sigma*SQRT2PI)*np.exp(-0.5*(((x-mean)/sigma)**2))


def _integrate_quad(func, bounds):
	"""

	:param func: The function to integrate (must accept a vector)
	:param bounds: The bounds for every dimension
	:return: Multiple definite integral
	"""

	def integrate_1d(func, bounds):
		"""
		:param func: The function to integrate (must accept arbitrary arguments)
		:param bounds: tuple (lower bound,upper bound)
		:return: function with one argument less (it is integrated over the first argument)
		"""
		return lambda *x: quad(func, bounds[0], bounds[1], args=x, epsabs=0)[0]

	dims = len(bounds)

	f = lambda *x: func(x)

	for i in range(dims - 1):
		f = integrate_1d(f, bounds[i])

	return quad(f, bounds[dims - 1][0], bounds[dims - 1][1], epsabs=0)[0]

def _integrate_romberg(func, bounds):
	"""

	:param func: The function to integrate (must accept a vector)
	:param bounds: The bounds for every dimension
	:return: Multiple definite integral
	"""

	def integrate_1d(func, bounds):
		"""
		:param func: The function to integrate (must accept arbitrary arguments)
		:param bounds: tuple (lower bound,upper bound)
		:return: function with one argument less (it is integrated over the first argument)
		"""
		return lambda *x: romberg(func, bounds[0], bounds[1], args=x)

	dims = len(bounds)

	f = lambda *x: func(x)

	for i in range(dims - 1):
		f = integrate_1d(f, bounds[i])

	return romberg(f, bounds[dims - 1][0], bounds[dims - 1][1])

def _integrate_nquad(func, bounds):
	"""
	Converting the arguments to nquad style

	:param func: function to integrate
	:type func: function with parameter x (iterable of length n)
	:param bounds: bounds for the integration
	:type bounds: iterable of pairs of length n
	:return: value of the integral
	"""
	f = lambda *x: func(x)
	return nquad(f,bounds)[0]

integrate = _integrate_nquad

def integrate_hermgauss(func,mean,sigma,order=1):
	"""
	1-d Gauss-Hermite quadrature

	:param func: lambda x: y (x: float)
	:param mean: mean of normal weight function
	:param sigma: standard dev of normal weight function
	:param order: the order of the integration rule
	:return: :math:`E[f(X)] (X \sim \mathcal{N}(\mu,\sigma^2)) = \int_{-\infty}^{\infty}f(x)p(x),\mathrm{d}x` with p being the normal density
	"""
	x,w = hermgauss(order)
	y = []
	sqrt2 = np.sqrt(2)
	for xi in x:
		y.append(func((xi*sigma*sqrt2+mean,))) #
	y = np.array(y)
	return (y*w).sum() / np.sqrt(np.pi) # * 1/(sigma*np.sqrt(2*np.pi)) * sigma * np.sqrt(2)



def integrate_hermgauss_nd(func,mean,Sigma_x,order):
	"""
	n-d Gauss-Hermite quadrature

	:param func: lambda x: y (x: vector of floats)
	:param mean: mean vector of normal weight function
	:param Sigma_x: covariance matrix of normal weight function
	:param order: the order of the integration rule
	:return: :math:`E[f(X)] (X \sim \mathcal{N}(\mu,\sigma^2)) = \int_{-\infty}^{\infty}f(x)p(x),\mathrm{d}x` with p being the normal density
	"""
	from itertools import product
	dim = len(mean)
	mean = np.array(mean)
	sigma = np.array([np.sqrt(Sigma_x[i][i]) for i in range(dim)])

	x,w = hermgauss(order)
	xs = product(x,repeat=dim)
	ws = np.array(list(product(w,repeat=dim)))
	y = []
	sqrt2 = np.sqrt(2)
	for i,x in enumerate(xs):
		y.append(func(x*sigma*sqrt2+mean)*ws[i].prod())
	y = np.array(y)
	return y.sum() / np.sqrt(np.pi)**dim # * 1/(sigma*np.sqrt(2*np.pi)) * sigma * np.sqrt(2)




def expected_value_monte_carlo(func,mu,Sigma_x,n=1000):
	"""

	:param func: a function that expects an 1D np array
	:param mu: the mean of a multivariate normal
	:param Sigma_x: the cov of a multivariate nromal
	:param n: the number of samples to use
	:return: The expected value of func(x) * p_mvnorm(x|mu,Sigma_x)
	"""

	from numpy.random import multivariate_normal
	vfunc = lambda x: list(map(func,x)) #np.vectorize(func)
	exp_val = np.mean(vfunc(multivariate_normal(mu,Sigma_x,n)))
	return exp_val

#TODO: Switch to scipys own new wrapper function minimize
def minimize(func,theta_start,bounds=None,constr=[],method="all",fprime=None):
	"""

	:param func: function to minimize
	:param theta_start: start parameters
	:param bounds: array of bounds. Each bound is a tuple (min,max)
	:param constr: inequality constraints >= 0 as array of functions
	:param method: all, tnc, l_bfgs_b, cobyla, slsqp, bfgs, powell, cg, simplex or list of some of them
	:param fprime: gradient
	:return: The theta with the minimal function value

	.. note:: constr for cobyla, slsqp, bounds for tnc, l_bfgs_b, slsqp
	"""
	names = []
	thetas = []
	funcvals = []
	times = []
	import time
	approx_grad = (fprime is None)

	if method == "tnc" or method == "all" or (isinstance(method, list) and "tnc" in method):
		start = time.time()
		theta_min = fmin_tnc(func,theta_start,bounds=bounds,approx_grad=approx_grad,fprime=fprime)
		names.append("tnc")
		thetas.append(theta_min[0])
		funcvals.append(func(theta_min[0]))
		end = time.time()
		times.append(end-start)

	if method == "l_bfgs_b" or method == "all" or (isinstance(method, list) and "l_bfgs_b" in method):
		start = time.time()
		theta_min = fmin_l_bfgs_b(func,theta_start,bounds=bounds,approx_grad=approx_grad,fprime=fprime)
		names.append("l_bfgs_b")
		thetas.append(theta_min[0])
		funcvals.append(func(theta_min[0]))
		end = time.time()
		times.append(end-start)

	if method == "cobyla" or method == "all" or (isinstance(method, list) and "cobyla" in method):
		start = time.time()
		theta_min = fmin_cobyla(func,theta_start,constr)
		names.append("cobyla")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end-start)

	if method == "slsqp" or method == "all" or (isinstance(method, list) and "slsqp" in method):
		start = time.time()
		theta_min = fmin_slsqp(func,theta_start,bounds=bounds,fprime=fprime,ieqcons =constr)
		names.append("slsqp")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end-start)

	if method == "bfgs" or method == "all" or (isinstance(method, list) and "bfgs" in method):
		start = time.time()
		theta_min = fmin_bfgs(func,theta_start,fprime=fprime)
		names.append("bfgs")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end-start)

	if method == "powell" or method == "all" or (isinstance(method, list) and "powell" in method):
		start = time.time()
		theta_min = fmin_powell(func,theta_start)
		names.append("powell")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end-start)

	if method == "cg" or method == "all" or (isinstance(method, list) and "cg" in method):
		start = time.time()
		theta_min = fmin_cg(func,theta_start,fprime=fprime)
		names.append("cg")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end-start)

	if method == "ncg" or method == "all" or (isinstance(method, list) and "ncg" in method):
		start = time.time()
		theta_min = fmin_ncg(func,theta_start,fprime)
		names.append("ncg")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end-start)

	if method == "simplex" or method == "all" or (isinstance(method, list) and "simplex" in method):
		start = time.time()
		theta_min = fmin(func, theta_start,maxiter=len(theta_start)*10000, maxfun=len(theta_start)*10000,ftol=1e-10,xtol=1e-10)
		names.append("simplex")
		thetas.append(theta_min)
		funcvals.append(func(theta_min))
		end = time.time()
		times.append(end - start)

	func_val = None
	theta_min = None
	for i, name in enumerate(names):
		if func_val is None or (funcvals[i] != -inf and funcvals[i] < func_val):
			func_val = funcvals[i]
			theta_min = thetas[i]

		print(name, "\t", times[i], "\t", funcvals[i], "\t", thetas[i])

	print(theta_min)

	return theta_min





