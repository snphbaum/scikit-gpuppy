# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


from itertools import combinations
import numpy as np
from scipy.misc import derivative
from skgpuppy.FFNI import PropagateMoments


#from hashlib import sha1

#tuple seems the fastes way to create hashables from small numpy arrays

hashable = tuple
#hashable = lambda thing: sha1(thing).hexdigest()

# class hashable:
# 	def __init__(self,thing):
# 		self._hash = hash(sha1(thing).hexdigest())
# 		self._thing = thing
#
# 	def __hash__(self):
# 		return self._hash
# 	def __eq__(self, other):
# 		return np.all(self._thing == other._thing)


def _setpartition(iterable, n=2):
	"""
	Gets the pairs for Isserli's theorem

	:param iterable: Iterable
	:param n: number of elements in each set
	:return:
	"""

	iterable = list(iterable)
	partitions = combinations(combinations(iterable, r=n), r=len(iterable) // n)
	for partition in partitions:
		seen = set()
		for group in partition:
			if seen.intersection(group):
				break
			seen.update(group)
		else:
			yield partition


def _fast_isserli(powerlist,Sigma_x):
	"""
	Get higher order mixed centralized moments of a multivariate gaussian using isserlis theorem
	http://en.wikipedia.org/wiki/Isserlis%27_theorem

	This is optimized by assuming the x_i to be independent

	:math:`E[(x-\mu_x)^{k_x} (y-\mu_y)^{k_y}] = E[(x-\mu_x)^{k_x}] \cdot E [(y-\mu_y)^{k_y}]` => no need for isserli

	:param powerlist: list of powers of the random variables of the multivariate normal
	:param Sigma_x: The covariance matrix
	:return:
	"""
	from scipy.misc import factorial2

	if powerlist.sum() % 2 != 0:
		#Odd order
		return 0

	for power in powerlist:
		if power % 2 != 0:
			return 0

	part = 1.0
	for i,power in enumerate(powerlist):
		part *= Sigma_x[i][i]**(power/2)
		part *= factorial2(power-1, exact=True)

	return part



def _Isserli(powerlist, Sigma_x, diagonal=True):
	"""
	Get higher order mixed centralized moments of a multivariate gaussian using isserlis theorem
	http://en.wikipedia.org/wiki/Isserlis%27_theorem

	:param powerlist: list of powers of the random variables of the multivariate normal
	:param Sigma_x: The covariance matrix
	:return:
	"""
	v = list(range(powerlist.sum()))
	if len(v) % 2 != 0:
		#Odd order
		return 0

	v1 = []
	for i,power in enumerate(powerlist):
		for j in range(power):
			v1.append(i)

	if diagonal:
		for power in powerlist:
			if power%2 != 0:
				return 0

	result = 0
	count = 0
	for s in _setpartition(v):
		part = 1
		# groups = []
		for group in s:
			# groups.append((v1[group[0]],v1[group[1]])) # Just for output
			part *= Sigma_x[v1[group[0]]][v1[group[1]]]
			if part == 0:
				break
		# if part != 0:
		# 	count += 1
		# 	print groups
		result += part
	# print count
	return result



class _ndderivative(object):
	"""
	Class to calculate multidimensional derivatives of arbitrary mixed order.
	Function calls are being cached for expensive functions.
	"""
	def __init__(self,func):
		class f_class(object):
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


		self.func = f_class(func)

	def ndderivative(self,mean,powerlist,dx=1e-2):
		"""

		:param mean: The mean
		:param powerlist: list of powers for the differentiation of each variable (diff is order agnostic)
		:param dx: the distance
		:return:
		"""

		def derive_1d(func, i,n):
			"""
			:param func: The function to derive (must accept arbitrary arguments)
			:param i: number of the dimension to derive
			:param n: order of derivation
			:return: derived function
			"""

			def derived_func(mean):
				def f(x,mean,i):
					m = mean[:]
					m[i] = x
					return func(m)
				if n%2 == 0:
					order = n+1
				else:
					order = n+2

				return derivative(f,mean[i], dx=dx, args=(mean,i),n=n,order=order)


			return derived_func
		f = self.func

		for i,power in enumerate(powerlist):
			if power != 0:
				f = derive_1d(f, i, n=power)

		return f(mean)


def _get_powerlists(order,dims,leq=False,powerlist = None):
	"""
	:param order: Get the powerlists of a Taylor series for that specific order
	:param dims: Number of dimensions
	:param leq: return the powerlists with for all orders leq order or just for that specific order
	:param powerlist: Just for recursion
	:return:
	"""
	results = []
	if powerlist is None:
		powerlist = []
	sum_so_far = sum(powerlist)
	for i in range(order+1-sum_so_far):
		p = powerlist[:]
		p.append(i)
		if len(p) == dims:
			if sum(p) == order or leq:
				#if leq is True, we generate all powerlists with leq order => required for the Taylor series
				# => for the Taylor series, we just have to sum up the results for all powerlists
				results.append(np.array(p))
		else:
			powerlists = _get_powerlists(order,dims,powerlist=p,leq=leq)
			results.extend(powerlists)
	return results


class TaylorPropagation(PropagateMoments):
	"""
	Class to perform error propagation using Taylor Series
	"""

	def __init__(self, func, mean, order, dx=1e-3):
		"""

		:param func: (n-d) function to approximate
		:param mean: approximate around this mean vector
		:param order: order of the taylor series
		:param dx: step size for the derivatives
		"""
		PropagateMoments.__init__(self, func, mean)
		self.dims = len(mean)
		self.order = order
		self.dx = dx
		self.powerlists = _get_powerlists(self.order, self.dims, leq=True)
		self.derivatives = []
		nddev = _ndderivative(func)
		for powerlist in self.powerlists:
			self.derivatives.append(nddev.ndderivative(self.mean, powerlist, dx=self.dx))
		self.termlist = []
		for i, powerlist in enumerate(self.powerlists):
			term = self.derivatives[i]  # _ndderivative(self.func,self.mean,powerlist,dx=self.dx)
			term /= self._factorials(powerlist)
			self.termlist.append(term)

		print("Function calls: ", nddev.func.calls)
		# import matplotlib.pyplot as plt
		# xs = np.array(nddev.func.cache.keys())
		#
		# plt.scatter(xs.T[0],xs.T[1])
		# plt.title('Output PDF')
		# plt.show()

	def __call__(self,x):
		return self.estimate(x)

	def estimate_many(self,x_list):
		"""
		Estimate the value of the approximated function at several x

		:param x_list:
		:return: Approximated value of func at the x values
		"""

		results = []
		for x in x_list:
			results.append(self.estimate(x))
		return results

	def estimate(self,x):
		"""
		Estimate the value of the approximated function at x

		:param x:
		:return: Approximated value of func at x
		"""
		result = 0.0
		for i,powerlist in enumerate(self.powerlists):
			term = 1.0
			for j,p in enumerate(powerlist):
				term *= (x[j] - self.mean[j])**p
			result += term*self.termlist[i]
		return result

	def _factorials(self,powerlist):
		from scipy.misc import factorial
		n = 1.0
		for p in powerlist:
			n *= factorial(p,exact=True)
		return n




	def _exn(self,n,Sigma_x):
		"""
		Generates the n-th moment (not centralized!) of the output distribution

		:param n: order of the moment
		:param Sigma_x: Covariance matrix
		:return: That moment
		"""
		#@profile
		def _exn_rec(n,term,product_powerlist,isserlimap):
			"""
			Helper function for recursion

			:param n:
			:param term:
			:param product_powerlist:
			:param isserlimap:
			:return:
			"""
			ex4 = 0
			if n > 0:
				for i, powerlist in enumerate(self.powerlists):
					ex4 += _exn_rec(n-1,term*self.termlist[i],product_powerlist+powerlist,isserlimap)
			else:
				ex4 = term
				hashable_pp_list = hashable(product_powerlist)
				if hashable_pp_list not in isserlimap:
					isserlimap[hashable_pp_list] = _fast_isserli(product_powerlist,Sigma_x)
				ex4 *= isserlimap[hashable_pp_list]

			return ex4
		isserlimap = {}
		return _exn_rec(n,1,0,isserlimap)

