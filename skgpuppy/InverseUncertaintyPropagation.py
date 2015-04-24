# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
from .UncertaintyPropagation import UncertaintyPropagationExact, UncertaintyPropagationApprox
from .Utilities import minimize
import time

class InverseUncertaintyPropagation(object):

	"""
	Here, we optimize the cost function :math:`\\sum_i = c_i n_i = \\sum_i \\frac{c_i}{v_i I_{ii}}`

	* :math:`c_i`: cost per sample for input i
	* :math:`n_i`: number of samples for input i

	"""

	def __init__(self, output_variance, gp, u, c,I, input_variances=None, upga_class=UncertaintyPropagationExact, coestimated=[]):
		"""

		:param output_variance: desired maximum output variance
		:param gp: A gaussian process representing one output of one simulation
		:param u: Input vector where the uncertainty should be estimated
		:param upga_class: Class for uncertainty propagation with gaussian approximation
		:param c: cost vector for the input variances
		:param I: diagonal of the fisher Information matrix
		:param input_variances: None or list of known input variances (each unknown variance should be None in the list)
		:param coestimated: Variables, that are coestimated: list of lists of coestimated variables
			e.g. [[0,1],[2,3]] if x_0 and x_1 are parameters of the same distribution and x_2 and x_3 are from another distribution.

		.. warning:: input_variances are ignored at the moment
		"""
		self.coestimated = coestimated
		self.gp = gp
		self.upga_class = upga_class
		self.u = u
		self.output_variance = output_variance
		self.c = c
		self.I = I

	def get_best_solution(self):
		"""
		Calculate the inverse uncertainty propagation.

		:return: Optimal variances, that lead to the desired output uncertainty with minimal sampling cost
		"""
		pass


class InverseUncertaintyPropagationNumerical(InverseUncertaintyPropagation):

	def get_best_solution(self,startvalue=None):
		"""
		See baseclass method

		:param startvalue: Supply a startvalue for the optimization
		"""

		c = self.c
		I = self.I
		mapping = np.array(list(range(len(c))))
		for group in self.coestimated:
			for i in group:
				if i != group[0]:
					I = np.delete(I,i)
					c = np.delete(c,i)
					mapping = np.delete(mapping,i)

		constrs = []
		#for i,gp in enumerate(self.gp):
		upga = self.upga_class(self.gp)
		def get_constr(u,output_variance,upga):
			def constr(vx):
				new_vx = np.zeros(len(self.c))
				for i,m in enumerate(mapping):
					new_vx[m] = vx[i] # restore the original values

				for group in self.coestimated:
					for i in group:
						if i != group[0]:
							new_vx[i] = np.log(np.exp(new_vx[group[0]]) * self.I[group[0]] /self.I[i])

				return output_variance - upga.propagate_GA(u,np.diag(np.exp(new_vx)))[1]

			return constr

		constrs.append(get_constr(self.u,self.output_variance,upga))


		func = lambda vx : np.sum(c/np.exp(vx)/I)
		
		#fprime = lambda vx: [-self.c[i]/self.I[i]*np.exp(-x) for i,x in enumerate(vx)]
		if startvalue == None:
			start = np.ones(len(c))*-10
		else:
			start = startvalue

		# Gradient based methods are not suitable, because we want to find the optimum in the exact case.
		# (Well, we could but we would have to calculate the gradient of the GA propagation of the variance)
		vx_min = np.exp(minimize(func,start,constr=constrs,method=["cobyla"]))#"cobyla" ,"slsqp"
		assert((vx_min > 0 ).all())

		new_vx = np.zeros(len(self.c))
		for i,m in enumerate(mapping):
			new_vx[m] = vx_min[i] # restore the original values

		for group in self.coestimated:
			for i in group:
				if i != group[0]:
					new_vx[i] = new_vx[group[0]] * self.I[group[0]] /self.I[i]

		return new_vx



class InverseUncertaintyPropagationApprox(InverseUncertaintyPropagation):

	def __init__(self, output_variance, gp, u, c, I, input_variances=None, coestimated=[]):
		"""

		:param output_variance: desired maximum output variance
		:param gp: A gaussian process representing one output of one simulation
		:param u: Input vector where the uncertainty should be estimated
		:param c: cost vector for the input variances
		:param I: diagonal of the fisher Information matrix
		:param input_variances: None or list of known input variances (each unknown variance should be None in the list)
		:param coestimated: Variables, that are coestimated list of lists of coestimated variables

		:note input_variances are ignored at the moment
		"""
		InverseUncertaintyPropagation.__init__(self, output_variance, gp, u, c, I, input_variances= input_variances, coestimated=coestimated,upga_class=UncertaintyPropagationApprox)


	def get_best_solution(self):

		#Optimization constraints for coestimated variables
		#==> same n for a series of variables
		#==> sigma_i**2 = sigma_1**2 * I_1 / I_i
		#==> variance_dv_h is just the weighted sum of different derivations:
		# dv/dv_h is zero for all v_i that have been replaced by v_1
		# dv/dv_i is the weighted sum of the original dv/dv_1 + dv/dv_i*I_1/I_i
		#==> weight_vector calculation is different

		d = len(self.u)

		upga = self.upga_class(self.gp)

		dvdv = np.array([upga._get_variance_dv_h(self.u,i) for i in range(d)])
		for group in self.coestimated:
			for i in group:
				if i != group[0]:
					dvdv[group[0]] += dvdv[i]*self.I[group[0]]/self.I[i]

		weight_vector = np.sqrt(self.c/dvdv/self.I)
		for group in self.coestimated:
			for i in group:
				if i != group[0]:
					weight_vector[i] = weight_vector[group[0]]*self.I[group[0]]/self.I[i]

		assert((weight_vector > 0 ).all())
		#
		#get_weighted_solution
		factor = upga._getFactor(self.u,np.diag(weight_vector),self.output_variance)
		optimum = factor * weight_vector
		assert((optimum > 0 ).all())

		#TODO: Handle multiple constraints
		return optimum