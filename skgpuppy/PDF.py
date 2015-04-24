# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
from scipy.stats import norm

class PDF(object):
	"""
	Superclass for normal and skew normal distributions.
	This class provides methods for plotting and getting quantiles.
	"""
	def output_pdf(self,mean,var,skew,kurtosis,x_list):
		"""
		Function to plot the output pdf

		:param mean:
		:param var:
		:param skew:
		:param kurtosis:
		:param x_list:
		:return:
		"""
		pass

	def estimate_min_max(self,mean,var,skew,kurtosis,percentile):
		"""
		Min and max of coverage intervall

		:param mean:
		:param var:
		:param skew:
		:param kurtosis:
		:param percentile: percent of coverage
		:return:
		"""
		pass


class Normal(PDF):
	def output_pdf(self,mean,var,skew,kurtosis,x_list):
		"""
		Function to plot the normal output pdf

		:param mean:
		:param var:
		:param skew:
		:param kurtosis:
		:param x_list:
		:return:
		"""
		result = []
		for x in x_list:
			result.append(norm.pdf(x,loc=mean,scale=np.sqrt(var)))

		return result

	def estimate_min_max(self,mean,var,skew,kurtosis,percentile):
		"""
		Min and max of coverage intervall using the normal distribution

		:param mean:
		:param var:
		:param skew:
		:param kurtosis:
		:param percentile: percent of coverage
		:return:
		"""

		_min = norm.ppf((1-percentile)/2,loc=mean,scale=np.sqrt(var))
		_max = norm.ppf(percentile/2+0.5,loc=mean,scale=np.sqrt(var))
		return _min, _max


class Skew_Normal(PDF):
	def _loc_scale_shape(self,mean,var,skew):
		"""
		Gets the location, scale and shape for the skew normal
		=> Solution of the equations of the method of moments

		:param mean:
		:param var:
		:param skew:
		:return:
		"""
		skew = np.sign(skew)*min(0.995,np.abs(skew))
		y23 = np.power(np.abs(skew),(2.0/3.0))
		pi23 = np.power(((4-np.pi)/2),(2.0/3.0))
		delta = np.sign(skew) * np.sqrt(np.pi/2*y23/(y23 + pi23))
		shape = delta / np.sqrt(1-delta**2)
		scale =  np.sqrt(var/(1-(2*delta**2)/np.pi))
		location = mean - scale * delta * np.sqrt(2/np.pi)

		return location,scale,shape

	def output_pdf(self,mean,var,skew,kurtosis,x_list):
		"""
		Helper function to plot the skew normal

		:param mean:
		:param var:
		:param skew:
		:param x_list:
		:return:
		"""
		location,scale,shape = self._loc_scale_shape(mean,var,skew)

		result = []
		for x in x_list:
			t= (x-location)/scale
			nonsn_result =  2/scale * norm.pdf(t) * norm.cdf(shape*t)
			result.append(nonsn_result)
		return result

	def estimate_min_max(self,mean,var,skew,kurtosis,percentile):
		"""
		Min and max of coverage intervall using the skew normal

		:param mean:
		:param var:
		:param skew:
		:param percentile: percent of coverage
		:return:
		"""
		#=> Called moment matching or method of moments and is superseeded by MLE 1930 (Battle of estimators)
		#http://www.johndcook.com/blog/2010/09/20/skewness-andkurtosis/
		#One could directly use a pearson distribution

		location,scale,shape = self._loc_scale_shape(mean,var,skew)

		from statsmodels.sandbox.distributions.extras import skewnorm2
		_min = skewnorm2.ppf((1-percentile)/2,shape,loc=location,scale=scale)
		_max = skewnorm2.ppf(percentile/2+0.5,shape,loc=location,scale=scale)
		return _min, _max


# rpy2 is GPL ==> can not be used
#
# class Pearson(PDF):
#
# 	def output_pdf(self,mean,var,skew,kurtosis,x_list):
# 		"""
# 		Function to plot the pearson output pdf

# 		:param mean:
# 		:param var:
# 		:param skew:
# 		:param kurtosis:
# 		:param x_list:
# 		:return:
# 		"""
# 		#In R: install.packages("PearsonDS")
# 		from rpy2.robjects.packages import importr
# 		from rpy2.robjects import FloatVector
#
# 		p = importr("PearsonDS")
#
# 		result = []
# 		for x in x_list:
# 			result.append(p.dpearson(x,FloatVector([0]),FloatVector([mean,var,skew,kurtosis]))[0])
#
#
# 		return result
#
#
# 	def estimate_min_max(self,mean,var,skew,kurtosis,percentile):
# 		"""
# 		Min and max of coverage intervall using the pearson distribution system
#
# 		:param mean:
# 		:param var:
# 		:param skew:
# 		:param kurtosis:
# 		:param percentile: percent of coverage
# 		:return:
# 		"""
# 		from rpy2.robjects.packages import importr
# 		from rpy2.robjects import FloatVector
#
# 		p = importr("PearsonDS")
# 		#print (1-percentile)/2
# 		#print [mean,var,skew,kurtosis]
# 		_min = p.qpearson((1-percentile)/2,FloatVector([0]),FloatVector([mean,var,skew,kurtosis]))[0]
# 		_max = p.qpearson(percentile/2+0.5,FloatVector([0]),FloatVector([mean,var,skew,kurtosis]))[0]
# 		return _min, _max