===============
Getting Started
===============

Requirements
------------

* scipy
* numpy
* statsmodels
* nose

Usage
-----

**Regression**

::

	import numpy as np
	from skgpuppy.GaussianProcess import GaussianProcess
	from skgpuppy.Covariance import GaussianCovariance

	# Preparing some parameters (just to create the example data)
	x = np.array([[x1,x2] for x1 in xrange(10) for x2 in xrange(10)]) # 2d sim input (no need to be a neat grid in practice)
	w = np.array([0.04,0.04])   # GP bandwidth parameter
	v = 2                       # GP variance parameter
	vt = 0.01                   # GP variance of the error epsilon

	# Preparing the parameter vector
	theta = np.zeros(2+len(w))
	theta[0] = np.log(v)  # We actually use the log of the parameters as it is easier to optimize (no > 0 constraint etc.)
	theta[1] = np.log(vt)
	theta[2:2+len(w)] = np.log(w)

	# Simulating simulation data by drawing data from a random Gaussian process
	t = GaussianProcess.get_realisation(x, GaussianCovariance(),theta)

	# The regression step is pretty easy:
	# Input data x (list of input vectors)
	# Corresponding simulation output t (just a list of floats of the same length as x)
	# Covariance function of your choice (only GaussianCovariance can be used for uncertainty propagation at the moment)
	gp_est = GaussianProcess(x, t,GaussianCovariance())

	# Getting some values from the regression GP for plotting
	x_new = np.array([[x1/2.0,x2/2.0] for x1 in xrange(20) for x2 in xrange(20)])
	means, variances = gp_est.estimate_many(x_new)

	# Plotting the output
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(x.T[0],x.T[1],t, cmap=cm.autumn, linewidth=0.2)
	ax.plot_trisurf(x_new.T[0],x_new.T[1],means, cmap=cm.winter, linewidth=0.2)
	plt.show()


**Uncertainty Propagation**

::

	# Continuing the regression example

	from skgpuppy.UncertaintyPropagation import UncertaintyPropagationApprox

	# The uncertainty to be propagated
	mean = np.array([5.0,5.0]) # The mean of a normal distribution
	Sigma = np.diag([0.01,0.01]) # The covariance matrix (must be diagonal because of lazy programming)

	# Using the gp_est from the regression example
	up = UncertaintyPropagationApprox(gp_est)

	# The propagation step
	out_mean, out_variance = up.propagate_GA(mean,Sigma)

	print out_mean, out_variance


**Inverse Uncertainty Propagation**

::

	# Continuing the propagation example

	from skgpuppy.InverseUncertaintyPropagation import InverseUncertaintyPropagationApprox

	# The fisher information matrix for the maximum likelihood estimation of x
	# This assumes both components of x to be rate parameters of exponential distributions
	I = np.array([1/mean[0]**2,1/mean[1]**2])

	# cost vector: the cost for collecting one sample for the estimation of the components of x
	c = np.ones(2) # Collecting one sample for each component of x costs 1

	# The cost for collecting enough samples to approximately get the Sigma from above (Cramer-Rao-Bound)
	print (c/I/np.diag(Sigma)).sum()

	# The desired output variance (in this example) is out_variance
	# Getting the Sigma that leads to the minimal data collection costs while still yielding out_variance
	# If multiple parameters from the same distribution (and therefore the same sample) have to be estimated, we could use the optional parameter "coestimated"
	iup = InverseUncertaintyPropagationApprox(out_variance,gp_est,mean,c,I)
	Sigma_opt = np.diag(iup.get_best_solution())

	# The optimal data collection cost to get the output variance out_variance
	print (c/I/np.diag(Sigma_opt)).sum()

	# Proof that we actually do get close to out_variance using Sigma_opt
	out_mean, out_variance2 = up.propagate_GA(mean,Sigma_opt)
	print out_mean, out_variance2

