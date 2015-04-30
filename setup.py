# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.

from setuptools import setup, find_packages, Extension

try:
	from Cython.Build import cythonize
	USE_CYTHON = True
except ImportError:
	USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("skgpuppy/UncertaintyPropagation2", ["skgpuppy/UncertaintyPropagation2"+ext])]

if USE_CYTHON:
	extensions = cythonize(extensions,compiler_directives={'boundscheck': False})


setup(
	name = "scikit-gpuppy",
	version = "0.9.3",
	packages = find_packages(),
	install_requires = ['scipy>=0.13.3', 'numpy>=1.8.2', 'statsmodels>=0.6.1','nose>=1.3.4'],
	extras_require = {
        'speed':  ["weave"],
        'speed alternative':  ["Cython>=0.20"]
    },
	package_data = {
		# If any package contains *.txt or *.rst files, include them:
		'': ['*.txt', '*.rst','*.pyx'],
    },
	ext_modules = extensions,
	author = "Philipp Baumgaertel",
	author_email = "philipp.baumgaertel@fau.de",
	description = "Gaussian Process Uncertainty Propagation with PYthon",
	license = "BSD",
	keywords = "gaussian process kriging random field simulation uncertainty propagation",
	url = "https://github.com/snphbaum/scikit-gpuppy",   # project home page, if any
	test_suite = 'nose.collector',
	classifiers = [
		"Programming Language :: Python",
		"Programming Language :: Python :: 2",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.4",
		"Development Status :: 4 - Beta",
		"Environment :: Other Environment",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: BSD License",
		"Natural Language :: English",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering :: Mathematics",
		],
	long_description="""\
https://github.com/snphbaum/scikit-gpuppy

This package provides means for modeling functions and simulations using Gaussian processes (aka Kriging, Gaussian random fields, Gaussian random functions).
Additionally, uncertainty can be propagated through the Gaussian processes.

.. note::
	The Gaussian process regression and uncertainty propagation are based on Girard's thesis [#]_.

	An extension to speed up GP regression is based on Snelson's thesis [#]_.

	.. warning::
		The extension based on Snelson's work is already usable but not as fast as it should be.
		Additionally, the uncertainty propagation does not yet work with this extension.

	An additional extension for Inverse Uncertainty Propagation is based on my paper (and upcoming PhD thesis) [#]_.

A simulation is seen as a function :math:`f(x)+\epsilon` (:math:`x \in \mathbb{R}^n`) with additional random error :math:`\epsilon \sim \mathcal{N}(0,v)`.
This optional error is due to the stochastic nature of most simulations.

The *GaussianProcess* module uses regression to model the simulation as a Gaussian process.

The *UncertaintyPropagation* module allows for propagating uncertainty
:math:`x \sim \mathcal{N}(\mu,\Sigma)` through the Gaussian process to estimate the output uncertainty of the simulation.

The *FFNI* and *TaylorPropagation* modules provide classes for propagating uncertainty through deterministic functions.

The *InverseUncertaintyPropagation* module allows for propagating the desired
output uncertainty of the simulation backwards through the Gaussian Process.
This assumes that the components of the input :math:`x` are estimated from samples using maximum likelihood estimators.
Then, the inverse uncertainty propagation calculates the optimal sample sizes for estimating :math:`x` that lead to the desired output uncertainty of the simulation.

.. [#] Girard, A. Approximate Methods for Propagation of Uncertainty with Gaussian Process Models, University of Glasgow, 2004
.. [#] Snelson, E. L. Flexible and efficient Gaussian process models for machine learning, Gatsby Computational Neuroscience Unit, University College London, 2007
.. [#] Baumgaertel, P.; Endler, G.; Wahl, A. M. & Lenz, R. Inverse Uncertainty Propagation for Demand Driven Data Acquisition, Proceedings of the 2014 Winter Simulation Conference, IEEE Press, 2014, 710-721
"""
)
