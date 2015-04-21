.. scikit-GPUPPY documentation master file, created by
	sphinx-quickstart on Wed Apr 15 14:23:22 2015.
	You can adapt this file completely to your liking, but it should at least
	contain the root `toctree` directive.

===================================================================
scikit-GPUPPY: Gaussian Process Uncertainty Propagation with PYthon
===================================================================

This package provides means for modeling functions and simulations using Gaussian processes (aka Kriging, Gaussian random fields, Gaussian random functions).
Additionally, uncertainty can be propagated through the Gaussian processes.

.. note::
	The Gaussian process regression and uncertainty propagation are based on Girard's thesis [#]_.

	An extension to speed up GP regression is based on Snelson's thesis [#]_.

	.. warning::
		The extension based on Snelson's work is already usable but not as fast as it should be.
		Additionally, the uncertainty propagation does not yet work with this extension.

	An additional extension for :doc:`iup` is based on my paper (and upcoming PhD thesis) [#]_.


A simulation is seen as a function :math:`f(x)+\epsilon` (:math:`x \in \mathbb{R}^n`) with additional random error :math:`\epsilon \sim \mathcal{N}(0,v)`.
This optional error is due to the stochastic nature of most simulations.

The :doc:`GaussianProcess <api/skgpuppy.GaussianProcess>` module uses regression to model the simulation as a Gaussian process.
(See :doc:`regression` for an explanation)

The :doc:`UncertaintyPropagation <api/skgpuppy.UncertaintyPropagation>` module allows for propagating uncertainty
:math:`x \sim \mathcal{N}(\mu,\Sigma)` through the Gaussian process to estimate the output uncertainty of the simulation.
(See :doc:`up` for an explanation)

The :doc:`FFNI <api/skgpuppy.FFNI>` and :doc:`TaylorPropagation <api/skgpuppy.TaylorPropagation>` modules provide classes for propagating uncertainty through deterministic functions.

The :doc:`InverseUncertaintyPropagation <api/skgpuppy.InverseUncertaintyPropagation>` module allows for propagating the desired
output uncertainty of the simulation backwards through the Gaussian Process.
This assumes that the components of the input :math:`x` are estimated from samples using maximum likelihood estimators.
Then, the inverse uncertainty propagation calculates the optimal sample sizes for estimating :math:`x` that lead to the desired output uncertainty of the simulation.
(See :doc:`iup` for an explanation)


Contents
--------

.. toctree::
	:maxdepth: 2

	getting_started
	regression
	up
	iup
	api/skgpuppy


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
----------

.. [#] Girard, A. Approximate Methods for Propagation of Uncertainty with Gaussian Process Models, University of Glasgow, 2004
.. [#] Snelson, E. L. Flexible and efficient Gaussian process models for machine learning, Gatsby Computational Neuroscience Unit, University College London, 2007
.. [#] Baumgaertel, P.; Endler, G.; Wahl, A. M. & Lenz, R. Inverse Uncertainty Propagation for Demand Driven Data Acquisition, Proceedings of the 2014 Winter Simulation Conference, IEEE Press, 2014, 710-721
