===============================
Inverse Uncertainty Propagation
===============================

The :doc:`InverseUncertaintyPropagation <api/skgpuppy.InverseUncertaintyPropagation>` module allows for propagating a desired
output uncertainty :math:`v_{\text{out}}` of the simulation backwards through the Gaussian Process.
This assumes that the components of the input :math:`x` are estimated from samples using maximum likelihood estimators.
Then, the inverse uncertainty propagation calculates the optimal sample sizes for estimating :math:`x` that lead to the desired output uncertainty of the simulation.

.. note::
	The following explanation is an excerpt from my paper [#]_.

We assume that measurements are utilized to estimate the uncertain simulation input parameter vector :math:`x \sim \mathcal{N}(u,\Sigma_u)`, 
which consists of the parameters of several input distributions.
Additionally, we assume the variances of the input parameters to be independent from each other: :math:`\Sigma_{u} = \mathrm{diag}(v) = \mathrm{diag}(v_1,\dots,v_D)`.
This assumption holds as for most input distributions an orthogonal parameterization 
can be found that leads to independent maximum likelihood estimates for each input parameter.
The cost for the input variance vector :math:`v= (v_{1},\dots,v_{D})` is
the number of samples to achieve a variance of :math:`v_i` multiplied with the cost :math:`c_i` to get one measurement for input :math:`x_i`.
This can be estimated using the Cramer-Rao bound:

.. math::
	\mathrm{cost}(v) = \sum_{h=1}^D \frac{c_h}{v_h \mathcal{I}_h(u_h)}

The Fisher information :math:`\mathcal{I}` is a measure of the amount of information gained by one observation.

The optimal :math:`\mathring{v}` that leads to minimal data collection cost while achieving output uncertainty :math:`v_{\text{out}}` can be estimated analytically
using the :doc:`InverseUncertaintyPropagation <api/skgpuppy.InverseUncertaintyPropagation>` module.

Coestimated Parameters
----------------------

Some input distributions may contain several parameters, which are estimated from the same sample using the maximum likelihood method.
Most distributions can be parameterized in a way that renders the estimates for its parameters independent.
However, the cost function has to incorporate the fact that all parameter estimates for this distribution stem from the same sample.
Hence, parameters :math:`i` and :math:`j` are estimated with the same sample size :math:`n_i = n_j` and therefore using the Cramer-Rao bound:
:math:`v_i \mathcal{I}_i(u_i) = v_j \mathcal{I}_j(u_j)`.
This is incorporated in our IUP method.

Unknown Input Parameters
------------------------

We assumed to know the true value :math:`u` of uncertain input parameters in advance, as we needed the true value for the inverse uncertainty propagation.
In order to be able to use our method for real simulations, we need to relax this constraint.
Using the IUP from,  we obtain the optimal data collection strategy (number of samples) by estimating the input variances leading to the lowest data acquisition cost.
Our framework is basically a deterministic function :math:`\mathrm{Opt}(u)`, which determines the optimal costs and data collection strategy for a given :math:`u`.
Now, we can ask experts to give us an approximation for :math:`u` or we can collect a small sample of data to get a rough preliminary estimation for :math:`u`.
We can represent this uncertain knowledge as a normal distribution and use well known uncertainty propagation methods for deterministic functions.
(Implemented in the :doc:`FFNI <api/skgpuppy.FFNI>` and :doc:`TaylorPropagation <api/skgpuppy.TaylorPropagation>` modules)
That way, we are able to get best, worst and average case estimates for the data collection costs and the required number of samples.

**Usage**: See :doc:`getting_started`

References
----------

.. [#] Baumgaertel, P.; Endler, G.; Wahl, A. M. & Lenz, R. Inverse Uncertainty Propagation for Demand Driven Data Acquisition, Proceedings of the 2014 Winter Simulation Conference, IEEE Press, 2014, 710-721
	(https://www6.cs.fau.de/publications/public/2014/WinterSim2014_baumgaertel.pdf)
