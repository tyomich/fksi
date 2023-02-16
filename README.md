# fksi

Simple python realisation of the Simpson's algorithm for computing Fresnel-Kirchhoff integral with an axially symmetric phase delay specified by an array of values sampled at arbitrary radii; the delay is internally represented by a spline interpolation of order up to the fourth, and the algorithm adapts to a user-requested precision limit. The algorithm is described in the appendix to Suvorov et al. 2023.

The main funtion, defined in fksi.py, is fksi(y, w, x, f) where y is the observer position and w frequency while x and f are phase delay sampling radii (assumed increasing) and values (in radians); the phase delay is supposed to vanish outside the maximum radius. The additional parameters are 
et=2e-17 - requested precision, 
o=1 - requested spline polynomial order (up to 4 but beware Runge phenomenon), 
maxev=np.infty - limit on the total number of integrand evalauations, 
diagnose=False - if True, prints out diagnostic information, 
ftype=float, ctype=complex - type of arguments to pass to scipy.special.j0 and numpy.exp.

A few auxiliary functions are defined, including sc(xd, fd, o=1) computing the interpolating polynomial coefficients.

fksi.ipynb is a jupyter notebook with a few usage examples.
