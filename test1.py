## test impl RSH

import timeit


## the rsh use the scipy impl anyway so theres no point to this.

from lie_learn.representations.SO3.spherical_harmonics import rsh, block_sh_ph

import numpy as np

from scipy.special import sph_harm

def sph_real(l, m, phi, theta):
	scale = 1
	if m != 0:
		scale = np.sqrt(2)
	if m >= 0:
		return scale * sph_harm(m, l, phi, theta).real
	else:
		return scale * sph_harm(abs(m), l, phi, theta).imag


max_degree = 5
normalization='quantum'
condon_shortley=True


phi = np.random.uniform(-np.pi,np.pi,1000)
theta   = np.random.uniform(0,np.pi,1000)

def testEqual():
	for l in range(0, max_degree):
		for m in range(-l, l + 1):
			Y1 = rsh(l, m, theta, phi,normalization=normalization, condon_shortley=condon_shortley)
			Y2 = sph_real(l, m, phi, theta)

			print(np.allclose(Y1, Y2))


def evalScipy():
	for l in range(0, max_degree):
		for m in range(-l, l + 1):
			# Y1 = rsh(l, m, theta, phi,normalization=normalization, condon_shortley=condon_shortley)
			Y2 = sph_real(l, m, phi, theta)


def evalLie():
	for l in range(0, max_degree):
		for m in range(-l, l + 1):
			# Y1 = rsh(l, m, theta, phi,normalization=normalization, condon_shortley=condon_shortley)
			# Y2 = sph_real(l, m, phi, theta)


testEqual()
# result = timeit.timeit('evalScipy()', globals=globals(), number=10)

# print("scipy:", result)

# result = timeit.timeit('evalLie()', globals=globals(), number=10)

# print("lie_learn:", result)

# no work :
L = block_sh_ph(max_degree, theta, phi)

print(L)