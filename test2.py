import timeit

from scipy.spatial.transform import Rotation as R


from lie_learn.SO3.SO3FFT import SO3_FFT_synthesize
from lie_learn.SO3.irrep_bases import change_of_basis_matrix
from lie_learn.SO3.pinchon_hoggan_dense import SO3_irrep

import numpy as np

np.random.seed(2022)

l_max = 24

phi = np.random.uniform(-np.pi,np.pi,1000)
theta = np.random.uniform(0,np.pi,1000)


k = 0.01

l_max_weights = l_max

w1 = np.zeros(l_max**2)
w2 = np.zeros(l_max**2)

for l in range(0, l_max):
	i = int(l)**2
	i2 = int(l+1)**2

	A = np.exp(-k*l*(l+1))
	for m in range(i,i2):
		w1[m] = A * np.random.normal()
		w2[m] = A * np.random.normal()

print(w1)

# change of basis from real to complex
r2c = [
	change_of_basis_matrix(
		l,
		frm=("real", "quantum", "centered", "cs"),
		to=("complex", "quantum", "centered", "cs"),
	)
	for l in range(l_max_weights)
]

# zero fill initial (cf. fft zero padding to get higher resolution)
f_hat = np.array(
	[np.zeros((2 * ll + 1, 2 * ll + 1)) for ll in range(l_max)], dtype=object
)

# calculate the outer conjugate product of each subspace
# and compute the real to complex change of basis:
#  f_hat = B * (w1 (x) w2) * B^H
# where B is the change of basis matrix, (x) is outer product
for l in range(l_max_weights):
	f_hat[l] = r2c[l].dot(
		np.outer(w1[l**2 : (l + 1)**2], w2[l**2 : (l + 1)**2]).dot(
			r2c[l].T.conj()
		)
	)

print("start ifft")
# take inverse fourier transform
res = SO3_FFT_synthesize(f_hat)
print("end ifft")

# the correlation function of two real functions is also real
assert np.allclose(res.imag, 0)

res = res.real

# get maximum indices
ind = np.unravel_index(res.argmax(), res.shape)
# convert grid indices to euler angles
alpha = 2 * np.pi * ind[0] / res.shape[0]
beta = np.pi * (2 * ind[1] + 1) / (2 * res.shape[1])
gamma = 2 * np.pi * ind[2] / res.shape[2]

print(ind)
print(alpha, beta, gamma)


rot = R.from_euler("zyz", [alpha, beta, gamma])
g = rot.as_euler("ZYZ")
g[1] = -g[1]

r_mat = np.array(
		[np.squeeze(SO3_irrep(np.atleast_2d(g).T, l)) for l in range(l_max)],
		dtype=object,
	)

weights_rot = np.empty(w1.shape)
# rotate each degree by the respective block
for l in range(l_max):
	weights_rot[l**2 : (l + 1)**2] = r_mat[l].dot(
		w1[l**2 : (l + 1)**2]
	)

print("--------------------------------------------")



result = timeit.timeit('SO3_FFT_synthesize(f_hat)', globals=globals(), number=10)
print("fft:", result)


def cb_calc():
	r2c = [
		change_of_basis_matrix(
			l,
			frm=("real", "quantum", "centered", "cs"),
			to=("complex", "quantum", "centered", "cs"),
		)
		for l in range(l_max_weights)
	]

result = timeit.timeit('cb_calc()', globals=globals(), number=10)
print("cb_calc:", result)


def outer():
	for l in range(l_max_weights):
		f_hat[l] = r2c[l].dot(
			np.outer(w1[l**2 : (l + 1)**2], w2[l**2 : (l + 1)**2]).dot(
				r2c[l].T.conj()
			)
		)

result = timeit.timeit('outer()', globals=globals(), number=10)
print("outer:", result)


def rotate_calc():
	r_mat = np.array(
		[np.squeeze(SO3_irrep(np.atleast_2d(g).T, l)) for l in range(l_max)],
		dtype=object,
	)

result = timeit.timeit('rotate_calc()', globals=globals(), number=10)
print("rotation:", result)


### 32, 1000
# fft: 0.5448626000000001
# cb_calc: 0.1223109
# outer: 0.024562799999999996
# rotation: 0.05395109999999992