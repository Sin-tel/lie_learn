
import numpy as np

import os

base = 'J_block_0-150.npy'
path = os.path.join(os.path.dirname(__file__), 'pinchon_hoggan', base)
Jb = np.load(path, allow_pickle=True)


def block_sh_ph(L_max, theta, phi):
    """
    Compute all spherical harmonics up to (not including) degree L_max, for angles theta and phi.

    This function is currently rather hacky, but the method used here is very fast and stable, compared
    to builtin scipy functions.

    :param L_max:
    :param theta:
    :param phi:
    :return:
    """

    from .pinchon_hoggan.pinchon_hoggan import apply_rotation_block, make_c2b
    from .irrep_bases import change_of_basis_function

    irreps = np.arange(L_max)

    ls = [[ls] * (2 * ls + 1) for ls in irreps]
    ls = np.array([ll for sublist in ls for ll in sublist])  # 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
    ms = [list(range(-ls, ls + 1)) for ls in irreps]
    ms = np.array([mm for sublist in ms for mm in sublist])  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...

    # Get a vector Y that selects the 0-frequency component from each irrep in the centered basis
    # If D is a Wigner D matrix, then D Y is the center column of D, which is equal to the spherical harmonics.
    Y = (ms == 0).astype(float)

    # Change to / from the block basis (since the rotation code works in that basis)
    c2b = change_of_basis_function(irreps,
                                   frm=('real', 'quantum', 'centered', 'cs'),
                                   to=('real', 'quantum', 'block', 'cs'))
    b2c = change_of_basis_function(irreps,
                                   frm=('real', 'quantum', 'block', 'cs'),
                                   to=('real', 'quantum', 'centered', 'cs'))

    Yb = c2b(Y)

    # Rotate Yb:
    c2b = make_c2b(irreps)

    global Jb

    J_block = list(Jb[irreps])

    g = np.zeros((theta.size, 3))
    g[:, 0] = phi
    g[:, 1] = theta
    TYb = apply_rotation_block(g=g, X=Yb[np.newaxis, :],
                               irreps=irreps, c2b=c2b,
                               J_block=J_block, l_max=np.max(irreps))

    # print(Yb.shape, TYb.shape)

    # Change back to centered basis
    TYc = b2c(TYb.T).T  # b2c doesn't work properly for matrices, so do a transpose hack

    # print(TYc.shape)

    # Somehow, the SH obtained so far are equal to real, nfft, cs spherical harmonics
    # Change to real quantum centered cs
    c = change_of_basis_function(irreps,
                                 frm=('real', 'nfft', 'centered', 'cs'),
                                 to=('real', 'quantum', 'centered', 'cs'))
    TYc2 = c(TYc)
    # print(TYc2.shape)

    return TYc2

