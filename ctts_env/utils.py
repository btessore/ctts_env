import numpy as np


def surface_integral(t, p, q, axi_sym=False):
    """
    derive the ntegral for points with values q at the surface
    of a sphere of radius 1.0.

    t :: theta coordinates, 1d array
    p :: phi coordinates, 1d array

    return :: S in units or 1.0 sphere radius squared.
                  Omega, the total solid angle of the sphere (=1 in unit of 4pi)
    """
    ct = np.cos(t)
    S = 0
    dOmega = 0
    if axi_sym:
        # 2.5d
        fact = 2 * np.pi
        # 2d ? Can be done better
        if t.min() >= 0 and t.max() <= np.pi / 2:
            fact *= 2
        i = 0
        int_theta = 0
        for j in range(1, len(t)):
            dOmega += abs(ct[j] - ct[j - 1]) / 4 / np.pi
            int_theta += 0.5 * (q[j, i] + q[j - 1, i]) * abs(ct[j] - ct[j - 1])
        S = int_theta * fact
        dOmega *= fact

        S *= 1.0 / dOmega
        return S, dOmega

    int_phi = 0
    for i in range(len(p)):
        int_theta = 0
        for j in range(1, len(t)):
            if i:
                dOmega += abs(ct[j] - ct[j - 1]) * (p[i] - p[i - 1]) / 4 / np.pi
            int_theta += 0.5 * (q[j, i] + q[j - 1, i]) * abs(ct[j] - ct[j - 1])
        if i:
            S += 0.5 * (int_theta + int_phi) * (p[i] - p[i - 1])
        int_phi = int_theta

    S *= 1.0 / dOmega
    return S, dOmega


def spherical_to_cartesian(r, t, p, ct, st, cp, sp):

    x = r * st * cp + t * ct * cp - sp * p
    y = r * st * sp + t * ct * sp + cp * p
    z = r * ct - t * st

    return x, y, z


def cartesian_to_spherical(x, y, z, ct, st, cp, sp):

    r = x * st * cp + y * st * sp + z * ct
    t = x * ct * cp + y * ct * sp - z * st
    p = -x * sp + y * cp

    return r, t, p


def centrifugal_barrier(beta):
    barrier = (2 / (2 + np.cos(np.deg2rad(beta)) ** 2)) ** (1 / 3)
    return barrier  # in units of Rco !


def Gamma(Rt, dr):
    """
    Axisymmetric area of the shock.
    Rt :: Inner truncation radius
    dr :: width of the accretion on the disc
    """
    return np.sqrt(1.0 - 1 / (Rt + dr)) - np.sqrt(1.0 - 1 / Rt)


def shock_area(Rt, dr, beta=0, f=1):
    """
    Area of the shock for arbitrary obliquity of the magnetic dipole.
    Rt      :: Inner truncation radius
    dr      :: width of the accretion on the disc
    beta    :: magnetic obliquity [deg]
    f       :: shape factor. In secondary columns are completly removed,
                f = 0.5. (f in ~[0.5, 1])
    """
    return f * Gamma(Rt, dr) * np.cos(np.deg2rad(beta))


def fw3d_average(d):
    """'
    d is a 3d array
    Faster in python than explicit sum.
    """
    from scipy import ndimage

    shape = d.shape

    # 	kernel = np.ones((3,3,3)) / 27
    kernel = np.zeros((3, 3, 3))
    K1 = np.matrix("1 2 1; 2 4 2; 1 2 1")
    K2 = np.matrix("2 4 2; 4 8 4; 2 4 2")
    K3 = np.matrix("1 2 1; 2 4 2; 1 2 1")
    kernel[0] = K1
    kernel[1] = K2
    kernel[2] = K3
    kernel /= 64
    return ndimage.convolve(d, kernel, mode="nearest")


def reduction(q, l=1):
    """
    Reduce the data information with full-weighted method.
    """
    for k in range(l):
        q = fw3d_average(q)
    return
