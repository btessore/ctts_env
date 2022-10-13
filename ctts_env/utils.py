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
    # 	for i in range(len(p)):
    # 		for j in range(1,len(t)):
    # 			if i:
    # 				dOmega += abs(ct[j] - ct[j-1]) * (p[i] - p[i-1]) / 4 / np.pi
    # 				S += 0.5*(q[j,i]+q[j-1,i-1]) * abs(ct[j] - ct[j-1]) * (p[i] - p[i-1]) / 4 / np.pi
    #

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
        return S, dOmega * fact

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

    return S, dOmega


def compute_cells_volume(r, t, p):
    # r is Nr size
    # t is Nt size
    # p is Np size
    # -> minimum size is 2 for gradient

    dt = np.gradient(t)
    dr = np.gradient(r)
    dp = np.gradient(p)
    vol = np.zeros(len(r) * len(t) * len(p))
    id = -1
    for i in range(len(r)):
        for j in range(len(t)):
            for k in range(len(p)):
                id += 1
                vol[id] = r[i] * r[i] * dr[i] * np.sin(t[j]) * dt[j] * dp[k]

    return vol


def compute_cells_surface(t, p, R=1):
    # t is Nt size
    # p is Np size

    dt = np.gradient(t)
    dp = np.gradient(p)
    surf = np.zeros(len(t) * len(p))
    id = -1
    for j in range(len(t)):
        for k in range(len(p)):
            id += 1
            surf[id] = R ** 2 * np.sin(t[j]) * dt[j] * dp[k]

    return surf


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
