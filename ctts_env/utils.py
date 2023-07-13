import numpy as np


def surface_integral(t, p, q, axi_sym=False):
    """
    derive the integral for points with values q at the surface
    of a sphere of radius 1.0.

    t :: theta coordinates, 1d array
    p :: phi coordinates, 1d array

    return ::
        S           : integral of q over the stellar surface in units of r=1^2

        dOmega/4pi : the total area of the sphere in units of 4pi * 1^2
    """
    ct = np.cos(t)
    S = 0
    dOmega_o_4pi = 0
    if axi_sym:
        # 2.5d
        fact = 2 * np.pi
        # 2d ? Can be done better
        if t.min() >= 0 and t.max() <= np.pi / 2:
            fact *= 2
        i = 0
        int_theta = 0
        for j in range(1, len(t)):
            dOmega_o_4pi += abs(ct[j] - ct[j - 1]) / 4 / np.pi
            int_theta += 0.5 * (q[j, i] + q[j - 1, i]) * abs(ct[j] - ct[j - 1])
        S = int_theta * fact
        dOmega_o_4pi *= fact

        S *= 1.0 / dOmega_o_4pi
        return S, dOmega_o_4pi

    int_phi = 0
    for i in range(len(p)):
        int_theta = 0
        for j in range(1, len(t)):
            if i:
                dOmega_o_4pi += abs(ct[j] - ct[j - 1]) * (p[i] - p[i - 1]) / 4 / np.pi
            int_theta += 0.5 * (q[j, i] + q[j - 1, i]) * abs(ct[j] - ct[j - 1])
        if i:
            S += 0.5 * (int_theta + int_phi) * (p[i] - p[i - 1])
        int_phi = int_theta

    S *= 1.0 / dOmega_o_4pi
    return S, dOmega_o_4pi


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


def _old_bin_format_to_new(fold, fnew):
    """
    *** Not tested yet ***
    fold: filename in the old bin format generated with _write_deprec.
    fnew: filename to write the old format bin in the new format.

    The new format is essentially the same but not produced with scipy.io
    FortranFile, which is compatible with old fortan i/o binary handling.
    the access="sequential" has been changed to access="stream" in the fortran code
    which is more modern.

    """

    # I simply open the old file format (f) and write immediately after
    # in the new file format (fn) like Grid()_write() does.

    from scipy.io import FortranFile

    f = FortranFile(fold, "r")
    fn = open(fnew, "wb")
    Nr = f.read_ints()
    fn.write(np.array(Nr, dtype=np.int32).tobytes())
    fn.write(np.single(f.read_record(np.single)).tobytes())
    Nt = f.read_ints()
    fn.write(np.array(Nt, dtype=np.int32).tobytes())
    fn.write(np.single(f.read_record(np.single)).tobytes())
    Np = f.read_ints()
    fn.write(np.array(Np, dtype=np.int32).tobytes())
    fn.write(np.single(f.read_record(np.single)).tobytes())
    shape = (Nr, Nt, Np)

    fn.write(np.array(f.read_ints(), dtype=np.int32).tobytes())
    fn.write(np.array(f.read_reals(), dtype=float).tobytes())
    fn.write(np.array(f.read_reals(), dtype=float).tobytes())

    # data might be transposed on reading though to check

    fn.write(np.reshape(f.read_record(float), shape).tobytes())
    fn.write(np.reshape(f.read_record(float), shape).tobytes())
    fn.write(f.read_record(float).tobytes())

    v3d = np.reshape(f.read_record(np.float32), (3, Np[0], Nt[0], Nr[0]))
    # float 32 for real and float for double precision kind=dp
    fn.write(np.float32(v3d).tobytes())
    fn.write(f.read_record(float).tobytes())

    fn.write(f.read_record(np.intc).tobytes())

    f.close()
    fn.close()

    return
