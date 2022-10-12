import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# Lambda = Q/nH**2 in erg/cm3/s, Radiative loss ("cooling") function.
# 	--> reference table from Hartmann et al. 1982.
logT_hart82 = np.array([3.70, 3.80, 3.90, 4.00, 4.20, 4.60, 4.90])  # log10
logLambda_hart82 = np.array(
    [-28.3, -26.0, -24.5, -23.6, -22.6, -21.8, -21.2]
)  # radiative loss

# Table bounded by "realistic" values expected in accretion columns, at maximum.
# tab_logT_prepend = np.array([2.7, 3., 3.18, 3.30, 3.48,3.60])
tab_logT = np.array([3.70, 3.80, 3.90, 4.00, 4.08])  # , 4.20, 4.60, 4.90])
# tab_logRL_prepend = np.array([-51.3,-44.4,-40.26,-37.5,-33.36,-30.6])
tab_logRadLoss = np.array(
    [-28.3, -26.0, -24.5, -23.6, -23.1]
)  # , -22.6])#, -21.8, -21.2])
spl_order = 1
logL = InterpolatedUnivariateSpline(
    logT_hart82, logLambda_hart82, k=spl_order
)  # extrapolated (ext=0)
logT_extrp = InterpolatedUnivariateSpline(
    tab_logRadLoss, tab_logT, k=spl_order
)  # extrapolated (ext=0)
logT_bound = InterpolatedUnivariateSpline(
    tab_logRadLoss, tab_logT, k=spl_order, ext=3
)  # bounded by min max of Hartmann (ext=3).


def T_to_logRadLoss(x):
    """
    from x in Kelvin, return the log10 of
    the radiative loss function at that x.
    """
    return logL(np.log10(x))


def logRadLoss_to_T(x, extrapolate_up=False):
    """
    Return the temperature from x, the radiative loss function.
    Range of values returned depends on the tables (tab_logT,tab_logRadLoss),
    and the extrapolation methods.

    The values of x below tab_logRadLoss.min() are linearly extrapolated (extrapolate_down=True)

    """

    tmp = np.zeros(x.shape)
    # extrapolate downward tab_logT.min()
    tmp[x < tab_logRadLoss[0]] = 10 ** logT_extrp(x[x < tab_logRadLoss[0]])

    if extrapolate_up:
        tmp[x >= tab_logRadLoss[0]] = 10 ** logT_extrp(x[x >= tab_logRadLoss[0]])
    else:
        tmp[x >= tab_logRadLoss[0]] = 10 ** logT_bound(x[x >= tab_logRadLoss[0]])

    return tmp


def compute_temp(r, rho, Tmax, B=0, type=0, Tmax_sec=0, mcol=[]):
    """
    r, rho must have the same shape

    TO DO: fortran wrapper

    NOTE: works only with analytical magnetospheric models.
    """
    #
    mask = rho > 0
    T = np.zeros(r.shape)

    if type == 1:  # normalize Hartmann to avg = Tmax
        rl = r[mask] ** -3 * rho[mask] ** -2
        T[mask] = logRadLoss_to_T(np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax))
        Tavg = np.average(T[mask], weights=rho[mask])
        T *= Tmax / Tavg

    elif type == 2:  # type==0 with B
        if B.shape != rho.shape:
            print("compute_temp error: B is required with type==2")
        rl = B[mask] * rho[mask] ** -2
        T[mask] = logRadLoss_to_T(np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax))

    elif type == 3:  # type==1 with B
        if B.shape != rho.shape:
            print("compute_temp error: B is required with type==2")
        rl = B[mask] * rho[mask] ** -2
        T[mask] = logRadLoss_to_T(np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax))
        Tavg = np.average(T[mask], weights=rho[mask])
        T *= Tmax / Tavg

    elif type == 4:  # hartmann with independent normalisation for secondary column
        if not Tmax_sec:
            Tmax_sec = Tmax  # same Tmax
        main_col = mcol
        sec_col = ~mcol
        rl = r[main_col] ** -3 * rho[main_col] ** -2
        T[main_col] = logRadLoss_to_T(np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax))
        rl = r[sec_col] ** -3 * rho[sec_col] ** -2
        T[sec_col] = logRadLoss_to_T(
            np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax_sec)
        )
    elif type == 5:  # type 4 with B
        if not Tmax_sec:
            Tmax_sec = Tmax  # same Tmax
        main_col = mcol * mask
        sec_col = ~mcol * mask
        rl = B[main_col] * rho[main_col] ** -2
        T[main_col] = logRadLoss_to_T(np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax))
        rl = B[sec_col] * rho[sec_col] ** -2
        T[sec_col] = logRadLoss_to_T(
            np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax_sec)
        )
    else:  # Classic Hartmann for axisymmetric models
        rl = r[mask] ** -3 * rho[mask] ** -2
        T[mask] = logRadLoss_to_T(np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax))

    return T
