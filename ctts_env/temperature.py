import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# Lambda = Q/nH**2 in erg/cm3/s, Radiative loss ("cooling") function.
# 	--> reference table from Hartmann et al. 1982.
logT_hart82 = np.array([3.70, 3.80, 3.90, 4.00, 4.20, 4.60, 4.90, 5.0])  # log10
logLambda_hart82 = np.array(
    [-28.3, -26.0, -24.5, -23.6, -22.6, -21.8, -21.2, -21.0]
)  # radiative loss

spl_order = 2
logL = InterpolatedUnivariateSpline(
    logT_hart82, logLambda_hart82, k=spl_order
)  # extrapolated (ext=0)
logT = InterpolatedUnivariateSpline(
    logLambda_hart82, logT_hart82, k=spl_order
)  # extrapolated (ext=0)
logT_bound = InterpolatedUnivariateSpline(
    logLambda_hart82, logT_hart82, k=spl_order, ext=3
)  # bounded by min max of Hartmann (ext=3).


def T_to_logRadLoss(x):
    """
    from x in Kelvin, return the log10 of
    the radiative loss function at that x.
    """
    return logL(np.log10(x))


def logRadLoss_to_T(x, extrapolate=False, T_min_limit=2000):
    """
    Return the temperature from x, the radiative loss function.
    The minium temperature is bounded.
    """

    if extrapolate:
        return np.maximum(10 ** logT(x), T_min_limit)
    else:
        return 10 ** logT_bound(x)
