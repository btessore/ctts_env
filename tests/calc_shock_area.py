"""
	
	Compute the shock area
	
"""
import ctts_env
from ctts_env.constants import *

import numpy as np
import os
from glob import glob


from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm, TwoSlopeNorm


##########################################################################################


def shock_surface_MK(Rt, beta):
    """
    Approximation for Rt >> 1
    from Mahdavi & Kenyon 98
    """
    return 1 / Rt * np.cos(np.deg2rad(beta)) * 0.25


def shock_surface_kulkarni(rmi, rmo, beta):
    """
    Kulkarni & Romanova, MNRAS 433, 3048–3061 (2013)
    \S 4.2.1, Eqs. 2 and 3.
    """
    sin2 = np.sin(np.pi / 2 - np.deg2rad(beta)) ** 2
    sto2 = 1 / rmo * sin2
    sti2 = 1 / rmi * sin2
    return np.sqrt(1.0 - sto2) - np.sqrt(1.0 - sti2)


def T_kin(rho, vr, f=3 / 4):
    """
    Compute the temperature of the shock assuming
    Ishock = f * kinetic energy
    """
    tk = (f * 0.5 * rho / sigma * abs(vr ** 3)) ** (1 / 4)
    return tk


def plot_shock_surface(
    star,
    rmi=2.2,
    rmo=3.0,
    Mdot=1e-8,
    Nt=400,
    Np=400,
    Nobliquity=50,
    obliquity_limits=[0, 89],
    save=False,
    verbose=False,
):
    """
    plot the shock surface for a magnetic configuration
    and for different obliquities from 0 to 90 degrees.

    star is an instance of ctts_env.Star(), it is mandatory for computing the shock
    surface from the density

    """

    # generate a grid at the surface of te star
    rr = [1.0]  # np.linspace(1.0, star.Rco, 10)
    tt = np.linspace(1e-5, np.pi - 1e-5, Nt)
    pp = np.linspace(1e-5, 2 * np.pi - 1e-5, Np)
    r, t, p = np.meshgrid(rr, tt, pp, indexing="ij")

    g = ctts_env.Grid(r, t, p)
    g.calc_cells_surface()

    S_cells = g.surface
    S_cells *= 4 * np.pi / S_cells.sum()

    beta_ma = np.linspace(obliquity_limits[0], obliquity_limits[1], Nobliquity)

    S = np.zeros(Nobliquity)
    S_check = np.zeros(Nobliquity)

    analytical_surf = 100 * ctts_env.utils.shock_area(rmi, rmo - rmi, beta=beta_ma)
    kulkarni = shock_surface_kulkarni(rmi, rmo, beta_ma)

    for k, tilt in enumerate(beta_ma):
        # g.add_magnetosphere_v1(star, rmi=rmi, rmo=rmo, Mdot=Mdot, beta=tilt)
        g.add_mag(star, rmi=rmi, rmo=rmo, Mdot=Mdot, beta=tilt)
        # g._plot_3d(view=(0, 90), show_star=True)
        # plt.show()
        if verbose:
            print("k = %d/%d" % (k + 1, Nobliquity))
            print(" beta = %.3f" % tilt)

        S_check[k] = S_cells[(g.rho > 0)].sum() / (4 * np.pi) * 100

        mask = 1.0 * (g.rho * g.v[0] < 0)[0]

        S[k], dOmega = ctts_env.utils.surface_integral(
            g.grid[1], g.grid[2], mask, axi_sym=False
        )
        S[k] *= 100 / (4 * np.pi)

        g.clean_grid()  # for next point, free the grid

    # Otherwise 2 columns
    # 	S[0] *= 0.5
    # 	S_check[0] *= 0.5

    # Mahdavi and Kenyon for Rt >> Rstar
    fig, ax = plt.subplots(figsize=(12, 5))
    #     axMK = ax.twinx()
    # #     axMK.plot(
    # #         beta_ma, 100 * shock_surface_MK(rmo, beta_ma), "-k", label="Mahdavi & Kenyon"
    # #     )
    #     axMK.set_ylabel("Mahdavi & Kenyon")
    #     axMK.legend()
    ax.plot(
        beta_ma,
        0.5 * 100 * shock_surface_MK(rmo, beta_ma),
        "--k",
        label="Mahdavi & Kenyon",
    )
    ax.plot(beta_ma, 100 * kulkarni, "--", color="darkorange", label="Kulkarni 13")
    ax.plot(beta_ma, 0.5 * 100 * kulkarni, "--", color="darkorange")
    # ax.plot(beta_ma, analytical_surf, "om")
    ax.plot(
        beta_ma,
        100 * ctts_env.utils.shock_area(rmi, rmo - rmi, beta=beta_ma, f=0.5),
        "--m",
        label="analytic",
    )
    ax.plot(
        beta_ma,
        [100 * ctts_env.utils.Gamma(rmi, rmo - rmi)] * Nobliquity,
        "-",
        color="0.7",
        label="Axisymmetric",
    )
    ax.plot(
        beta_ma,
        [50 * ctts_env.utils.Gamma(rmi, rmo - rmi)] * Nobliquity,
        "--",
        color="0.7",
        label="Axisymmetric / 2",
    )
    ax.plot(beta_ma, S, ".-b", label="")
    ax.plot(beta_ma, S_check, "xb", label="")
    ax.set_ylabel("Shock area (% stellar surface)")
    ax.legend()
    ax.set_xlabel("magnetic obliquity [deg]")
    if save:
        fig.savefig("shock_area_obliquity.png")
    plt.show()

    return


##########################################################################################


if __name__ == "__main__":
    # execute test code

    P = None
    star = ctts_env.Star(2, 0.8, 4000, P, 1)

    Nt = 300
    Np = 300
    Nobliquity = 10

    # generate a grid
    rmi = 2.2
    rmo = 3

    plot_shock_surface(
        star,
        rmi=rmi,
        rmo=rmo,
        Nt=Nt,
        Np=Np,
        obliquity_limits=[0.01, 88.5],
        Nobliquity=Nobliquity,
        save=False,
        verbose=True,
    )
