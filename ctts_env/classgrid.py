from .constants import (
    Rsun,
    Msun,
    Ggrav,
    Msun_per_year_to_SI,
    day_to_sec,
    Rsun_au,
    AMU,
)
from .utils import surface_integral, spherical_to_cartesian, cartesian_to_spherical
from .temperature import logRadLoss_to_T, T_to_logRadLoss
import numpy as np
from scipy.interpolate import CubicSpline
import sys

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from matplotlib.patches import Circle
# from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm, TwoSlopeNorm


class Star:
    def __init__(self, R, M, T, P, Beq):
        self.R = R
        self.M = M
        self.T = T
        self.P = P
        self.Beq = Beq

        self.R_m = self.R * Rsun
        self.R_au = self.R * Rsun_au
        self.M_kg = self.M * Msun
        self._m0 = self.Beq * 1.0
        self.S_m2 = 4 * np.pi * self.R_m**2
        self.Rco = 100  # large init value

        self._vff = np.sqrt(self.M_kg * Ggrav * 2.0 / self.R_m)
        if self.P:
            self._veq = 2.0 * np.pi / (self.P * day_to_sec) * self.R_m
            self._omega = 2 * np.pi / (self.P * day_to_sec)
            self.Rco = (Ggrav * self.M_kg / self._omega**2) ** (
                1 / 3
            ) / self.R_m  # Rstar
        else:
            self._veq = 0.0
            self._omega = 0.0

        return

    def _pinfo(self, fout=sys.stdout):
        print("** Stellar parameters:", file=fout)
        print(" ----------------------- ", file=fout)
        print("  R = %lf Rsun; M = %lf Msun" % (self.R, self.M), file=fout)
        if self.P:
            print("  P = %lf d" % self.P, file=fout)
        print("  Beq = %lf G" % self.Beq, file=fout)
        print("  T = %lf K" % self.T, file=fout)
        print(
            "  veq = %lf km/s; vff = %lf km/s" % (self._veq * 1e-3, self._vff * 1e-3),
            file=fout,
        )
        print("", file=fout)
        return


class Grid:
    def __init__(self, r, theta, phi):
        assert type(r) == np.ndarray, " r must be a numpy array!"
        assert type(theta) == np.ndarray, " theta must be a numpy array!"
        assert type(phi) == np.ndarray, " phi must be a numpy array!"

        self.shape = r.shape
        self.Ncells = np.product(self.shape)
        # only important for interpolation
        # Interpolation is much faster on a structured, regular grid
        self.structured = r.ndim > 1

        self._2d = phi.max() == phi.min()  # only a slice phi = array([0.]*Nr*Nt)
        if self._2d and theta[0, :, 0].max() > np.pi / 2:
            print("Warning: Nphi = 1 but model is 2.5d !")
            print(" might have bugs at writing and density normalisation.")

        if self.structured:
            self.grid = (r[:, 0, 0], theta[0, :, 0], phi[0, 0, :])
        else:
            self.grid = np.array([r, theta, phi]).T

        # 		if r.ndim>1:
        # 			self.r = r.flatten()
        # 		else:
        # 			self.r = r
        # 		if theta.ndim>1:
        # 			self.theta = theta.flatten()
        # 		else:
        # 			self.theta = theta
        # 		if phi.ndim>1:
        # 			self.phi = phi.flatten()
        # 		else:
        # 			self.phi = phi
        self.r = r
        self.theta = theta
        self.phi = phi

        self._cp = np.cos(self.phi)  # cos(phi)
        self._sp = np.sin(self.phi)  # sin(phi)
        self._st = np.sin(self.theta)  # sin(theta)
        self._ct = np.cos(self.theta)  # cos(theta)
        self.x = self.r * self._st * self._cp  # Rstar
        self.y = self.r * self._st * self._sp  # Rstar
        self.z = self.r * self._ct  # Rstar
        self._sign_z = np.sign(self.z)
        self.R = self.r * self._st

        shape = [3]
        for nn in self.shape:
            shape.append(nn)
        self.v = np.zeros(shape)
        self.B = np.zeros(shape)

        self.rho = np.zeros(self.shape)
        self.T = np.zeros(self.shape)
        self.ne = np.zeros(self.shape)  # electronic density

        self.Rmax = 0

        self.regions = np.zeros(self.shape, dtype=int)
        self.regions_label = [
            "",
            "Accr. Col",
            "Disc Wind",
            "Disc",
            "Dead zone",
            "Stellar wind",
            "dark",
        ]
        self.regions_id = [0, 1, 2, 3, 4, 5, -1]
        # 0 : transparent
        # -1: dark
        # 1 : accretion columns

        self._volume_set = False

        return

    def get_B_module(self):
        return np.sqrt((self.B**2).sum(axis=0))

    def get_v_module(self):
        return np.sqrt((self.v**2).sum(axis=0))

    def get_v_cart(self):
        vx, vy, vz = spherical_to_cartesian(
            self.v[0], self.v[1], self.v[2], self._ct, self._st, self._cp, self._sp
        )
        return vx, vy, vz

    def get_v_cyl(self):
        vx, vy, vz = self.get_v_cart()
        vR = vx * self._cp + vy * self._sp
        return vR, vz, self.v[2]

    def calc_cells_volume(self, vol=[]):
        """
        3d grid's cells volume calculation
        dvolume = dS * dr
        dvolume = np.gradient(self.r,axis=0) * self.calc_cells_surface()
        """
        # if volume is set, simply return.
        if self._volume_set:
            return

        # Volume from an external grid ?
        if np.any(vol):
            self.volume = np.copy(vol)
        # Estimate the volume from this grid
        else:
            dr = np.gradient(self.r, axis=0)
            dt = np.gradient(self.theta, axis=1)
            dp = np.gradient(self.phi, axis=2)
            self.volume = self.r**2 * dr * self._st * dt * dp

        self._smoothing_length = 1 / 3 * self.volume ** (1 / 3)
        self._volume_set = True
        return

    def calc_cells_surface(self):
        """
        3d grid's cell surface calculation
        """
        dt = np.gradient(self.theta, axis=1)
        dp = np.gradient(self.phi, axis=2)
        self.surface = self.r**2 * self._st * dt * dp
        return

    def calc_cells_limits(self, rmin, rmax):
        """
        From the cell centres (self.r, self.theta, self.phi), computes the cell limits
        There is one more point in each direction.
        In 2d symmetry (no 2.5d), sin(theta) limits go from 1 to 0 (pi/2 to 0).
        in 3d, sin(theta) limits go from 1 to -1.
        """

        if self.shape[-1] > 1:
            self._p_lim = np.zeros(self.shape[-1] + 1)
        else:
            self._p_lim = np.zeros(1)
        self._r_lim = np.zeros(self.shape[0] + 1)
        jend = (self.shape[1] // 2, self.shape[1])[self._2d]
        self._sint_lim = np.zeros(self.shape[1] + 1) - 1000  # debug

        self._p_lim[-1] = 2 * np.pi
        self._p_lim[0] = 0.0

        self._r_lim[0] = rmin
        for i in range(1, self.shape[0]):
            # self._r_lim[i] = 0.5 * (self.r[i,0,0]+self.r[i-1,0,0])
            dr = self.r[i, 0, 0] - self.r[i - 1, 0, 0]
            self._r_lim[i] = self._r_lim[i - 1] + dr
        self._r_lim[self.shape[0]] = rmax

        # w = self._st[0, :, 0]
        # because theta goes to pi to 0 in general for that grid but it is
        # preferable for the limits in mcfost to have it from 1 to -1 so pi/2 to pi/2
        # Still, theta goes from pi to 0.
        w = np.sin(self.theta[0, :, 0] - (np.pi / 2, 0)[self._2d])  # [1, -1]
        self._sint_lim[0] = 1.0
        for j in range(1, jend):
            self._sint_lim[j] = 0.5 * (w[j] + w[j - 1])
            # dt = w[j] - w[j - 1]
            # self._sint_lim[j] = self._sint_lim[j - 1] + dt
        # print("0", self._sint_lim)
        self._sint_lim[jend] = 0
        # print("1", self._sint_lim)
        if not self._2d:
            self._sint_lim[jend + 1 :] = -self._sint_lim[0:jend][::-1]

        cost_lim = np.sqrt(1.0 - self._sint_lim**2)
        self._tlim = np.arcsin(self._sint_lim)  # [pi/2, -pi/2] in 3d
        # print("2", self._sint_lim)

        for k in range(1, self.shape[2]):
            self._p_lim[k] = 0.5 * (self.phi[0, 0, k] + self.phi[0, 0, k - 1])
            # dp = self.phi[0, 0, k] - self.phi[0, 0, k - 1]
            # self._p_lim[k] = self._p_lim[k - 1] + dp
        return

    def clean_grid(self, regions_to_clean=[]):
        """
        Clean an Grid instance by setting v, rho, T and Rmax to 0
        for a specific region or all if regions_to_clean is empty

        Only clean public variables.
        Private variables, belonging to specifc regions (mag,wind), for
        instance, (_Rt, _dr, _rho_axi etc...) are not cleaned. They are
        overwritten at each call of the proper method.
        """
        if not np.any(regions_to_clean):
            mask = np.ones(self.r.shape, dtype=bool)
        else:
            mask = self.regions == regions_to_clean[0]
            for ir in range(1, len(regions_to_clean)):
                mask *= self.regions == ir

        self.regions[mask] = 0
        self.v[:, mask] *= 0
        self.rho[mask] *= 0
        self.T[mask] *= 0
        self.Rmax = 0
        return

    def _check_overlap(self):
        """
        *** building ***
        Check that a region is not already filled with density
        in that case, avoid it or set the density to 0 ?
        """
        return

    def add_disc(
        self, star, Rin, Md, p, beta, H0, r0=1, wall=False, phi0=0, Rwi=1, Aw=1
    ):
        """
        Dust and gas disc.
        The temperature of the gas in the disc follows the dust temperature.
        Still, below T_molec all the atoms are bound to a molecule and atomic densities
        are zero (not atomic transfer).

        Rin :: inner edge of the disc (outer edge is fixed by the grid)
        Md  :: disc's mass in Mstar (not Msun)
        p   :: exponent
        beta    :: exponent
        H0  :: disc scale height in Rstar
        r0  :: reference radius for the scale height in Rstar

        wall    :: add a wall to the disc ?
        phi0    :: origin of the wall in the azimuthal plane (max at phi0)
        Rwi :: inner radius of the wall (where it starts)
        Aw  :: width of the wall (in the z direction)
        """

        # TO DO define a mask to identify a region corresponding to the  disc
        # because at the moment it expands on the whole volume
        # with potential overlap with other regions
        midplane = np.argmin((self.theta[0, :, 0] - np.pi / 2) ** 2)
        Rout = self.R.max()

        K = star.M_kg * Md / (2 * np.pi) ** 1.5 / H0 / r0**2 / star.R_m**3
        if p == -2:
            K2 = 1.0 / np.log(Rout / Rin)
        else:
            K2 = ((p + 2) * r0 ** (p + 2)) / (Rout ** (p + 2) - Rin ** (p + 2))

        K3_exp = 2 * (H0 * (self.R / r0) ** beta) ** 2
        K3 = (self.R / r0) ** (p - beta) * np.exp(-(self.z**2) / K3_exp)

        rho = K * K2 * K3
        rho[self.R <= Rin] = 0.0
        # mask for the disc ? has a function of scale height ?
        # for each r the disc is where z < eps * H ?
        self.rho = rho

        dwidth = 0
        if (wall) and (not self._2d):
            # midplane from the grid resolution ?
            # zmin = dwidth + np.amin(abs(self.z), axis=1)
            zmin = 0.0
            dwall = abs(
                Rwi
                - self.R[
                    np.argmin((self.R[:, midplane, 0] - Rwi) ** 2) + 1, midplane, 0
                ]
            )
            dwall = Rwi * 0.1
            north = (1.0 + np.cos(self.phi + phi0)) / 2.0
            sud = (1.0 + np.cos(self.phi + np.pi + phi0)) / 2.0
            wall_mask = (self.R <= Rwi + dwall) * (
                self.z <= zmin[:, None, :] + Aw * north
            ) * (self.z >= 0) | (self.z >= -zmin[:, None, :] - Aw * sud) * (self.z < 0)
            mask = (self.R > Rwi) * wall_mask
            Rwo = Rwi + dwall
            if p == -2:
                K2_wall = 1.0 / np.log(Rwo / Rwi)
            else:
                K2_wall = ((p + 2) * r0 ** (p + 2)) / (Rwo ** (p + 2) - Rwi ** (p + 2))
            self.rho[wall_mask] = self.rho[self.rho > 0].min()  # K * K2_wall * K3[mask]

        self.regions[self.rho > 0] = 2
        # accreting velocity ??
        # Keplerian velocity
        self.v[2] = np.sqrt(Ggrav * star.M_kg * self.R**2 / self.r**3 / star.R_m)
        return

    def add_dark_disc(self, Rin, dwidth=0, Td=0, wall=False, phi0=0, Rwi=1, Aw=1, Tw=0):
        """
        Optically thick and ultra-cool disc.
        rho value does not matter (dark). Temperature
        could be use for an emission component.

        Rin :: inner radius at which the disc starts
        dwidth  :: the constant width (in the z direction) of the disc
        Td  :: (constant) Temperature of the disc
        wall    :: add a wall to the disc ?
        phi0    :: origin of the wall in the azimuthal plane (max at phi0)
        Rwi :: inner radius of the wall (where it starts)
        Aw  :: width of the wall (in the z direction)
        Td  :: (constant) Temperature of the wall
        """
        # zmin = dwidth + np.amin(abs(self.z), axis=(1, 2))
        # mask = (self.R > Rin) * (abs(self.z) <= zmin[:, None, None])
        zmin = dwidth + np.amin(abs(self.z), axis=1)
        mask = (self.R > Rin) * (abs(self.z) <= zmin[:, None, :])
        self.regions[mask] = -1
        self.rho[mask] = 1e-5  # kg/m3
        self.T[mask] = Td
        # add wall after disc
        if (wall) and (not self._2d):
            midplane = np.argmin((self.theta[0, :, 0] - np.pi / 2) ** 2)
            dwall = abs(
                Rwi
                - self.R[
                    np.argmin((self.R[:, midplane, 0] - Rwi) ** 2) + 1, midplane, 0
                ]
            )
            north = (1.0 + np.cos(self.phi + phi0)) / 2.0
            sud = (1.0 + np.cos(self.phi + np.pi + phi0)) / 2.0
            wall_mask = (self.R <= Rwi + dwall) * (
                self.z <= zmin[:, None, :] + Aw * north
            ) * (self.z >= 0) | (self.z >= -zmin[:, None, :] - Aw * sud) * (self.z < 0)
            mask = (self.R > Rwi) * wall_mask
            self.T[wall_mask] = Tw
            # wall following accretion column building
            # zmax = Aw
            # rmax = np.sqrt(Aw ** 2 + self.R[self._lmag].max() ** 2)
            # rM = rmax ** 3 / (self._xp ** 2 + self._yp ** 2)
            # col_mask = (rM > Rwi) * (rM <= Rwi + dwall)
            # north = col_mask * self.z > 0
            # south = col_mask * self.z < 0
            # mask = self.z <= zmin[:, None, :] + Aw * north
            # mask = (self.z <= zmin[:, None, :] + Aw * north) * (
            #     self.z >= -zmin[:, None, :] - Aw * south
            # )
            self.regions[mask] = -1
            self.rho[mask] = 1e-5
        return

    def add_mag(
        self,
        star,
        rmi=2.2,
        rmo=3.0,
        Mdot=1e-8,
        beta=0.0,
        Tmax=8000,
        verbose=False,
        V0=0,
        lsec=False,
    ):
        """
        star    :: An instance of the class Star

        rmi     :: inner radius of the magnetosphere (Rstar)
        rmo     :: outer radius of the magnetosphere (Rstar)
                    rmo must be lower than (2 / (2 + cos(beta) ** 2)) ** (1 / 3) Rco
        Mdot    :: mass accretion rate (Msun/yr)
        beta    :: obliquity of the magnetic dipole (degrees). Beta must be > 0 and < 90 at the moment.
                                The magnetic field is tilted about the rotation axis (// z) of the star. The tilted
                                dipole is // to x axis.
        verbose :: print info if True
        Tmax    :: value of the temperature maximum in the magnetosphere
        V0      :: value of the velocity at the injection point (m/s)
        lsec    :: keep the secondary columns that have positive and negative v^2 along the field line.
        """
        self._beta = beta
        ma = np.deg2rad(self._beta)
        self.Rmax = rmo

        Rt_on_Rco = (2 / (2 + np.cos(ma) ** 2)) ** (1 / 3)
        if rmo > Rt_on_Rco * star.Rco:
            print("(ERROR) Outer truncation radius cannot be larger than alpha * Rco !")
            print(
                "rmo = %.3f R*; alpha = %.3f; Rco = %.3f R*"
                % (rmo, Rt_on_Rco, star.Rco)
            )
            exit()

        if self._beta != 0 and self._2d:
            print(
                "(add_magnetosphere) WARNING : Using a 2d grid for a non-axisymmetric model!"
            )

        self._Macc = Mdot * Msun_per_year_to_SI
        self._Rt = rmi
        self._dr = rmo - rmi

        # coordinates tilted about z, in F'
        self._xp = self.r * (self._cp * self._st * np.cos(ma) - self._ct * np.sin(ma))
        self._yp = self.r * (self._sp * self._st)
        self._zp = self.r * (self._cp * self._st * np.sin(ma) + self._ct * np.cos(ma))
        Rp = np.sqrt(self._xp**2 + self._yp**2)

        cpp = self._xp / Rp
        spp = self._yp / Rp
        ctp = self._zp / self.r
        stp = Rp / self.r

        # tanphi0 = (
        #     np.cos(ma)
        #     * self._st
        #     * self._sp
        #     / (np.cos(ma) * self._st * self._cp - np.sin(ma) * self._ct)
        # )
        # phi0 = np.arctan(tanphi0)
        # sinphi0 = np.sin(phi0)
        # r0 = (self.r / self._st ** 2) * (sinphi0 ** 2 / self._sp ** 2)
        # r0 = (
        #     self.r
        #     * np.cos(ma) ** 2
        #     / (
        #         np.cos(ma) ** 2 * self._st ** 2
        #         + np.sin(ma) ** 2 * self._ct ** 2
        #         - 2 * np.cos(ma) * np.sin(ma) * self._st * self._ct * self._cp
        #     )
        # )
        # v_square = (
        #     2 * Ggrav * star.M_kg / star.R_m * (1 / self.r - 1 / r0)
        #     + (self.R ** 2 - r0 ** 2) * (star.R_m * star._omega) ** 2
        #     + V0 ** 2
        # )
        ##### TMP #####
        # ##self._laccr *= cpp * self.z >= 0
        ###############
        self._ldead_zone = np.zeros(self.shape)
        self._v2_dead_zone = np.zeros(self.shape)
        # to test
        # check the points for which the field line passing by these points accrete
        v_square = np.zeros(self.shape)
        # if the number of point is not enough, the field line
        # might not be resolve leading to unconsistent results
        # like positive while some points are negative
        N_fl = 10000
        # record the valid values of R0
        # R0 = np.zeros(self.shape)
        self._lpropeller = np.zeros(self.shape, dtype=bool)
        self._lsecondary_col = np.zeros(self.shape, dtype=bool)
        for k in range(self.shape[2]):
            for j in range(self.shape[1]):
                for i in range(self.shape[0]):
                    tanphi0 = (
                        np.cos(ma)
                        * self._st[i, j, k]
                        * self._sp[i, j, k]
                        / (
                            np.cos(ma) * self._st[i, j, k] * self._cp[i, j, k]
                            - np.sin(ma) * self._ct[i, j, k]
                        )
                    )
                    phi0 = np.arctan(tanphi0)
                    if self._beta == 0:  # avoid 0 division error
                        r0 = self.r[i, j, k] / self._st[i, j, k] ** 2
                        ts = np.arcsin(np.sqrt(self.r[i, j, k] / r0))
                    else:  # non-zero obliquity
                        r0 = (
                            self.r[i, j, k]
                            * np.sin(phi0) ** 2
                            / self._st[i, j, k] ** 2
                            / self._sp[i, j, k] ** 2
                        )
                        ts = np.arcsin(
                            np.sqrt(
                                self.r[i, j, k]
                                / r0
                                * np.sin(phi0) ** 2
                                / self._sp[i, j, k] ** 2
                            )
                        )
                    if r0 >= rmi and r0 <= rmo:
                        # R0[i, j, k] = r0
                        # ts = np.arcsin(
                        #     np.sqrt(
                        #         self.r[i, j, k]
                        #         / r0
                        #         * np.sin(phi0) ** 2
                        #         / self._sp[i, j, k] ** 2
                        #     )
                        # )
                        te = np.pi / 2  # * (0.99, 1)[beta == 0]
                        # if self.z[i, j, k] < 0:
                        #     ts = np.pi - ts
                        t_fl = np.linspace(ts, te, N_fl)

                        # build the field line the point (i,j,k) belongs to, from the disc to the stellar surface
                        y_fl = np.sin(t_fl) ** 2
                        if self._beta == 0:
                            r_fl = r0 * y_fl
                        else:  # non-zero obliquity
                            r_fl = (
                                r0 * self._sp[i, j, k] ** 2 / np.sin(phi0) ** 2 * y_fl
                            )
                        v2_fl = (
                            2 * Ggrav * star.M_kg / star.R_m * (1 / r_fl - 1 / r0)
                            + (y_fl * r_fl**2 - r0**2)
                            * (star.R_m * star._omega) ** 2
                            + V0**2
                        )
                        # compute invariant # TO DO!
                        # need B field because need v
                        # e_minus_lomegastar = self._calc_invariant()

                        if np.alltrue(v2_fl > 0):
                            v_square[i, j, k] = (
                                2
                                * Ggrav
                                * star.M_kg
                                / star.R_m
                                * (1 / self.r[i, j, k] - 1 / r0)
                                + (self.R[i, j, k] ** 2 - r0**2)
                                * (star.R_m * star._omega) ** 2
                                + V0**2
                            )
                        elif lsec:  # part of the gas along the field line is ejecting.
                            v_square[i, j, k] = (
                                2
                                * Ggrav
                                * star.M_kg
                                / star.R_m
                                * (1 / self.r[i, j, k] - 1 / r0)
                                + (self.R[i, j, k] ** 2 - r0**2)
                                * (star.R_m * star._omega) ** 2
                                + V0**2
                            )
                            self._lsecondary_col[i, j, k] = True
                            # default False
                            if v_square[i, j, k] < 0:
                                self._lpropeller[i, j, k] = True
                                v_square[i, j, k] *= -1.0
                    elif r0 < rmi:
                        self._ldead_zone[i, j, k] = 1
                        self._v2_dead_zone[i, j, k] = abs(
                            2
                            * Ggrav
                            * star.M_kg
                            / star.R_m
                            * (1 / self.r[i, j, k] - 1 / r0)
                            + (self.R[i, j, k] ** 2 - r0**2)
                            * (star.R_m * star._omega) ** 2
                            + V0**2
                        )
        #############################################################################

        # ###self._laccr = (v_square >= 0) * (r0 >= rmi) * (r0 <= rmo)
        self._laccr = v_square > 0
        self.regions[self._laccr] = 1  # non-transparent regions.

        # smaller arrays, only where accretion takes place
        # m is the magnetic moment at the pole and at r=1.
        # - because vr must be negative around the pole. x2 because _m0 is at the equator.
        m = -2.0 * star._m0 / self.r[self._laccr] ** 3
        self.B = np.zeros(self.v.shape)
        # (Br, Btheta, Bphi)
        self.B[0, self._laccr] = (
            m * (self._st * self._cp * np.sin(ma) + self._ct * np.cos(ma))[self._laccr]
        )
        self.B[1, self._laccr] = (
            -m
            / 2
            * (self._ct * self._cp * np.sin(ma) - self._st * np.cos(ma))[self._laccr]
        )
        self.B[2, self._laccr] = m / 2 * (self._sp * np.sin(ma))[self._laccr]
        B = self.get_B_module()

        sig_z = self._sign_z[self._laccr]
        # useful to change the sign of vr for propeller region or already in B ?
        # _psign = (
        #     -2 * self._lpropeller[self._laccr] + 1
        # )  # -1 if ejecting, 1 otherwise (accreting)
        v = np.sqrt(v_square[self._laccr])

        vr = v * self.B[0, self._laccr] / B[self._laccr] * sig_z
        vt = v * self.B[1, self._laccr] / B[self._laccr] * sig_z
        vp = v * self.B[2, self._laccr] / B[self._laccr] * sig_z  # why sig_z here ?
        u_phi = star._veq * self.R[self._laccr] + vp
        self.v[0, self._laccr] = vr
        self.v[1, self._laccr] = vt
        self.v[2, self._laccr] = u_phi
        # v = np.sqrt(vr * vr + vt * vt + vp * vp)

        # Compute the inveriant e - lOmega* (V0 not included!)
        # self._invariant_part1 = 0.5 * (vr * vr + vt * vt + u_phi * u_phi)  # u^2 / 2
        # self._invariant_part2 = (
        #     -Ggrav * star.M_kg / (self.r[self._laccr] * star.R_m)
        # )  # -GM/R
        # self._invariant_part3 = (
        #     -(star.R_m * self.R[self._laccr]) * star._omega * self.v[2, self._laccr]
        # )  # -r u_phi * Omega*
        # self._invariant_part4 = (
        #     -Ggrav * star.M_kg / (star.R_m * R0[self._laccr])
        # )  # -GM/R0
        # self._invariant_part5 = (
        #     -0.5 * (star.R_m * R0[self._laccr]) ** 2 * star._omega ** 2
        # )  # 1/2 R0^2 * Omega*^2
        # self._invariant = (
        #     self._invariant_part1
        #     + self._invariant_part2
        #     + self._invariant_part3
        #     - self._invariant_part4
        #     - self._invariant_part5
        # )  # = 0

        # TO DO: define a non-constant eta
        eta = 1.0  # mass-to-magnetic flux ratio, set numerically

        self.rho[self._laccr] = eta * B[self._laccr] / v
        # normalisation of the density
        if self.structured:
            # takes values at the stellar surface or at rmin.
            # multiply mass_flux by rmin**2 ?
            rhovr = self.rho[0] * self.v[0, 0] * (self.regions[0] == 1)
            # integrate over the shock area
            # mass_flux in units of rhovr
            mass_flux, dOmega = surface_integral(
                self.grid[1], self.grid[2], -rhovr, axi_sym=self._2d
            )
            # similar to
            # mf = (0.5*(-rhovr[0,1:,1:] - rhovr[0,:-1,:-1]) * abs(ct[:,:-1]) * dp[1:,:]).sum()
            # with ct = np.diff(self._ct[0],axis=0); dp = np.diff(self.phi[0],axis=1)
            if verbose:
                print("dOmega = %.4f" % (dOmega))
                print("mass flux (before norm) = %.4e [v_r B/v]" % mass_flux)
            eta = self._Macc / mass_flux / star.R_m**2
        else:
            print("Error unstructured grid not yet")
        self.rho[self._laccr] *= eta
        # shock area
        self._f_shock = surface_integral(
            self.grid[1], self.grid[2], 1.0 * (rhovr < 0), axi_sym=self._2d
        )[0] / (4 * np.pi)
        if verbose:
            print(
                "The shock covers a fraction  %.3f %s of the stellar surface"
                % (self._f_shock * 100, "%")
            )

        # recompute mass flux after normalisation
        mass_flux_check = (
            surface_integral(
                self.grid[1], self.grid[2], -rhovr * eta, axi_sym=self._2d
            )[0]
            * star.R_m**2
        )
        if verbose:
            print(
                "Mass flux (after norm) = %.4e Msun.yr^-1"
                % (mass_flux_check / Msun_per_year_to_SI)
            )
            print("(check) Mdot/Mdot_input = %.3f" % (mass_flux_check / self._Macc))
        if abs(mass_flux_check / self._Macc - 1.0) > 1e-5:
            print(mass_flux_check, self._Macc)
            print(
                "WARNING : problem of normalisation of mass flux in self.add_magnetosphere()."
            )

        # Computes the temperature of the form Lambda_cool = Qheat / nH^2
        Q = B[self._laccr]
        rl = Q * self.rho[self._laccr] ** -2
        lgLambda = np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax)
        if lsec:  # force similar T independently of density constrast
            self.T[self._laccr] = 0.0
            lgLambda = -1 + np.zeros(self.shape)
            rl = B[self._lsecondary_col] * self.rho[self._lsecondary_col] ** -2
            lgLambda[self._lsecondary_col] = np.log10(rl / rl.max()) + T_to_logRadLoss(
                Tmax
            )
            cond = (~self._lsecondary_col) * self._laccr
            rl = B[cond] * self.rho[cond] ** -2
            lgLambda[cond] = np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax)
            lgLambda = lgLambda[self._laccr]
        self.T[self._laccr] = logRadLoss_to_T(lgLambda)

        return

    def setup_dead_zone(self, star, rho, T):
        """
        ** building **
        The density and the temperature are assumed to be
        constant and given by rho and T, respectively.

        The dead zone is in solid body rotation only.

        """
        try:
            b = np.deg2rad(self._beta)
        except:
            print("Cannot add a dead zone if no accreting magnetosphere present!")
            exit()
            return

        ldz = self._ldead_zone > 0
        # v = np.sqrt(self._v2_dead_zone[ldz])
        # sig_z = self._sign_z[ldz]
        # m = -2.0 * star._m0 / self.r[ldz] ** 3
        # br = m * (self._st * self._cp * np.sin(b) + self._ct * np.cos(b))[ldz]
        # bt = -m / 2 * (self._ct * self._cp * np.sin(b) - self._st * np.cos(b))[ldz]
        # bphi = m / 2 * (self._sp * np.sin(b))[ldz]
        # Bmag = np.sqrt(br**2 + bt**2 + bphi**2)
        # self.v[0, ldz] = v * br / Bmag * sig_z
        # self.v[1, ldz] = v * bt / Bmag * sig_z
        # self.v[2, ldz] = v * bphi / Bmag * sig_z + star._veq * self.R[ldz]
        self.v[2, ldz] = star._veq * self.R[ldz]

        self.regions[ldz] = 4
        self.rho[ldz] = rho
        self.T[ldz] = T

        return

    def add_magnetosphere_v1(
        self,
        star,
        rmi=2.2,
        rmo=3.0,
        Mdot=1e-8,
        beta=0.0,
        Tmax=8000,
        verbose=False,
        no_sec=True,
    ):
        """
        star 	:: An instance of the class Star

        rmi 	:: inner radius of the magnetosphere (Rstar)
        rmo  	:: outer radius of the magnetosphere (Rstar)
        Mdot 	:: mass accretion rate (Msun/yr)
        beta 	:: obliquity of the magnetic dipole (degrees). Beta must be > 0 and < 90 at the moment.
                                The magnetic field is tilted about the rotation axis (// z) of the star. The tilted
                                dipole is // to x axis.

        verbose :: print info if True
        no_sec 	:: flag to remove secondary columns

        NOTE: old version working but not totally fully consistent.
              here for debug and comparisons with old models.

        """

        self._Rt = rmi
        self._dr = rmo - rmi
        self._beta = beta
        self._Macc = Mdot * Msun_per_year_to_SI
        self._no_sec = no_sec

        if self._beta != 0 and self._2d:
            print(
                "(add_magnetosphere) WARNING : Using a 2d grid for a non-axisymmetric model!"
            )

        ma = np.deg2rad(self._beta)
        self.Rmax = max(self.Rmax, rmo * (1.0 + np.tan(ma) ** 2))

        # Constant for density in axisymmetric cases
        m0 = (
            (self._Macc * star.R_m)
            / ((1.0 / rmi - 1.0 / rmo) * 4.0 * np.pi)
            / np.sqrt(2.0 * Ggrav * star.M_kg)
        )

        # coordinates tilted about z, in F'
        self._xp = self.r * (self._cp * self._st * np.cos(ma) - self._ct * np.sin(ma))
        self._yp = self.r * (self._sp * self._st)
        self._zp = self.r * (self._cp * self._st * np.sin(ma) + self._ct * np.cos(ma))
        Rp = np.sqrt(self._xp**2 + self._yp**2)  # + tiny_val

        cpp = self._xp / Rp
        spp = self._yp / Rp
        ctp = self._zp / self.r
        stp = Rp / self.r  # np.sqrt(1.0 - ctp ** 2)

        sintheta0p_sq = (1.0 + np.tan(ma) ** 2 * cpp**2) ** -1  # sin(theta0')**2
        yp = stp**2
        # In the Frame of the disc (i.e., not tilted)
        y = self._st**2  # Note: y is 0 if theta = 0 +- pi
        dtheta = self.grid[1][1] - self.grid[1][0]
        y[self.theta % np.pi == 0.0] = np.sin(dtheta) ** 2
        rM = self.r / y
        rMp = self.r / yp
        rlim = rMp * sintheta0p_sq

        # should not be negative in the accretion columns.
        fact = (1.0 / self.r - 1.0 / rM) ** 0.5

        # condition for accreting field lines
        # -> Axisymmetric case #
        lmag_axi = (rM >= rmi) * (rM <= rmo)
        self._rho_axi = np.zeros(self.shape)
        self._rho_axi[lmag_axi] = (  # not normalised to Mdot
            m0
            * (star.R_m * self.r[lmag_axi]) ** (-5.0 / 2.0)
            * np.sqrt(4.0 - 3 * y[lmag_axi])
            / np.sqrt(1.0 - y[lmag_axi])
        )
        # TO DO: norm
        #######################

        # condition for accreting field lines
        lmag = (rlim >= rmi) * (rlim <= rmo)
        self._lmag = lmag

        # Secondary and main columns
        mcol = (cpp * self.z >= 0.0) * lmag
        self._mcol = np.zeros(self.shape, dtype=bool)
        self._scol = np.zeros(self._mcol.shape, dtype=bool)
        self._mcol[mcol] = True  # main columns
        self._scol = ~self._mcol  # secondary columns
        if no_sec:
            self._lmag *= self._mcol
            lmag *= self._mcol

        self.regions[lmag] = 1  # non-transparent regions.

        # smaller arrays, only where accretion takes place
        m = star._m0 / self.r[lmag] ** 3  # magnetic moment at r
        self.B = np.zeros(self.v.shape)
        self.B[0, lmag] = (
            2.0
            * m
            * (
                np.cos(ma) * self._ct[lmag]
                + np.sin(ma) * self._cp[lmag] * self._st[lmag]
            )
        )
        self.B[1, lmag] = m * (
            np.cos(ma) * self._st[lmag] - np.sin(ma) * self._cp[lmag] * self._ct[lmag]
        )
        self.B[2, lmag] = m * np.sin(ma) * self._sp[lmag]
        B = self.get_B_module()

        sig_z = self._sign_z[lmag]

        vpol = star._vff * fact[lmag]
        vtor = vpol * self.B[2, lmag] / B[lmag]

        vr = -vpol * self.B[0, lmag] / B[lmag] * sig_z
        vt = -vpol * self.B[1, lmag] / B[lmag] * sig_z
        self.v[0, lmag] = vr
        self.v[1, lmag] = vt
        self.v[2, lmag] = vtor

        V = self.get_v_module()
        self.rho[lmag] = B[lmag] / V[lmag]
        # normalisation of the density
        if self.structured:
            # takes values at the stellar surface or at rmin.
            # multiply mass_flux by rmin**2 ?
            rhovr = self.rho[0] * self.v[0, 0] * (self.regions[0] == 1)
            # integrate over the shock area
            # mass_flux in units of rhovr
            mass_flux, dOmega = surface_integral(
                self.grid[1], self.grid[2], -rhovr, axi_sym=self._2d
            )
            # similar to
            # mf = (0.5*(-rhovr[0,1:,1:] - rhovr[0,:-1,:-1]) * abs(ct[:,:-1]) * dp[1:,:]).sum()
            # with ct = np.diff(self._ct[0],axis=0); dp = np.diff(self.phi[0],axis=1)
            if verbose:
                print("dOmega = %.4f" % (dOmega))
                print("mass flux (before norm) = %.4e [v_r B/V]" % mass_flux)
            rho0 = self._Macc / mass_flux / star.R_m**2
        else:
            print("Error unstructured grid not yet")

        self.rho[lmag] *= rho0
        vrot = self.r[lmag] * np.sqrt(y[lmag]) * star._veq
        self.v[2, lmag] += vrot

        # recompute mass flux after normalisation
        mass_flux_check = (
            surface_integral(
                self.grid[1], self.grid[2], -rhovr * rho0, axi_sym=self._2d
            )[0]
            * star.R_m**2
        )
        if verbose:
            print(
                "Mass flux (after norm) = %.4e Msun.yr^-1"
                % (mass_flux_check / Msun_per_year_to_SI)
            )
            print("(check) Mdot/Mdot_input = %.3f" % (mass_flux_check / self._Macc))
        if abs(mass_flux_check / self._Macc - 1.0) > 1e-5:
            print(mass_flux_check, self._Macc)
            print(
                "WARNING : problem of normalisation of mass flux in self.add_magnetosphere()."
            )
        self._f_shock = surface_integral(
            self.grid[1], self.grid[2], 1.0 * (rhovr < 0), axi_sym=self._2d
        )[0]

        # Computes the temperature of the form Lambda_cool = Qheat / nH^2
        Q = B[lmag]
        # Q = self.r[lmag] ** -3
        rl = Q * self.rho[lmag] ** -2
        lgLambda = np.log10(rl / rl.max()) + T_to_logRadLoss(Tmax)
        self.T[lmag] = logRadLoss_to_T(lgLambda)

        # In case we keep secondary columns (no_sec = False)
        # The temperature is normalised so that in average Tavg = Tmax.
        # Otherwise, the maximum of T is in the secondary columns.
        if not no_sec and self._beta != 0.0:  # only if the model is not axisymmetric
            Tavg = np.average(self.T[lmag], weights=self.rho[lmag])
            self.T[lmag] *= Tmax / Tavg
            print("Tmax (after norm to <T>) = %lf K" % self.T[lmag].max())
            print("  <T> = %lf K" % np.average(self.T[lmag], weights=self.rho[lmag]))

        return

    def add_disc_wind_knigge95(
        self,
        star,
        Rin=5,
        Rout=50,
        Mloss=1e-8,
        alpha=0.5,
        gamma=-0.5,
        ls=50,
        zs=10,
        beta=0.5,
        fesc=2,
        Tmax=10000,
        Td_in=2000.0,
        z_limit=0,
        beta_temp=1,
        scale_as_zoR0=True,
        z_cutoff=True,
    ):
        """
        Knigge et al. 1995, MNRAS 273, 225
        see also,
        Kurosawa et al. 2011, MNRAS 416, 2623

        Mloss       :: mass ejection rate in Msun/yr
        gamma       :: temperature exponant such that T \propto R**gamma (q in Kurosawa's paper). gamma < 0
        alpha       :: mass loss rate power law per unit area (alpha > 0)
        ls          :: disc wind length scale-to-Rin ratio in unit of Rin (Rs in Kurosawa's).
        zs          :: location above or below the midplane at R=0 where the field lines diverge (Source location, d in Kurosawa's).
        beta        :: exponent of the radial velocity of the wind (acceleration parameter)
        fesc        :: terminal velocity of the disc wind in unit of the escape velocity
        Tmax        :: Temperature max of the disc wind
        Td_in       :: temperature of the inner rim of the disc in z=0.
        beta_temp   :: temperature exponent for the disc wind. Isothermal if 0
        z_limit     :: the wind temperature grows from the midplane to z_limit from T0
                       to Tmax. The law is \propto (Tmax-T0) * (abs(z)/z_limit)**beta_temp + T0
        scale_as_zoR0:: if True, the temperature reaches its maximum in z/R0 = z_limit.
                        Otherwise, for z = z_limit.
        z_cutoff     :: (bool) if True the wind starts at abs(z) > z_limit (default 0 == midplane).
        """
        Td_min = 100  # K, minimum temperature allowed in the disc
        ## condition to be in the disc wind region ##
        ldw = (self.R >= Rin * (abs(self.z) + zs) / zs) * (
            self.R <= Rout * (abs(self.z) + zs) / zs
        )
        if z_cutoff:
            ldw *= abs(self.z) >= z_limit
        # -> special condition with a cut-off in z
        # ldw = (
        #     (self.R >= Rin * (abs(self.z) + zs) / zs)
        #     * (self.R <= Rout * (abs(self.z) + zs) / zs)
        #     * (abs(self.z) >= z_limit)
        # )  #            * (abs(self.z) / self.R >= z_limit)
        self.regions[ldw] = 2
        ## disc wind length scale ##
        Rs = ls * Rin
        Mloss_SI = Mloss * Msun_per_year_to_SI
        self._Mloss_dw = Mloss

        ## mass-loss normalisation ##
        # the local mass-loss rate (mdot) is proportional to the midplane temperature
        # as mdot \propto T(R)**(4*alpha). The midplane temperature itself, beeing
        # proportional R**gamma. therefore, mdot \propto R**p with p = 4 * alpha * gamma
        p_ml = 4.0 * gamma * alpha
        if p_ml > 0:
            print("(ERROR) p_ml must be negative!")
            exit()

        # mloss_surf in kg/s/m2 prop to integral over RdR of R^p_ml to check
        # here it is the inverse of the integral R^(p_ml+1)dR
        if p_ml == -2:
            fact = 1.0 / abs(np.log(Rout) - np.log(Rin))
        else:
            fact = (p_ml + 2) / (
                (Rout ** (p_ml + 2) - Rin ** (p_ml + 2)) * star.R_m ** (p_ml + 2)
            )  # m^-(p_ml + 2)
        norm_mloss = Mloss_SI * fact  # in kg/s/m^(p_ml + 2)
        # mass-loss on the disc surface
        mloss_loc = (
            norm_mloss * (star.R_m * self.R[ldw]) ** p_ml / (4 * np.pi)
        )  # kg/s/m^2 : norm_mloss in kg/s/m^(p_ml+2)--> m^p_ml * m^(-p_ml - 2) = m^-2

        ## temperature of the disc ##
        Tdisc = np.maximum(Td_in * (self.R[ldw] / Rin) ** gamma, Td_min)
        sound_speed_disc = 1e4 * np.sqrt(Tdisc * 1e-4)  # m/s

        ## velocities ##
        # the escape velocity is star._vff
        # for each R found the corresponding wi i.e., R for z=0
        wi = zs / (abs(self.z[ldw]) + zs) * self.R[ldw]
        # sqrt(G * M / wi_in_m), _vff is at the stellar surface in m.
        vkep = (
            star._vff / np.sqrt(2.0) / np.sqrt(wi)
        )  # keplerian velocity express from escape velocity
        vphi = vkep * (wi / self.R[ldw])  # angular momentum conservation along z

        # distance from the source point where the field lines diverge
        q = np.sqrt(self.R**2 + (abs(self.z) + zs) ** 2)[ldw]
        cos_delta = (abs(self.z)[ldw] + zs) / q
        l = q - zs / cos_delta
        vesc = star._vff / np.sqrt(self.R[ldw])
        cs = sound_speed_disc  # 1e4 * (Rin / wi) ** 0.5  # m/s
        vq = cs + (fesc * vesc - cs) * (1.0 - Rs / (l + Rs)) ** beta

        # beta for each field lines, such that at z_limit, vq = 200 km/s
        # r0 = np.sqrt((q - l) ** 2 - zs**2)
        # l0 = np.sqrt(r0**2 + (z_limit + zs) ** 2) * (1.0 - zs / (z_limit + zs))
        # y = Rs / (Rs + l0)
        # beta_R0 = np.log((200e3 - cs) / (fesc * vesc - cs)) / np.log(1 - y)
        # print(beta_R0)
        # print(cs + (fesc * vesc - cs) * (1.0 - Rs / (l0 + Rs)) ** beta_R0)
        # vq = cs + (fesc * vesc - cs) * (1.0 - Rs / (l + Rs)) ** beta_R0

        ########################################################################
        # needed because oorigin in z shifted by zs #
        rp = np.sqrt(
            self.x[ldw] ** 2
            + self.y[ldw] ** 2
            + (self.z[ldw] + np.sign(self.z[ldw]) * zs) ** 2
        )
        tdw = np.arccos((np.abs(self.z[ldw]) + zs) / rp)
        pdw = self.phi[ldw]
        vx = vq * np.sin(tdw) * np.cos(pdw) - vphi * np.sin(pdw)
        vy = vq * np.sin(tdw) * np.sin(pdw) + vphi * np.cos(pdw)
        vz = np.sign(self.z[ldw]) * vq * np.cos(tdw)
        self.v[0, ldw], self.v[1, ldw], self.v[2, ldw] = cartesian_to_spherical(
            vx,
            vy,
            vz,
            self._ct[ldw],
            self._st[ldw],
            self._cp[ldw],
            self._sp[ldw],
        )
        ########################################################################

        ## density ##
        rho_dw = mloss_loc / (vq * cos_delta) * (zs / (q * cos_delta)) ** 2  # kg/m3
        self.rho[ldw] = rho_dw

        ## temperature ##
        self.T[ldw] = Tmax
        if z_cutoff:
            return

        zz0 = z_limit  # / np.sqrt((q - l) ** 2 - zs**2)
        if scale_as_zoR0:
            zz = self.z[ldw] / np.sqrt((q - l) ** 2 - zs**2)  # / self.R[ldw]
        # print("R0=",np.sqrt((q - l) ** 2 - zs**2))
        # z_limit / R0
        else:
            zz = self.z[ldw]
        tt = np.minimum((Tmax - Tdisc) * (abs(zz) / zz0) ** beta_temp + Tdisc, Tmax)
        self.T[ldw] = tt

        return

    def add_disc_wind_BP82(self, star):
        """
        Disc wind model of Blandford & Payne 1982
        see: Milliner et al. 2019
        """

        return

    def add_disc_wind(
        self,
        star,
        Rin=5,
        Rout=10,
        Macc=1e-8,
        Tmax=10000.0,
        wind_model="sol40.dat",
        z_limit=0,
    ):
        """
        Descriptor to do.
        """
        self._Rwind_in = Rin * star.R_m
        self._Rwind_out = Rout * star.R_m

        Macc_SI = Macc * Msun_per_year_to_SI

        fw = open(wind_model, "r")
        fw.readline()
        xi = float(fw.readline().strip().split()[1])
        fw.close()

        quant = [
            "y",
            "theta",
            "r/r_0",
            "n_MHD",
            "u_r",
            "u_phi",
            "u_z",
            "T_MHD",
            "B_r",
            "B_phi",
            "B_z",
            "T_dyn",
        ]

        pure_data = np.transpose(
            np.genfromtxt(wind_model, skip_header=17, dtype="float")
        )
        labeled_data = {quant[n]: pure_data[n, :] for n in range(len(pure_data[:, 0]))}

        interpzFunc_in = CubicSpline(
            labeled_data["y"] * labeled_data["r/r_0"] * Rin * star.R_au,
            labeled_data["r/r_0"] * Rin * star.R_au,
        )
        interpzFunc_out = CubicSpline(
            labeled_data["y"] * labeled_data["r/r_0"] * Rout * star.R_au,
            labeled_data["r/r_0"] * Rout * star.R_au,
        )

        y = np.abs(np.divide(self.z, self.R))

        R_core = interpzFunc_in(np.abs(self.z) * star.R_au)
        R_jet = interpzFunc_out(np.abs(self.z) * star.R_au)

        sub_alfvenic_R = labeled_data["r/r_0"][288] * self.R * star.R_au
        sub_alfvenic_z = labeled_data["y"][288] * sub_alfvenic_R

        sub_alfvenic = np.less_equal(np.abs(self.z) * star.R_au, np.abs(sub_alfvenic_z))

        quantity = "n_MHD"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)

        A = Macc / np.sqrt(star.M)
        beta = -3 / 2 + xi

        result = A * np.multiply(values_interp, (star.R_au * self.R) ** beta)
        less_than = np.less_equal(self.R * star.R_au, R_core)
        greater_than = np.greater_equal(self.R * star.R_au, R_jet)
        inside_range = np.logical_or(
            (np.logical_or(less_than, greater_than)), sub_alfvenic
        )
        mask = np.ma.masked_where(inside_range, result)
        mask = np.ma.filled(mask, 0)
        mask = (mask > 0) * ~sub_alfvenic * (abs(self.z) > z_limit)

        print(sub_alfvenic.max())
        self.regions[sub_alfvenic] = -1
        self.regions[mask] = 2

        self.rho[mask] = result[mask] * 1e6 * AMU  # kg/m3
        self.T[mask] = Tmax

        quantity = "u_r"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)

        A = np.sqrt(star.M) * 1e3
        beta = -1 / 2

        vR = A * np.multiply(values_interp, (star.R_au * self.R) ** beta)

        quantity = "u_z"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)

        vz = self._sign_z * A * np.multiply(values_interp, (star.R_au * self.R) ** beta)

        quantity = "u_phi"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)

        vp = A * np.multiply(values_interp, (star.R_au * self.R) ** beta)

        # we use spherical coordinates so convert to cartesian ...
        vx = self._cp[mask] * vR[mask] - self._sp[mask] * vp[mask]
        vy = self._sp[mask] * vR[mask] + self._cp[mask] * vp[mask]

        # ... then to spherical ...
        self.v[0, mask], self.v[1, mask], self.v[2, mask] = cartesian_to_spherical(
            vx,
            vy,
            vz[mask],
            self._ct[mask],
            self._st[mask],
            self._cp[mask],
            self._sp[mask],
        )

        # ... then check that the cylindrical obtained are correct
        vR_check, vz_check, vp_check = self.get_v_cyl()

        print(
            "diff(vR):", np.max(abs((vR_check[mask] - vR[mask]) / (1e-100 + vR[mask])))
        )
        print("diff(vz):", np.max(abs(vz_check[mask] - vz[mask]) / (1e-100 + vz[mask])))
        print("diff(vp):", np.max(abs(vp_check[mask] - vp[mask]) / (1e-100 + vp[mask])))

        A = np.sqrt(Macc * np.sqrt(star.M))
        beta = -5 / 4 + xi / 2
        quantity = "B_r"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)
        BR = A * np.multiply(values_interp, (star.R_au * self.R) ** beta)
        quantity = "B_z"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)
        Bz = A * np.multiply(values_interp, (star.R_au * self.R) ** beta)
        quantity = "B_phi"
        interpValFunc = CubicSpline(labeled_data["y"], labeled_data[quantity])
        values_interp = interpValFunc(y)
        Bphi = A * np.multiply(values_interp, (star.R_au * self.R) ** beta)
        B = np.sqrt(BR**2 + Bz**2 + Bphi**2)

        ## TO DO: add the same T law as for Knigge's disc winds if correct. ##
        self.T[mask] = Tmax

        return

    def add_conical_stellar_wind(
        self, star, Rej=1, Mloss=1e-8, thetao=30, v0=0, vinf=1e6, beta=0.5, Tmax=1e4
    ):
        """
        Kurosawa et al. 2011, MNRAS 416, 2623

        TO improve with Wilson et al. 2022, MNRAS 514, 2162–2180
        for a better match close to the stellar surface

        thetao      :: HALF opening angle of the conical wind (max pi/2).
                        The wind occupies the region 0 to thetao in the northern hemisphere, and pi to pi-thetao in the southern.
                        When thetao=pi/2, the wind becomes sphericaly symmetric (expands to 0 to pi/2 and to pi/2 to pi.).
        Rej         :: Radius at which the wind is launched. if Rej > Rt+dr, there is no overlapping with the magnetosphere.
                        Otherwise, the thetao must be changed accordingly to avoid overlap (Rej=1).
        Mloss       :: mass ejection rate in Msun/yr
        v0 & vinf   :: velocities at Rej and at r=infinity, respectively in m/s !!
        beta        :: exponent of the radial velocity of the wind (acceleration parameter)
        Tmax        :: Temperature max of the wind
        """

        oa = min(np.deg2rad(thetao), np.pi / 2)
        cos_top = np.cos(oa)

        lsw = (self.r > Rej) * (np.abs(self._ct) > cos_top)
        self.regions[lsw] = 5

        Mloss_SI = Mloss * Msun_per_year_to_SI

        vr = (vinf - v0) * (1 - Rej / self.r[lsw]) ** beta + v0  # m/s
        # 4*pi*star.R_m^2
        rho_sw = Mloss_SI / (star.S_m2 * self.r[lsw] ** 2 * (1 - cos_top))

        self.rho[lsw] = rho_sw
        self.T[lsw] = Tmax
        self.v[0, lsw] = vr

        return

    # building not working properly because density is normalised only for
    # spherically symmetric flows
    def add_stellar_wind(
        self,
        star,
        Rmin=1.0,
        Mloss=1e-14,
        beta=0.5,
        Tmax=1e4,
        v0=0,
        vinf=1000.0,
    ):
        """
        Adding a stellar wind.
        ** building: density not well normalised if not spherically symmetric **
        """
        tmp = np.copy(self.regions)
        try:
            tmp[self._ldead_zone == 1] = 1
        except:
            print("No (accreting) magnetosphere associated to the stellar wind.")
        theta_max = np.amin(self.theta, where=tmp > 0, axis=0, initial=2 * np.pi)
        lwind = ((self.regions == 0) * (self.r >= Rmin)) * (
            self.theta < theta_max[None, :, :]
        )

        vr = 1e3 * (v0 + (vinf - v0) * (1.0 - Rmin / self.r[lwind]) ** beta)
        self.v[0, lwind] = vr

        self.rho[lwind] = (
            Mloss
            * Msun_per_year_to_SI
            / (4 * np.pi * self.r[lwind] ** 2 * vr)
            / star.R_m**2
        )
        # TO DO: Normalize density
        #
        self.T[lwind] = Tmax
        self.regions[lwind] = 5

        return

    def _write(
        self,
        filename,
        Thp=0,
        Tpre_shock=9000.0,
        laccretion=True,
        rlim_au=[0, 1000],
        coord=3,
    ):
        """

        This method writes the Grid() instance to a binary file, to be used
        by the RT code MCFOST.

        Velocity fields in spherical coordinates:
        However, for debugging with MCFOST, it is possible to write the velocity
        field in cartian (coord=1) or cylindrical coordinates (coord=2).

        """

        self.calc_cells_limits(rlim_au[0], rlim_au[1])

        print("r limits:")
        print(self._r_lim[0], self._r_lim[-1])
        print("r min/max ():", self.r.min(), self.r.max())
        print("theta limits:")
        print(np.rad2deg(self._tlim[0]), np.rad2deg(self._tlim[-1]))
        print("theta min/max:")
        print(
            np.rad2deg(self.theta[0, :, 0].min()), np.rad2deg(self.theta[0, :, 0].max())
        )
        print("phi limits:")
        print(np.rad2deg(self._p_lim[0]), np.rad2deg(self._p_lim[-1]))
        print("phi max/min:")
        print(np.rad2deg(self.phi[0, 0, :].min()), np.rad2deg(self.phi[0, 0, :].max()))

        f = open(filename, "wb")

        # write sizes along each direction + limits (size + 1)
        f.write(np.array(self.shape[0], dtype=np.int32).tobytes())
        f.write(np.single(self._r_lim).tobytes())
        f.write(np.array(self.shape[1], dtype=np.int32).tobytes())
        f.write(np.single(self._sint_lim).tobytes())
        f.write(np.array(self.shape[2], dtype=np.int32).tobytes())
        f.write(np.single(self._p_lim).tobytes())

        # f.write(np.array((0, 1)[laccretion]).tobytes())
        f.write(np.array((0, 1)[laccretion], dtype=np.int32).tobytes())
        f.write(np.array(Thp, dtype=float).tobytes())
        f.write(np.array(Tpre_shock, dtype=float).tobytes())

        f.write(self.T[:, :, :].T.tobytes())
        f.write(self.rho[:, :, :].T.tobytes())
        f.write(self.ne[:, :, :].T.tobytes())
        v3d = np.zeros((3, self.shape[2], self.shape[1], self.shape[0]))
        # Similar to mcfost vfield_coord
        if coord == 1:  # cart
            print("*** Using cartesian velocity fields (coord=%d)" % coord)
            self.v[0], self.v[1], self.v[2] = self.get_v_cart()
        elif coord == 2:  # cyl
            print("*** Using cylindrical velocity fields (coord=%d)" % coord)
            self.v[0], self.v[1], self.v[2] = self.get_v_cyl()
        v3d[0] = self.v[0, :, :, :].T
        v3d[1] = self.v[2, :, :, :].T
        v3d[2] = self.v[1, :, :, :].T
        # float 32 for real and float for double precision kind=dp
        f.write(np.float32(v3d).tobytes())
        # vturb -> 0
        f.write(np.zeros(np.product(self.shape)).tobytes())
        #
        dz = np.copy(self.regions[:, :, :])
        dz[dz > 0] = 1
        f.write(np.int32(dz).T.tobytes())
        f.close()
        return

    def _write_deprec(
        self,
        filename,
        Thp=0,
        Tpre_shock=9000.0,
        laccretion=True,
        rlim_au=[0, 1000],
    ):
        """
        Deprecated version not working with recent version

        This method writes the Grid() instance to a binary file, to be used
        by the RT code MCFOST.

        Velocity field in spherical coordinates.

        """

        self.calc_cells_limits(rlim_au[0], rlim_au[1])

        print("r limits:")
        print(self._r_lim[0], self._r_lim[-1])
        print("r min/max ():", self.r.min(), self.r.max())
        print("theta limits:")
        print(np.rad2deg(self._tlim[0]), np.rad2deg(self._tlim[-1]))
        print("theta min/max:")
        print(
            np.rad2deg(self.theta[0, :, 0].min()), np.rad2deg(self.theta[0, :, 0].max())
        )
        print("phi limits:")
        print(np.rad2deg(self._p_lim[0]), np.rad2deg(self._p_lim[-1]))
        print("phi max/min:")
        print(np.rad2deg(self.phi[0, 0, :].min()), np.rad2deg(self.phi[0, 0, :].max()))

        from scipy.io import FortranFile

        f = FortranFile(filename, "w")
        # order = "F"

        # beware write
        # f.write_record(self.shape[0])
        # f.write_record(self._rlim)  # in au
        # f.write_record(self.shape[1])
        # f.write_record(self._tlim)
        # f.write_record(self.shape[2])
        # f.write_record(self._plim)
        # # cell centres
        # f.write_record(np.single(self.r[:, 0, 0]))
        # f.write_record(np.single(self.theta[0, :, 0]))
        # f.write_record(np.single(self.phi[0, 0, :]))

        # write sizes along each direction + limits (size + 1)
        f.write_record(self.shape[0])
        f.write_record(np.single(self._r_lim))
        f.write_record(self.shape[1])
        f.write_record(np.single(self._sint_lim))
        f.write_record(self.shape[2])
        f.write_record(np.single(self._p_lim))

        f.write_record((0, 1)[laccretion])
        f.write_record(float(Thp))
        f.write_record(float(Tpre_shock))

        f.write_record(self.T[:, :, :].T)  # .flatten(order=order))
        f.write_record(self.rho[:, :, :].T)  # .flatten(order=order))
        f.write_record(self.ne[:, :, :].T)  # .flatten(order=order))
        v3d = np.zeros((3, self.shape[2], self.shape[1], self.shape[0]))
        v3d[0] = self.v[
            0, :, :, :
        ].T  # .flatten(order=order)  # vfield3d(:,1) = vx, vr, vR
        v3d[1] = self.v[
            2, :, :, :
        ].T  # .flatten(order=order)  # vfield3d(:,2) = vy, vphi
        v3d[2] = self.v[
            1, :, :, :
        ].T  # .flatten(order=order)  # vfield3d(:,3) = vz, vtheta
        # float 32 for real and float for double precision kind=dp
        f.write_record(np.float32(v3d))
        # vturb -> 0
        f.write_record(np.zeros(np.product(self.shape)))
        #
        dz = np.copy(self.regions[:, :, :])  # .flatten(order=order))
        dz[dz > 0] = 1
        f.write_record(np.intc(dz.T))
        f.close()
        return

    def _write_deprec_ascii(
        self,
        filename,
        Thp=0,
        Tpre_shock=9000.0,
        laccretion=True,
        Voronoi=False,
        mask=[],
        vcoord=2,
    ):
        """
        ** Deprecated ASCII version **
        ** Still works for Voronoi-MHD files at the moment **

        This method writes the Grid() instance to an ascii file, to be used
        by the RT code MCFOST.

        if Voronoi, the data coordinates and vectors are in cartesian coordinates in AU.
        mask cells for Voronoi only.

        """
        if Voronoi and not self._volume_set:
            print("You need to compute the volume with Voronoi==True!")
            return

        if Voronoi:
            vfield_coord = 1
            v1, v2, v3 = self.get_v_cart()
            x1 = self.x.flatten()
            x2 = self.y.flatten()
            x3 = self.z.flatten()
            Nrec = 12
            fmt = ["%.8e"] * 10 + ["%d"] + ["%.8e"]
            # alpha_smooth = max(1.0,1.6*(128//self.shape[-1])**(1/3))
            hsmooth = 1 / 3 * self.volume ** (1 / 3)
            if ~np.any(mask):
                mask = [True] * self.Ncells
        else:
            vfield_coord = vcoord
            if vfield_coord == 2:
                v1, v2, v3 = self.get_v_cyl()
            elif vfield_coord == 3:
                v1, v2, v3 = self.v
            else:
                print("Error vfield_coord %d not allowed ! " % vfield_coord)
                exit()
            x1 = self.R.flatten()
            x2 = self.z.flatten()
            x3 = self.phi.flatten()
            Nrec = 11
            fmt = ["%.8e"] * 10 + ["%d"]
            mask = [True] * self.Ncells

        header = (
            "%d\n" % (vfield_coord)
            + "{:4.4f}".format(Thp)
            + " {:4.4f}".format(Tpre_shock)
            + " {:b}".format(laccretion)
        )

        data = np.zeros((Nrec, np.count_nonzero(mask)))
        data[0], data[1], data[2] = (
            x1.flatten()[mask],
            x2.flatten()[mask],
            x3.flatten()[mask],
        )  # units does not matter here, only if Voronoi
        data[3], data[4], data[5] = (
            self.T.flatten()[mask],
            self.rho.flatten()[mask],
            self.ne.flatten()[mask],
        )
        data[6], data[7], data[8] = (
            v1.flatten()[mask],
            v2.flatten()[mask],
            v3.flatten()[mask],
        )
        dz = np.copy(self.regions)
        dz[dz > 0] = 1
        data[9], data[10] = np.zeros(np.count_nonzero(mask)), dz.flatten()[mask]

        if Voronoi:
            data[Nrec - 1] = hsmooth.flatten()[mask]
        np.savetxt(filename, data.T, header=header, comments="", fmt=fmt)

        return

    # def plot_regions(self, ax, q, clb_lab="", log_norm=True, cmap="magma"):
    #     """
    #     **Building**
    #     plot the quantity q (self.rho, self.T ..) define on an instance of Grid()
    #     """

    #     if log_norm:
    #         norm = LogNorm(vmin=q[q > 0].min(), vmax=q.max())
    #     else:
    #         norm = Normalize(vmin=q.min(), vmax=q.max())

    #     im = ax.pcolormesh(
    #         self.x[:, :, 0], self.z[:, :, 0], q[:, :, 0], norm=norm, cmap=cmap
    #     )
    #     Np = self.shape[-1]
    #     im = ax.pcolormesh(
    #         self.x[:, :, Np // 2],
    #         self.z[:, :, Np // 2],
    #         q[:, :, Np // 2],
    #         norm=norm,
    #         cmap=cmap,
    #     )

    #     ax.set_xlabel("x [Rstar]")
    #     ax.set_ylabel("z [Rstar]")

    #     stdisc = Circle((0, 0), 1, fill=False)
    #     ax.add_patch(stdisc)
    #     ax_divider = make_axes_locatable(ax)
    #     cax = ax_divider.append_axes("right", size="7%", pad="2%")
    #     clb = plt.colorbar(im, cax=cax)
    #     clb.set_label(clb_lab)

    #     return

    def _plot_3d(
        self,
        Ng=50,
        show=False,
        _mayavi=False,
        cmap="gist_stern",
        show_disc=True,
        show_star=True,
        show_axes=True,
        show_T=False,
        view=(0, 0),
        logscale=False,
        p_scale=0.5,
        show_vr=False,
    ):
        """
        *** Building ***
        to do: colors, add different regions
        view = (incl,az) incl = 0, z axis pointing toward the obs. incl = 90, z is up
        """
        if _mayavi:
            try:
                from mayavi import mlab
                from scipy.interpolate import RegularGridInterpolator

            except:
                _mayavi = False
                print("(self._plot_3d) : cannot import mayavi.")
                print(" Using matplotlib.")

        if _mayavi:
            fig3D = mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None)
        else:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            fig3D = plt.figure()
            ax3d = Axes3D(fig3D)

        # Rmax = self._Rt + self._dr
        mask = self.rho > 0
        Rmax = self.r[mask].max()
        mask_surf = mask.reshape(-1, self.shape[-1])
        lmag = np.any(self.regions == 1)
        # smaller array
        if show_T:
            data_to_plot = self.T[mask]
        elif show_vr:
            data_to_plot = self.v[0, mask]
        else:
            data_to_plot = self.rho[mask]  # self.get_B_module()[mask]
        # Color scale scaling for the scatter density
        if logscale:
            data_to_plot = np.log10(data_to_plot)
        else:
            if p_scale > 0:
                data_to_plot = data_to_plot**p_scale

        zhat = (0, 0, 1)
        xhat = (1, 0, 0)
        yhat = (0, 1, 0)
        color_axes = (0, 0, 0)

        if lmag and self._beta != 0.0:
            bhat = (np.sin(np.deg2rad(self._beta)), 0, np.cos(np.deg2rad(self._beta)))

        # draw the star (all units in Rstar)
        rt, rp = np.mgrid[0 : np.pi : 1j * Ng, 0 : 2 * np.pi : 1j * Ng]
        rx = 1 * np.cos(rp) * np.sin(rt)
        ry = 1 * np.sin(rp) * np.sin(rt)
        rz = 1 * np.cos(rt)

        # draw a disc in the plane theta = pi/2 (z=0)
        r_disc_min = self.r[self.regions > 0].min()
        r_disc_max = self.r[self.regions > 0].max()
        if lmag:
            r_disc_max *= 10
        Rd, zd = np.mgrid[r_disc_min : r_disc_max : 1j * Ng, 0 : 0 : 1j * Ng]
        dx = Rd * np.cos(rp)
        dy = Rd * np.sin(rp)
        dz = zd
        if _mayavi:
            color_star = (255 / 255, 140 / 255, 0)
            color_disc = (169 / 255, 169 / 255, 169 / 255)
            color_bhat = (0, 0, 139 / 255)
        else:
            color_star = "darkorange"
            color_disc = "gray"
            color_bhat = "DarkBlue"

        if _mayavi:
            if show_star:
                mlab.mesh(rx, ry, rz, color=color_star, representation="surface")
            if show_disc:
                mlab.mesh(dx, dy, dz, color=color_disc, representation="surface")

            if show_axes:
                rotation_axis = mlab.quiver3d(
                    zhat[0],
                    zhat[1],
                    zhat[2],
                    zhat[0],
                    zhat[1],
                    2 * zhat[2],
                    color=color_axes,
                )
                x_axis = mlab.quiver3d(
                    xhat[0],
                    xhat[1],
                    xhat[2],
                    2 * xhat[0],
                    xhat[1],
                    xhat[2],
                    color=color_axes,
                )
                y_axis = mlab.quiver3d(
                    yhat[0],
                    yhat[1],
                    yhat[2],
                    yhat[0],
                    2 * yhat[1],
                    yhat[2],
                    color=color_axes,
                )
                if lmag and self._beta != 0:
                    mlab.quiver3d(
                        bhat[0],
                        bhat[1],
                        bhat[2],
                        2 * bhat[0],
                        2 * bhat[2],
                        2 * bhat[2],
                        color=color_bhat,
                    )

                mlab.orientation_axes()

                Xm, Ym, Zm = np.mgrid[
                    -Rmax : Rmax : Ng * 1j,
                    -Rmax : Rmax : Ng * 1j,
                    -Rmax : Rmax : Ng * 1j,
                ]
                finterp = RegularGridInterpolator(self.grid, self.rho, method="linear")
                vol_density = mlab.pipeline.scalar_field(
                    Xm, Ym, Zm, finterp((Xm, Ym, Zm))
                )  # ,vmin=,vmax=)
                #     vol_density = ChangeVolColormap(vol_density,cmapName="Reds",vmin=vmin,vmax=vmax,alpha=1.0)
                mlab.pipeline.volume(vol_density)

        else:
            if show_star:
                ax3d.plot_surface(rx, ry, rz, antialiased=True, color=color_star)
            if show_disc:
                ax3d.plot_surface(
                    dx, dy, dz, color=color_disc, antialiased=False, alpha=0.5
                )
            # ax3d.plot_surface(
            #     self.x.reshape(-1, self.shape[-1]) * mask_surf,
            #     self.y.reshape(-1, self.shape[-1]) * mask_surf,
            #     self.z.reshape(-1, self.shape[-1]) * mask_surf,
            #     color="lightgray",
            #     alpha=1,
            # )
            ax3d.scatter(
                self.x[mask],
                self.y[mask],
                self.z[mask],
                c=data_to_plot,
                cmap=cmap,
            )

            if show_axes:
                ax3d.quiver3D(
                    zhat[0],
                    zhat[1],
                    zhat[2],
                    zhat[0],
                    zhat[1],
                    2 * zhat[2],
                    color=color_axes,
                )
                ax3d.quiver3D(
                    xhat[0],
                    xhat[1],
                    xhat[2],
                    2 * xhat[0],
                    xhat[1],
                    xhat[2],
                    color=color_axes,
                )
                ax3d.quiver3D(
                    yhat[0],
                    yhat[1],
                    yhat[2],
                    yhat[0],
                    2 * yhat[1],
                    yhat[2],
                    color=color_axes,
                )

                if lmag and self._beta != 0:
                    ax3d.quiver3D(
                        bhat[0],
                        bhat[1],
                        bhat[2],
                        2 * bhat[0],
                        2 * bhat[1],
                        2 * bhat[2],
                        color=color_bhat,
                    )
                ax3d.view_init(90 - view[0], 90 - view[1])

            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.set_xlim(-Rmax, Rmax)
            ax3d.set_ylim(-Rmax, Rmax)
            ax3d.set_zlim(-Rmax, Rmax)

        if show:
            plt.show()

        return

    def _pinfo(self, fout=sys.stdout):
        """
        Print info about the grid and the different regions to fout.
        By defualt fout is the standard output. a file instance
        can be passed  (f = open('file','w')) to write these info. to
        the the disc.
        """
        print("** Grid's regions:", file=fout)
        print(" ----------------------- ", file=fout)
        print("Rmax = %lf Rstar" % self.Rmax, file=fout)
        for ir in self.regions_id:
            # Don't print transparent and dark regions at the moment.
            if ir == 0 or ir == -1:
                continue
            cond = self.regions == ir
            if np.any(cond):
                r = self.r[cond]
                rho = self.rho[cond]
                T = self.T[cond]
                Tavg = np.average(T, weights=rho)
                vr, vtheta, vphi = self.v[:, cond]
                vx, vy, vz = self.get_v_cart()
                vx = vx[cond]
                vy = vy[cond]
                vz = vz[cond]
                vR = self._cp[cond] * vx + self._sp[cond] * vy
                print(" <//> %s" % self.regions_label[ir], file=fout)
                # Info. specific to a regions, existing only if the proper method has been called.
                if ir == 1:
                    print(
                        "   rmi = %lf Rstar; rmo = %lf Rstar"
                        % (self._Rt, self._Rt + self._dr),
                        file=fout,
                    )
                    # with new mag, there is no main/sec columns
                    try:  # tmp
                        print(
                            "   no sec. columns ? %s" % ("No", "Yes")[self._no_sec],
                            file=fout,
                        )
                    except:
                        pass
                    print("   beta_ma = %lf deg" % self._beta, file=fout)
                    print(
                        "   Macc = %.3e Msun/yr" % (self._Macc / Msun_per_year_to_SI),
                        file=fout,
                    )
                    print("   S_shock = %.4f %s" % (self._f_shock, "%"), file=fout)
                    print("", file=fout)

                print("  --  Extent -- ", file=fout)
                print(
                    "   min(r) = %.4f R*; max(r) = %.4f R*"
                    % (r[rho > 0].min(), r.max()),
                    file=fout,
                )

                print("  -- Density -- ", file=fout)
                print(
                    "   min(rho) = %.4e kg/m3; <rho> = %.4e kg/m3; max(rho) = %.4e kg/m3"
                    % (rho[rho > 0].min(), np.mean(rho), rho.max()),
                    file=fout,
                )

                print("  -- Temperature -- ", file=fout)
                print(
                    "   min(T) = %.4e K; <T>_rho = %.4e K; max(T) = %.4e K"
                    % (T[rho > 0].min(), Tavg, T.max()),
                    file=fout,
                )

                print("  -- Velocities -- ", file=fout)
                print(
                    "   |Vx| %lf km/s %lf km/s"
                    % (1e-3 * abs(vx).max(), 1e-3 * abs(vx).min()),
                    file=fout,
                )
                print(
                    "   |Vy| %lf km/s %lf km/s"
                    % (1e-3 * abs(vy).max(), 1e-3 * abs(vy).min()),
                    file=fout,
                )
                print(
                    "   |Vz| %lf km/s %lf km/s"
                    % (1e-3 * abs(vz).max(), 1e-3 * abs(vz).min()),
                    file=fout,
                )
                print(
                    "   |VR| %lf km/s %lf km/s"
                    % (1e-3 * abs(vR).max(), 1e-3 * abs(vR).min()),
                    file=fout,
                )
                print(
                    "   |Vr| %lf km/s %lf km/s"
                    % (1e-3 * abs(vr).max(), 1e-3 * abs(vr).min()),
                    file=fout,
                )
                print(
                    "   |Vtheta| %lf km/s %lf km/s"
                    % (1e-3 * abs(vtheta).max(), 1e-3 * abs(vtheta).min()),
                    file=fout,
                )
                print(
                    "   |Vphi| %lf km/s %lf km/s"
                    % (1e-3 * abs(vphi).max(), 1e-3 * abs(vphi).min()),
                    file=fout,
                )

                print("", file=fout)
        return

    def _check_naninf(self, _attrs=["rho", "T", "v"]):
        # list_attr = [
        #     [getattr(self, attr), attr]
        #     for attr in dir(self)
        #     if (not attr.startswith("_") and type(getattr(self, attr)) != np.ndarray)
        # ]
        list_attr = [[getattr(self, attr), attr] for attr in _attrs]
        for l in list_attr:
            if np.any(np.isnan(l[0])) or np.any(np.isinf(l[0])):
                print("WARNING : self.%s has some nan/inf values!" % l[1])
        return
