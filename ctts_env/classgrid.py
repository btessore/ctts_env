from .constants import (
    Rsun,
    Msun,
    Ggrav,
    Msun_per_year_to_SI,
    day_to_sec,
    Rsun_au,
)
from .utils import surface_integral, spherical_to_cartesian
from .temperature import logRadLoss_to_T, T_to_logRadLoss
import numpy as np
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
        self.S_m2 = 4 * np.pi * self.R_m ** 2
        self.Rco = 100  # large init value

        self._vff = np.sqrt(self.M_kg * Ggrav * 2.0 / self.R_m)
        if self.P:
            self._veq = 2.0 * np.pi / (self.P * day_to_sec) * self.R_m
            self._omega = 2 * np.pi / (self.P * day_to_sec)
            self.Rco = (Ggrav * self.M_kg / self._omega ** 2) ** (
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
        # Still, can be 2.5d (z < 0 and z > 0)

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

        self.rho = np.zeros(self.shape)
        self.T = np.zeros(self.shape)
        self.ne = np.zeros(self.shape)  # electronic density

        self.Rmax = 0

        self.regions = np.zeros(self.shape, dtype=int)
        self.regions_label = ["", "Accr. Col", "Disc Wind", "Disc", "dark"]
        self.regions_id = [0, 1, 2, 3, -1]
        # 0 : transparent
        # -1: dark
        # 1 : accretion columns

        self._volume_set = False

        return

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
            self.volume = self.r ** 2 * dr * self._st * dt * dp

        self._smoothing_length = 1 / 3 * self.volume ** (1 / 3)
        self._volume_set = True
        return

    def calc_cells_surface(self):
        """
        3d grid's cell surface calculation
        """
        dt = np.gradient(self.theta, axis=1)
        dp = np.gradient(self.phi, axis=2)
        self.surface = self.r ** 2 * self._st * dt * dp
        return

    def _check_overlap(self):
        """
        *** building ***
        Check that a region is not already filled with density
        in that case, avoid it or set the density to 0 ?
        """
        return

    def add_disc(self):
        """
        Dust and gas disc
        """
        return

    def add_dark_disc(self, Rin, dwidth=0, Td=0, wall=False, phi0=0, Rwi=1, Aw=1):
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
            mask = (
                (self.R > Rwi)
                * (self.R <= Rwi + dwall)
                * (
                    (self.z <= zmin[:, None, :] + Aw * north) * (self.z >= 0)
                    | (self.z >= -zmin[:, None, :] - Aw * sud) * (self.z < 0)
                )
            )
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
        Rp = np.sqrt(self._xp ** 2 + self._yp ** 2)

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

        # to test
        # check the points for which the field line passing by these points accrete
        v_square = np.zeros(self.shape)
        # if the number of point is not enough, the field line
        # might not be resolve leading to unconsistent results
        # like positive while some points are negative
        N_fl = 10000
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
                    r0 = (
                        self.r[i, j, k]
                        * np.sin(phi0) ** 2
                        / self._st[i, j, k] ** 2
                        / self._sp[i, j, k] ** 2
                    )
                    if r0 >= rmi and r0 <= rmo:
                        ts = np.arcsin(
                            np.sqrt(
                                self.r[i, j, k]
                                / r0
                                * np.sin(phi0) ** 2
                                / self._sp[i, j, k] ** 2
                            )
                        )
                        te = np.pi / 2
                        # if self.z[i, j, k] < 0:
                        #     ts = np.pi - ts
                        t_fl = np.linspace(ts, te, N_fl)

                        # build the field line the point (i,j,k) belongs to, from the disc to the stellar surface
                        y_fl = np.sin(t_fl) ** 2
                        r_fl = r0 * self._sp[i, j, k] ** 2 / np.sin(phi0) ** 2 * y_fl
                        v2_fl = (
                            2 * Ggrav * star.M_kg / star.R_m * (1 / r_fl - 1 / r0)
                            + (y_fl * r_fl ** 2 - r0 ** 2)
                            * (star.R_m * star._omega) ** 2
                            + V0 ** 2
                        )

                        if np.alltrue(v2_fl > 0):
                            v_square[i, j, k] = (
                                2
                                * Ggrav
                                * star.M_kg
                                / star.R_m
                                * (1 / self.r[i, j, k] - 1 / r0)
                                + (self.R[i, j, k] ** 2 - r0 ** 2)
                                * (star.R_m * star._omega) ** 2
                                + V0 ** 2
                            )
        #############################################################################

        # ###self._laccr = (v_square >= 0) * (r0 >= rmi) * (r0 <= rmo)
        self._laccr = v_square > 0
        self.regions[self._laccr] = 1  # non-transparent regions.

        # smaller arrays, only where accretion takes place
        # m is the magnetic moment at the pole and at r=1.
        # - because vr must be negative around the pole. x2 because _m0 is at the equator.
        m = -2.0 * star._m0 / self.r[self._laccr] ** 3
        self._B = np.zeros(self.v.shape)
        # (Br, Btheta, Bphi)
        self._B[0, self._laccr] = (
            m * (self._st * self._cp * np.sin(ma) + self._ct * np.cos(ma))[self._laccr]
        )
        self._B[1, self._laccr] = (
            -m
            / 2
            * (self._ct * self._cp * np.sin(ma) - self._st * np.cos(ma))[self._laccr]
        )
        self._B[2, self._laccr] = m / 2 * (self._sp * np.sin(ma))[self._laccr]
        B = self.get_B_module()

        sig_z = self._sign_z[self._laccr]
        v = np.sqrt(v_square[self._laccr])

        vr = v * self._B[0, self._laccr] / B[self._laccr] * sig_z
        vt = v * self._B[1, self._laccr] / B[self._laccr] * sig_z
        vp = v * self._B[2, self._laccr] / B[self._laccr] * sig_z
        u_phi = star._veq * self.R[self._laccr] + vp
        self.v[0, self._laccr] = vr
        self.v[1, self._laccr] = vt
        self.v[2, self._laccr] = u_phi
        V = np.sqrt(vr * vr + vt * vt + vp * vp)

        # Compute the inveriant e - lOmega* (V0 not included!)
        # for testing -> r0 is not defined beware
        # self._invariant_part1 = 0.5 * (vr * vr + vt * vt + u_phi * u_phi)  # u^2 / 2
        # self._invariant_part2 = (
        #     -Ggrav * star.M_kg / (self.r[self._laccr] * star.R_m)
        # )  # -GM/R
        # self._invariant_part3 = (
        #     -(star.R_m * self.R[self._laccr]) * star._omega * self.v[2, self._laccr]
        # )  # -r u_phi * Omega*
        # self._invariant_part4 = (
        #     -Ggrav * star.M_kg / (star.R_m * r0[self._laccr])
        # )  # -GM/R0
        # self._invariant_part5 = (
        #     -0.5 * (star.R_m * r0[self._laccr]) ** 2 * star._omega ** 2
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

        self.rho[self._laccr] = eta * B[self._laccr] / V
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
            eta = self._Macc / mass_flux / star.R_m ** 2
        else:
            print("Error unstructured grid not yet")
        self.rho[self._laccr] *= eta

        # recompute mass flux after normalisation
        mass_flux_check = (
            surface_integral(
                self.grid[1], self.grid[2], -rhovr * eta, axi_sym=self._2d
            )[0]
            * star.R_m ** 2
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
        self.T[self._laccr] = logRadLoss_to_T(lgLambda)

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
        Rp = np.sqrt(self._xp ** 2 + self._yp ** 2)  # + tiny_val

        cpp = self._xp / Rp
        spp = self._yp / Rp
        ctp = self._zp / self.r
        stp = Rp / self.r  # np.sqrt(1.0 - ctp ** 2)

        sintheta0p_sq = (1.0 + np.tan(ma) ** 2 * cpp ** 2) ** -1  # sin(theta0')**2
        yp = stp ** 2
        # In the Frame of the disc (i.e., not tilted)
        y = self._st ** 2  # Note: y is 0 if theta = 0 +- pi
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
        self._B = np.zeros(self.v.shape)
        self._B[0, lmag] = (
            2.0
            * m
            * (
                np.cos(ma) * self._ct[lmag]
                + np.sin(ma) * self._cp[lmag] * self._st[lmag]
            )
        )
        self._B[1, lmag] = m * (
            np.cos(ma) * self._st[lmag] - np.sin(ma) * self._cp[lmag] * self._ct[lmag]
        )
        self._B[2, lmag] = m * np.sin(ma) * self._sp[lmag]
        B = self.get_B_module()

        sig_z = self._sign_z[lmag]

        vpol = star._vff * fact[lmag]
        vtor = vpol * self._B[2, lmag] / B[lmag]

        vr = -vpol * self._B[0, lmag] / B[lmag] * sig_z
        vt = -vpol * self._B[1, lmag] / B[lmag] * sig_z
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
            rho0 = self._Macc / mass_flux / star.R_m ** 2
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
            * star.R_m ** 2
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

    def add_disc_wind(
        self,
        star,
        Rin=3.0,
        Rout=15,
        Mloss=1e-10,
        alpha=0.5,
        gamma=-0.5,
        ls=10,
        zs=15,
        beta=0.5,
        fesc=2,
    ):
        """
        ** builing **
                Mloss :: mass ejection rate in Msun/yr
                gamma :: temperature exponant such that T \propto R**gamma
                alpha :: mass loss rate power law per unit area
                ls :: disc wind length scale in unit of Rin
                zs :: location above or below the midplane at R=0 where the field lines diverge (Source location).
                beta :: exponent of the radial velocity of the wind (acceleration parameter)
                fesc :: terminal velocity of the disc wind in unit of the escape velocity
        """
        ldw = (self.R >= Rin * (abs(self.z) + zs) / zs) * (
            self.R <= Rout * (abs(self.z) + zs) / zs
        )
        self.regions[ldw] = 2
        # if not "vff" in star.keys():
        # 	OmegasK = np.sqrt(star["M"]*Msun*GSI*2./star['R']/Rsun)
        # else:
        # 	OmegasK = star["vff"]

        # p_ml = 4.0*gamma * alpha #assuming the mass loss / m2 varies with R as mloss = cste * R^p_ml.
        # #p_ml + 1 should be < 0
        # if p_ml + 1 >= 0:
        # 	print("dk_wind error: p_ml must be < 0 !",(p_ml))
        # 	exit()

        # l_dw = ls * Rin
        # beta_dw = 0.5
        # fesc = 2.0
        # Mloss_si = Mloss * Msun_per_year_to_si

        # #mloss_surf in kg/s/m2 prop to integral over RdR of R^p_ml
        # if p_ml + 1 == -1:
        # 	norm_mloss_surf_theo = abs( np.log(Rout) - np.log(Rin) )
        # else:
        # 	norm_mloss_surf_theo = (Rout**(p_ml + 2) - Rin**(p_ml + 2))/(p_ml + 2)
        # norm_mloss_surf_theo = Mloss_si / norm_mloss_surf_theo * (star["R"]*Rsun)**-2 #in kg/s/m2 / R^(p_ml+2)
        # #norm in kg/s/(Rstar_m * R)^2/R^p_ml
        # #such that norm * Rm**p_ml in kg/s/m2
        # norm_mloss = norm_mloss_surf_theo

        # dw = (g['R'] >= Rin * (abs(g['z']) + zs) / zs) * (g['R'] <= Rout * (abs(g['z']) + zs) / zs)
        # g['dz'][dw] = 2

        # #smaller arrays of size np.count_nonzero(dw)
        # sign_z = np.sign(g['z'])[dw]

        # #the constant norm_mloss takes Rm in stellar radius
        # mloss_loc = norm_mloss * g['R'][dw] **p_ml #norm_mloss in kg/s/m2/R**p_ml
        # #for each Rm found the corresponding wi i.e., Rm for z=0
        # wi = zs / (abs(g['z'][dw]) + zs) * g['R'][dw]
        # vKz0 = OmegasK / np.sqrt(2.0) / np.sqrt(wi)
        # vK = vKz0 * (wi / g['R'][dw]) #output

        # #distance from the source point where the field lines diverge
        # q = np.sqrt( g['R']**2 + (abs(g["z"]) + zs)**2 )[dw]
        # cos_delta = (abs(g["z"])[dw] + zs)  / q
        # l = q - zs / cos_delta
        # vesc = OmegasK / np.sqrt(g['R'][dw])
        # cs = 1e4 * (Rin / wi)**0.5 #m/s
        # vr = cs + (fesc*vesc  - cs) * (1.0 - ls/(l+ls))**beta

        # sintheta_dw = (g['z'][dw] + zs) / q
        # vR = vr * sintheta #+vtheta * costheta
        # vz = sign_z * vr * np.sqrt(1.0 - sintheta * sintheta) #-vtheta * sintheta

        # rho0 = mloss_loc / (vr * cos_delta) * (zs / (q * cos_delta))**2

        # rho[dw] = rho0
        return

    def get_B_module(self):
        return np.sqrt((self._B ** 2).sum(axis=0))

    def get_v_module(self):
        return np.sqrt((self.v ** 2).sum(axis=0))

    def get_v_cart(self):
        vx, vy, vz = spherical_to_cartesian(
            self.v[0], self.v[1], self.v[2], self._ct, self._st, self._cp, self._sp
        )
        return vx, vy, vz

    def get_v_cyl(self):
        vx, vy, vz = self.get_v_cart()
        vR = vx * self._cp + vy * self._sp
        return vR, vz, self.v[2]

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

    def _write(
        self,
        filename,
        Thp=0,
        Tpre_shock=9000.0,
        laccretion=True,
        Voronoi=False,
        mask=[],
    ):
        """

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
            vfield_coord = 2
            v1, v2, v3 = self.get_v_cyl()
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
        view=(0, 0),
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

        Rmax = self._Rt + self._dr
        mask = self.rho > 0
        mask_surf = mask.reshape(-1, self.shape[-1])

        zhat = (0, 0, 1)
        xhat = (1, 0, 0)
        yhat = (0, 1, 0)
        color_axes = (0, 0, 0)

        if self._beta != 0.0:
            bhat = (np.sin(np.deg2rad(self._beta)), 0, np.cos(np.deg2rad(self._beta)))

        # draw the star (all units in Rstar)
        rt, rp = np.mgrid[0 : np.pi : 1j * Ng, 0 : 2 * np.pi : 1j * Ng]
        rx = 1 * np.cos(rp) * np.sin(rt)
        ry = 1 * np.sin(rp) * np.sin(rt)
        rz = 1 * np.cos(rt)

        # draw a disc in the plane theta = pi/2 (z=0)
        Rd, zd = np.mgrid[self._Rt : self._Rt * 10 : 1j * Ng, 0 : 0 : 1j * Ng]
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
                if self._beta != 0:
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
                c=self.get_B_module()[mask],
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

                if self._beta != 0:
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
            # Dont' print transparent and dark regions
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
                        "rmi = %lf Rstar; rmo = %lf Rstar"
                        % (self._Rt, self._Rt + self._dr),
                        file=fout,
                    )
                    # with new mag, there is no main/sec columns
                    try:  # tmp
                        print(
                            "no sec. columns ? %s" % ("No", "Yes")[self._no_sec],
                            file=fout,
                        )
                    except:
                        pass
                    print("beta_ma = %lf deg" % self._beta, file=fout)
                    print(
                        "Macc = %.3e Msun/yr" % (self._Macc / Msun_per_year_to_SI),
                        file=fout,
                    )
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
