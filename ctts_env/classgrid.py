
from .constants import Rsun, Msun, Ggrav, Msun_per_year_to_si, day_to_sec
from .utils import surface_integral
from .temperature import logRadLoss_to_T, T_to_logRadLoss
import numpy as np
#use astropy units ? 

class Star:
	def __init__(self,R,M,T,P,Beq):
		self.R = R
		self.M = M
		self.T = T
		self.P = P
		self.Beq = Beq
		
		self._vff = np.sqrt(self.M*Msun*Ggrav*2./self.R/Rsun)
		if self.P:
			self._veq = 2.0 * np.pi / (self.P * day_to_sec) * Rsun * self.R
		else:
			self._veq = 0.0
		
		return
		
	def R_m(self):
		return self.R * Rsun
	def R_au(self):
		return self.R * Rsun_au
		
	def M_kg(self):
		return self.M * Msun
		
	def m0(self):
		return self.Beq * 1.0 #magnetic moment at the equator at the stellar surface
		
	def S_m2(self):
		return 4*np.pi * self.R_m()**2

#use vectorised versionn
class Grid():

	def __init__(self,r,theta,phi,mcfost_grid=False):
	
		assert type(r) == np.ndarray, " r must be a numpy array!"
		assert type(theta) == np.ndarray, " theta must be a numpy array!"
		assert type(phi) == np.ndarray, " phi must be a numpy array!"

		self.shape = r.shape
		#only important for interpolation
		#Interpolation is much faster on a structured, regular grid
		self.structured = (False,True)[r.ndim > 1]
		
		self._2d = (phi.max() == phi.min()) #only a slice phi = array([0.]*Nr*Nt)
		
		self.mcfost = False
		if self.structured:
			if mcfost_grid:
				self.mcfost = True
				self.grid = (r[0,0,:],theta[0,:,0],phi[:,0,0])
			else:
				self.grid = (r[:,0,0],theta[0,:,0],phi[0,0,:])
		else:
			self.grid = np.array([r,theta,phi]).T
	
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
		

		self._cp = np.cos(self.phi) #cos(phi)
		self._sp = np.sin(self.phi) #sin(phi)
		self._st = np.sin(self.theta) #sin(theta)
		self._ct = np.cos(self.theta) #cos(theta)
		
		self.x = self.r * self._st * self._cp #Rstar
		self.y = self.r * self._st * self._sp #Rstar
		self.z = self.r * self._ct #Rstar 	
		self._sign_z = np.sign(self.z)
		self.R = self.r * self._st	
		
		shape = [3]
		for nn in self.shape:
			shape.append(nn)
		self.v = np.zeros(shape)
		
		self.rho = np.zeros(self.shape)
		self.T = np.zeros(self.shape)

		self.Rmax = 0
		
		self.regions = np.zeros(self.shape,dtype=int)
		self.regions_label = ['','Accr. Col', 'Disc Wind', 'Disc','dark']
		#regions==0: transparent
		#regions==-1: dark
		#regions==1: accretion columns
		
		
		return
		
	def add_magnetopshere(self,star,rmi=2.2,rmo=3.0,Mdot=1e-8,beta=0.,Tmax=8000,verbose=False,no_sec=False):

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
		"""
	
		self._Rt = rmi
		self._dr = rmo - rmi
		self._beta = beta
		
		ma = np.deg2rad(self._beta)
		
		self.Rmax = max(self.Rmax,rmo * (1.0 + np.tan(ma)**2))
		
		self._Macc = Mdot * Msun_per_year_to_si


		#Constant for density
		m0 = (self._Macc * star.R_m()) / ((1.0/rmi - 1.0/rmo) * 4.*np.pi) / np.sqrt(2.*Ggrav*star.M_kg())	
		
		#coordinates tilted about z, in F'
		self._xp = self.r * ( self._cp * self._st * np.cos(ma) - self._ct * np.sin(ma) )
		self._yp = self.r * ( self._sp * self._st )
		self._zp = self.r * ( self._cp * self._st * np.sin(ma) + self._ct * np.cos(ma) )
		Rp = np.sqrt(self._xp**2 + self._yp**2)
	
		cpp = self._xp/Rp
		spp = self._yp/Rp
		ctp = self._zp/self.r
		stp = np.sqrt(1.0 - ctp**2)
		
		sintheta0p_sq = (1.0 + np.tan(ma)**2 * cpp**2)**-1 #sin(theta0')**2
		yp = stp**2
		#In the Frame of the disc (i.e., not tilted)
		y = self._st**2 #Note: y is 0 if theta = 0 +- pi
		dtheta = self.grid[2][1] - self.grid[2][0]
		y[self.theta%np.pi == 0.0] = np.sin( dtheta )**2
		rM = self.r / y
		rMp = self.r * sintheta0p_sq  / yp

		#condition for accreting field lines		
		lmag = (rMp >= rmi) * (rMp <= rmo)
		self._lmag = lmag
		self.regions[lmag] = 1

		
		m = star.m0() / self.r**3 #magnetic moment at r		

		self._B = np.zeros(self.v.shape)
		self._B[0] = 2.0 * m * (np.cos(ma) * self._ct + np.sin(ma) * self._cp * self._st)
		self._B[1] = m * ( np.cos(ma)*self._st - np.sin(ma)*self._cp*self._ct )
		self._B[2] = m * np.sin(ma) * self._sp + 1e-50
		B = np.sqrt((self._B**2).sum(axis=0))
		
		#smaller arrays, only where accretion takes place
		sig_z = self._sign_z[lmag]
		#should not be negative, hence nan. Hopefully it is close to 0
		#when negative (numerical errors ?) This trick avoids nan/0 temperature.
		fact = abs(1./self.r[lmag] - 1./rMp[lmag])**0.5

		vpol = star._vff * fact
		vtor = vpol * self._B[2,lmag] / B[lmag]
		
		vr = -vpol * self._B[0,lmag] / B[lmag] * sig_z
		vt = -vpol * self._B[1,lmag] / B[lmag] * sig_z
		self.v[0,lmag] = vr; self.v[1,lmag] = vt; self.v[2,lmag] = vtor
		
		mcol = (cpp * self.z >= 0.0) * lmag
		self._mcol = np.zeros(self.shape,dtype=bool)
		self._scol = np.zeros(self._mcol.shape,dtype=bool)
		self._mcol[mcol] = True#main columns
		self._scol = ~self._mcol #secondary columns
		if no_sec:
			self.regions[lmag][self._scol] = 0 #transparent
		
		V = np.sqrt((self.v**2).sum(axis=0))
		self.rho[lmag] = B[lmag] / V[lmag]
		#normalisation of the density
		if self.structured:
			#takes values at the stellar surface or at rmin.
			#multiply mass_flux by rmin**2 ? 
			if self.mcfost:
				rhovr = self.rho[:,:,0] * self.v[0,:,:,0] * self._lmag[:,:,0] #x lmag in case rho is filled with density
															    # better to use rhovr < 0 as a condition
			else:
				rhovr = self.rho[0] * self.v[0,0] * self._lmag[0]
			#integrate over the shock area
			#mass_flux in units of rhovr / 4pi
			mass_flux, dOmega = surface_integral(self.grid[1],self.grid[2],-rhovr,axi_sym=self._2d)
			if verbose:
				print("dOmega = ", dOmega)
			rho0 = self._Macc / mass_flux / star.S_m2()
		else:
			print("Error unstructured grid not yet")
		
		self.rho[lmag] *= rho0
		vrot = self.r[lmag] * np.sqrt(y[lmag]) * star._veq
		self.v[2,lmag] += vrot 		
		
		#recompute mass flux after normalisation
		mass_flux_check = surface_integral(self.grid[1],self.grid[2],-rhovr*rho0,axi_sym=self._2d)[0] * 4*np.pi*star.R_m()**2
		if verbose:
			print("(check) Mdot/Mdot_input = %.3f"%(mass_flux_check/self._Macc))
		if abs(mass_flux_check/self._Macc - 1.0) > 1e-5:
			print(mass_flux_check, self._Macc)
			print("WARNING : problem of normalisation of mass flux in self.add_magnetosphere().")
 			
			
		#Computes the temperature of the form Lambda_cool = Qheat / nH^2
		Q = B[lmag] #self.r[lmag]**-3
		rl = Q * self.rho[lmag]**-2
		lgLambda = np.log10(rl/rl.max()) + T_to_logRadLoss(Tmax)
		self.T[lmag] = logRadLoss_to_T(lgLambda)
		
		#In case we keep secondary columns (no_sec = False)
		#The temperature is normalised so that in average Tavg = Tmax.
		#Otherwise, the maximum of T is in the secondary columns.
		if not no_sec:
			Tavg = np.average(self.T[lmag],weights=self.rho[lmag])
			self.T[lmag] *= Tmax/Tavg

		return
		
		
	def add_disc_wind(self):
		return