
import numpy as np

def surface_integral(t,p,q,axi_sym=False):
	'''
		derive the ntegral for points with values q at the surface
		of a sphere of radius 1.0.	
		
		t :: theta coordinates, 1d array
		p :: phi coordinates, 1d array	
		
		return :: S in units or 1.0 sphere radius squared.
			      Omega, the total solid angle of the sphere (=1 in unit of 4pi)
	'''
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
		fact = 2*np.pi
		if (t.min()>=0 and t.max()<=np.pi/2):
			fact *= 2
		i = 0
		int_theta = 0
		for j in range(1,len(t)):
			dOmega += abs(ct[j] - ct[j-1]) / 4 / np.pi
			int_theta += 0.5*(q[j,i]+q[j-1,i]) * abs(ct[j] - ct[j-1])
		S = int_theta * fact
		return S, dOmega * fact
			
	int_phi = 0
	for i in range(len(p)):
		int_theta = 0
		for j in range(1,len(t)):
			if i:
				dOmega += abs(ct[j] - ct[j-1]) * (p[i] - p[i-1]) / 4 / np.pi
			int_theta += 0.5*(q[j,i]+q[j-1,i]) * abs(ct[j] - ct[j-1])
		if i:
			S += 0.5 * (int_theta + int_phi) * (p[i]-p[i-1])
		int_phi = int_theta
			
	return S, dOmega