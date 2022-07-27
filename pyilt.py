import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

def logprune_idxs(size, nprune):
	if(size > nprune):
		nidx = []
		fids = np.round(np.logspace(np.log10(1), np.log10(size-1), nprune)) - 1
		cont = 0
		for idx, val in enumerate(fids):
			candidate = int(val)
			while(candidate in nidx):
				candidate += 1
			nidx.append(candidate)
		return np.array(nidx)
	else:
		return np.arange(size)
	

def ilt(time, decay, reg, nbins=256, tmin=1e-2, tmax=1e4, nprune=512):
	indexes = logprune_idxs(time.size, nprune)
	time = time[indexes]
	decay = decay[indexes]
	t2d = np.logspace(np.log10(tmin), np.log10(tmax), nbins)
	N = indexes.size
	G = t2d.size
	M = np.zeros([N,G])
	for i in range(G):
		M[:,i] = np.exp(-time/t2d[i])

	if(reg == 0):
		y, err = nnls(M,decay)
	else:
		Mreg = np.concatenate((M, reg*np.eye(G)))
		seq = np.concatenate((decay, np.zeros(G)))
		y, err = nnls(Mreg,seq)

	return t2d, y, err

def lcurve(time, decay, nregs=30, nbins=256, tmin=1e-2, tmax=1e4, nprune=512):
	s = np.logspace(-1.5, 1.5, nregs)
	ts = np.zeros([nregs,nbins])
	dts = np.zeros([nregs,nbins])
	errv = np.zeros(nregs)
	solv = np.zeros(nregs)
	for i in range(nregs):
		reg = s[i]
		t,dt,err=ilt(time,decay,reg,nbins,tmin,tmax,nprune)
		errv[i] = err
		solv[i] = np.sqrt(np.dot(dt,dt))
		ts[i,:] = t
		dts[i,:] = dt
		
		print(f'Iteration {i}, Reguralizer {reg}, Solution {solv[i]}')

	return ts, dts, errv, solv



if __name__ == '__main__':
	size = 10000
	times = np.linspace(0,2000,size)
	decay = np.exp(-times/1000)
	noise = 0.1*np.random.randn(size)
	decay = decay + noise
	bins, amps, errs, solvs = lcurve(times, decay)
	plt.loglog(errs,solvs)
	plt.show()
