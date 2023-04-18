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

def ilt_with_factor(time, decay, reg, nbins=256, tmin=1e-2, tmax=1e4, nprune=512, factor=1.0):
	
	indexes = logprune_idxs(time.size, nprune)
	time = time[indexes]
	decay = decay[indexes]
	
	t2d = np.logspace(np.log10(tmin), np.log10(tmax), nbins)

	N = indexes.size
	G = t2d.size
	M = np.zeros([N,G])

	if type(factor) is not np.ndarray:
		print(f"Warning: precond is not an instance of np.ndarray")
		factor = factor*np.ones(time.size)

	pfactor = factor[indexes]

	for i in range(N):
		M[i,:] = np.exp(-pfactor[i]*time[i]/t2d)
	
	if(reg == 0):
		y, err = nnls(M,decay)
	else:
		Mreg = np.concatenate((M, reg*np.eye(G)))
		seq = np.concatenate((decay, np.zeros(G)))
		y, err = nnls(Mreg,seq)

	return t2d, y, err

def ilt_with_precond(time, decay, reg, nbins=256, tmin=1e-2, tmax=1e4, nprune=512, precond=1.0):

	indexes = logprune_idxs(time.size, nprune)
	time = time[indexes]
	decay = decay[indexes]

	t2d = np.logspace(np.log10(tmin), np.log10(tmax), nbins)
	N = indexes.size
	G = t2d.size
	M = np.zeros([N,G])

	if not isinstance(precond, np.ndarray):
		print(f"Warning: precond is not an instance of np.ndarray")
		precond = np.ones([N,G])

	if(isinstance(precond, np.ndarray) and not np.array_equal(precond.shape, M.shape)):
		print(f"Warning: precond [{precond.shape}] and input matrix [{M.shape}] are uncompatible")
		precond = np.ones([N,G])

	pprecond = precond[indexes, :]

	for i in range(N):
		for j in range(G):
			M[i,j] = np.exp(-pprecond[i,j]*time[i]/t2d[j])

	if(reg == 0):
		y, err = nnls(M,decay)
	else:
		Mreg = np.concatenate((M, reg*np.eye(G)))
		seq = np.concatenate((decay, np.zeros(G)))
		y, err = nnls(Mreg,seq)

	return t2d, y, err


def lcurve(time, decay, nregs=30, reglims=[-1.5,1.5], nbins=256, tmin=1e-2, tmax=1e4, nprune=512):
	regs = np.logspace(reglims[0], reglims[1], nregs)
	ts = np.zeros([nregs,nbins])
	dts = np.zeros([nregs,nbins])
	errv = np.zeros(nregs)
	solv = np.zeros(nregs)

	for i, reg in enumerate(regs):
		t,dt,err=ilt(time,decay,reg,nbins,tmin,tmax,nprune)
		errv[i] = err
		solv[i] = np.sqrt(np.dot(dt,dt))
		ts[i,:] = t
		dts[i,:] = dt

		print(f'Iteration {i+1}/{nregs}, Reguralizer {reg:.6f}, Solution {solv[i]:.6f}      ', end='\r')
	print('')

	return ts, dts, regs, errv, solv

def lcurve_with_factor(time, decay, nregs=30, reglims=[-1.5,1.5], nbins=256, tmin=1e-2, tmax=1e4, nprune=512, factor=1.0):
	regs = np.logspace(reglims[0], reglims[1], nregs)
	ts = np.zeros([nregs,nbins])
	dts = np.zeros([nregs,nbins])
	errv = np.zeros(nregs)
	solv = np.zeros(nregs)

	if type(factor) is not np.ndarray:
		print(f"Warning: precond is not an instance of np.ndarray")
		factor = factor*np.ones(time.size)

	for i, reg in enumerate(regs):
		t,dt,err=ilt_with_factor(time,decay,reg,nbins,tmin,tmax,nprune,factor)
		errv[i] = err
		solv[i] = np.sqrt(np.dot(dt,dt))
		ts[i,:] = t
		dts[i,:] = dt

		print(f'Iteration {i+1}/{nregs}, Reguralizer {reg:.6f}, Solution {solv[i]:.6f}      ', end='\r')
	print('')

	return ts, dts, regs, errv, solv

def lcurve_with_precond(time, decay, nregs=30, reglims=[-1.5,1.5], nbins=256, tmin=1e-2, tmax=1e4, nprune=512, precond=1.0):
	regs = np.logspace(reglims[0], reglims[1], nregs)
	ts = np.zeros([nregs,nbins])
	dts = np.zeros([nregs,nbins])
	errv = np.zeros(nregs)
	solv = np.zeros(nregs)

	if (type(precond) is not np.ndarray):
		print(f"Warning: precond is not an instance of np.ndarray")
		precond = np.ones([nprune, nbins])

	if not np.array_equal(precond.shape, np.array([nprune, nbins])):
		print(f"Warning: precond [{precond.shape}] and input matrix [{nprune},{nbins}] are uncompatible")
		precond = np.ones([nprune, nbins])

	for i, reg in enumerate(regs):
		t,dt,err=ilt_with_precond(time,decay,reg,nbins,tmin,tmax,nprune,precond)
		errv[i] = err
		solv[i] = np.sqrt(np.dot(dt,dt))
		ts[i,:] = t
		dts[i,:] = dt

		print(f'Iteration {i+1}/{nregs}, Reguralizer {reg:.6f}, Solution {solv[i]:.6f}      ', end='\r')
	print('')

	return ts, dts, regs, errv, solv

def plot_lcurve(errv, solv):
	fig, ax = plt.subplots(figsize=(4,4), constrained_layout=True)
	ax.set_title('L Curve criterion');
	ax.set_xlabel('Aproximation error');
	ax.set_ylabel('Magnitude');
	ax.loglog(errv,solv);
	
def plot_t2dist(bins, amps):
	fig, ax = plt.subplots(figsize=(4,4), constrained_layout=True)
	ax.set_title('T2 distribution');
	ax.set_xlabel('T2');
	ax.set_ylabel('Vol fraction');
	ax.semilogx(bins,amps);
	

# if __name__ == '__main__':
	# size = 10000
	# times = np.linspace(0,2000,size)
	# decay = np.exp(-times/1000)
	# noise = 0.1*np.random.randn(size)
	# decay = decay + noise
	# bins, amps, regs, errs, solvs = lcurve(times, decay)
	# plt.loglog(errs,solvs)
	# plt.show()

