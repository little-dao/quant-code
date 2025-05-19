#Chapter 2 of ML for Asset Managers

# Marcenko-Pastur PDF
import numpy as np, pandas as pd

#Stock return data, which is pure random, its coresponding correlation variables should be checked through eigenvalues of cov matrix to denoise

#mp theorem: if data is truly random noise, then eigenvalues will fall inside a range [lambda_min, lambda_max], depends on how many
# rows[T] and columns[N] are in the data matrix

#if some eigenvalues fall outside this range, then the data is not pure noise, and we can use this information to denoise the data
#meaning such out of range data is not noise(signal)
#range calcualtion formula is in the code:

def mpPDF(var,q,pts):
    #q=T/N number of observations per stock, thus if this number is small, threshold(range) of eigenvalues will be wide
    #lambda_min = var*(1 - np.sqrt(1/q))
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf,index=eVal)
    return pdf


#Testing the Marcenko-Pastur Theorem
from sklearn.neighbors._kde import KernelDensity

def getPCA(matrix): #recall what PCA is!!!
    #Get eVal,eVec from a Hermitian matrix, it's Hermitian since cov mtx is symmetric
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] #sorting eval desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

#get the KDE (observed distribution of observations)
def fitKDE(obs,bWidth=.25,kernel='gaussian',x=None):
    #Fit kernel to a series of observations, and derive the probabiltiy distribution of obs
    #x is the array of values on which the fit KDE will be evaluated(dont need)
    if len(obs.shape)==1:obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None: x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:x=x.reshape(-1,1)
    logProb=kde.score_samples(x) #log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

# 1. Generate base pure noise
x = np.random.normal(size=(1000, 1000))  # 1000 obs, 1000 assets

# 2. Inject signal: add a strong rank-1 factor (common to all columns)
factor = np.random.normal(size=(1000, 1))  # 1 factor, 1000 time points
x += factor @ np.ones((1, 1000)) * 0.1      # scale it reasonably!

eVal0,eVec0=getPCA(np.corrcoef(x,rowvar=False))
pdf0=mpPDF(1.,q=x.shape[0]/float(x.shape[1]),pts=1000)
pdf1=fitKDE(np.diag(eVal0),bWidth=.01)

#visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(pdf0.index,pdf0.values,label='Marcenko-Pastur PDF',color='blue')
plt.plot(pdf1.index,pdf1.values,label='Observed PDF',color='red')
plt.title('Marcenko-Pastur PDF vs Observed PDF')
plt.xlabel('Eigenvalue')
plt.ylabel('Density')
plt.legend()
plt.show()
plt.savefig('Marcenko-Pastur PDF.png')


#quantify result
q = x.shape[0] / float(x.shape[1])
lambda_max = (1 + (1 / np.sqrt(q)))**2

eigenvals = np.diag(eVal0)
signal_mask = eigenvals > lambda_max
n_signals = np.sum(signal_mask)
print(f"Detected {n_signals} signals above λ+ = {lambda_max:.4f}")

# Optional: variance explained
total_var = np.sum(eigenvals)
signal_var = np.sum(eigenvals[signal_mask])
print(f"Signal variance: {signal_var:.2f} / {total_var:.2f} ({100*signal_var/total_var:.2f}%)")

# Optional: signal-only eigenvectors/matrix
signal_vecs = eVec0[:, signal_mask]
signal_vals = eigenvals * signal_mask
signal_corr = signal_vecs @ np.diag(signal_vals[signal_mask]) @ signal_vecs.T
signal_corr = (signal_corr + signal_corr.T) / 2
