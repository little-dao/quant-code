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

def getPCA(matrix):
    #Get eVal,eVec from a Hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix)
    i