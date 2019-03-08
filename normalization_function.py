# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 03:41:47 2019

@author: Talha
"""

#%%
X=np.random.randint(0,256,(50000,3072))
def normal(x):
    """ make data set to spread equally around mean value.
    
    Parameters:
    ----------
        x : numpy array 
            x is an unrolled form of image array. (1024xR, 1024xG, 1024xB)
            x shape=(50000,3072)
    Return:
    -------
        norm_img : numpy array
            norm_img is standardized image of shape (3072, 50000)
    
    """
    import numpy as np
    return (x-np.mean(x,axis=0))/np.std(x,axis=0).T
    