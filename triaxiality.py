# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:02:32 2021

@author: Georg

Triaxiality support functions
"""

import numpy as np

def triax_function(D, R):
    """ Return triaxiality according to Bao and Wirczbiki
    D = cross section diameter
    R = neck radius
    """
    return 1./3. + np.sqrt(2.0) * np.log(1 + 0.5*D/(2*R))    

def bridgman_function(D, R):
    """ Return bridgman function according to Bao and Wirczbiky
    D = cross section diameter
    R = neck radius
    """
    # try: (1) compute stress triaxiality
    # (2) compute ratio from stress triaxiality
    #eta = 1./3. + np.sqrt(2.0) * np.log(1 + 0.5*D/(2*R))
    #a_2R = np.exp((eta-1./3.)/np.sqrt(2.0)) - 1
    #k = 1./((1 + 1./a_2R) * np.log(1 + a_2R))
    #return k

    # GCG best fit function to simulation data
    ratio = D / (2*R) # note that this is two times the original ratio of Bridgman, he has ratio = 0.5*D/(2*R)
    return 1./((1 + 1./ratio) * np.log(1 + ratio))

    #k = (1 + 2*R/D)
    
    ## Gromada et al, eq. 51 in Tu (2020)   
    r = 0.5 * D / (4*R) # corresponds to (a/4R); R is diameter/(4*notch_radius)
    beta = alpha = 0.5
    return 1./(1 + r + r * (1-beta) * alpha / (4 - alpha))


