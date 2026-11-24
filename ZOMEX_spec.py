import os
import glob
import numpy as np
class CreateSpectra() : 
    """
    All those functions are classmethod to create or load different kinds of spectra, and returning an output in the standard format : 
    The units are not important, the ouput spectrum of ZOMEX will be in the same units as the input spectras ;
    Output :  np.array([[wl_1,flux_1],
                        [wl_2,flux_2],
                        ............])
    """
    @classmethod
    def load_from_txt(_,file) : #Load from classical .txt 2 columns arrays
        return np.loadtxt(file)
    @classmethod
    def dirac_spectrum(_,wl_range = [2000,2050],n_points=30000,width_dirac = 1) :#width in number of points 
        wl = np.linspace(wl_range[0],wl_range[1],n_points)
        data = np.zeros_like(wl)
        data[int((wl_range[1]-wl_range[0] - width_dirac)/2):int((wl_range[1]-wl_range[0] + width_dirac)/2)] = 1
        return np.stack((wl,data),axis=1)