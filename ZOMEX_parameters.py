import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy import signal
class Directions() : 
    data_dir = "./data/"
    file_star_spec = "lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    file_star_wl = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    file_planet = "HD189_SPIRou.pkl"
    tellurics_file = "tellurics_ESO.txt"
    orders_file = "orders_spirou.dat"

    save_file = "test.pkl"

class OrbitalParameters() : 
    T0       =  2456107.85507       #Mid-transit time [BJD]
    Porb     = 1.809886                    #Orbital period [d]
    Rp       = 127937.13 #Planet radius  [km]
    Rs       = 1316082.6  #Stellar radius [km] 
    Ks        = 0.12    # RV semi-amplitude of the star orbital motion due to planet [km/s]
    Kp        = 196.4  # RV semi-aplitude of the planet
    V_inj     = 0.0   # injected Doppler shift of the model
    V0        = -1.08   #Stellar systemic velocity [km/s]

class DataSimulationParameters() : 
    ############ Spectral resolution parameters #######
    resolution = 115000
    c0 = 3e5
    sigma = c0/resolution/(2*np.sqrt(2*np.log(2)))
    pixel_size = c0/resolution/2
    npixel = 10
    pixel_array     = np.linspace(-pixel_size/2,pixel_size/2,npixel) # Window for integrating over a SPIRou pixel [km/s] -- Velocity bins of 2.28 km/s (Donati+2020b)
    ##################################################

    ############# Signal to noise parameters #########
    SN_scaling = 10
    consider_flux_variations = True
    consider_blaze = True
    ##################################################

    ############# Consider tellurics ? #########
    consider_tellurics = True
    ##################################################


class Tools() : 
    @classmethod
    def get_rvs(_,phase,k):
        """
        --> Inputs:     - t:   Time vector
                        - k:   Semi-amplitude of the planet-induced RV signal on the host star
                        - p:   Planet orbital period
                        - t0:  Planet mid-transit time
        --> Outputs:    - Planet-induced RV signature for the input time values 
        """
        return - k*np.sin(2.*np.pi*phase)
    @classmethod
    def rvp(_,phase,k,v):
        """
        --> Inputs:     - phase: Planet orbital phase (T-T_obs)/Porb
                        - k:     Semi-amplitude of the planet RV
                        - v:     Planet RV at mid-transit

        --> Outputs:    - Planet RV for the input orbital phases
        """
        return k*np.sin(2.*np.pi*phase)+v
    @classmethod
    def broaden(_,wl,R,sigma) :  
        c0 = 299792458.0
        #we take a kernel of size 50 000 km/s, it is exagerated but is safe
        vlim = 50000.0
        nv = 5000
        v = np.linspace(-vlim,vlim,nv)
        dv = v[1]-v[0]
        #we interpolate the model onto a regularly spaced speed array
        w0 = np.mean(wl)
        speed = c0*(w0/wl-1)
        speed_int = np.arange(0.995*np.min(speed),0.995*np.max(speed),step=dv)
        fmod = PchipInterpolator(speed[::-1],R[::-1])
        mod_int = fmod(speed_int)
        second_conv = np.exp(-v**2/2/sigma**2) #instrumental kernel
        conv2_mod = 1/sigma/np.sqrt(2*np.pi)*2*vlim/nv*signal.oaconvolve(mod_int,second_conv,mode="same")  
        wl_int = w0/(1+speed_int/c0)
        #the model is largely oversampled and we don't want that, it is going to kill the calculation time
        diff = len(conv2_mod)/len(wl)
        spacing = max(1,round(5.*diff/6))
        return (wl_int[nv:-nv:spacing][::-1],conv2_mod[nv:-nv:spacing][::-1])



