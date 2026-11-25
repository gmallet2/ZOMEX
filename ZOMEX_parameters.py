import numpy as np
class Directions() : 
    data_dir = "D:/IRAP/Doctorat/ANDES/data/"
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
