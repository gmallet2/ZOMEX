
import numpy as np
import batman
from scipy import interpolate
from scipy import signal

################################### Orbital parameters ##############

### Ephemerides (to compute orbital phase)
T0       =  2456107.85507       #Mid-transit time [BJD]
Porb     = 1.809886                    #Orbital period [d]

### Transit parameters -- Compute the transit window
### Using batman python package https://lweb.cfa.harvard.edu/~lkreidberg/batman/
### Get the limb-darkening coefficients in H band from Claret+2011: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/529/A75
Rp       = 127937.13 #Planet radius  [km]
Rs       = 1316082.6  #Stellar radius [km] 
ip       = 88.0    #Transit incl.  [deg]
ap       = 3.76   #Semi-maj axis  [R_star]
ep       = 0.0     #Eccentricity of Pl. orbit
wp       = 0.0     #Arg of periaps [deg]
ld_mod   = "linear"     #Limb-darkening model ["nonlinear", "quadratic", "linear"]
ld_coef  = [0.35]  #Limb-darkening coefficients 


### Radial velocity info
Ks        = 0.12    # RV semi-amplitude of the star orbital motion due to planet [km/s]
Kp        = 196.4  # RV semi-aplitude of the planet
V_inj     = 0.0   # injected Doppler shift of the model
V0        = -1.08   #S tellar systemic velocity [km/s]


### Include phase dependent model ? 
phase_dependency = False
#####################################################################



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


#####################
# ################# File names #####################
dire = "D:/IRAP/Doctorat/ANDES/data/"
file_star_spec = "lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
file_star_wl = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
file_planet = "HD189_SPIRou.pkl"


### if no phase depedency, just one model
file_model_radius = "data_CItau.txt"
file_model_wl = "wl_CItau.txt"


### if phase dependency, more complicated
dire_model_phase = dire+"W76b_spec/0G/transmission/"
prefix_model = "Spec_1_Wasp-76b-0G-longwl_phase_"
suffix_model = "_inc_0"
phase_model = [345.6, 352.5, 356.25, 0, 3.75, 7.5, 14.4]


tellurics_file = "tellurics_ESO.txt"

orders_file = "orders_spirou.dat"

save_file = "test.pkl"
#############################################################################
