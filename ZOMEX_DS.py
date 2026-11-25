#ZOMEX data simulator
#Inspired by ANDES Dynamic simulator from Florian Debras

import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline
from scipy import signal
from astropy.io import fits
from ZOMEX_FM import ZOMEXFM
import pickle
import h5py
import matplotlib.pyplot as plt
from parameters_and_functions_WASP76 import *
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




    
class ZOMEXDS() : 
    def __init__(self,data_dir,f_orders,f_planet,emission_or_transit = "emission"):
        self.data_dir = data_dir
        self.f_planet = f_planet
        self.emission_or_transit = emission_or_transit
        self.orders = np.loadtxt(data_dir+f_orders)  #SPIRou order wavelength
        print(self.orders)
        with h5py.File(self.f_planet, "r") as f:
            self.phases = f["phase"][:]/(np.pi*2) #from [0,2*pi] -> [0,1]
        #self.phases = (self.phases + 0.5) % 1 - 0.5
    def load_star(self,f_star_wl,f_star_data) : 
        star_spec_read = fits.open(self.data_dir+f_star_data) #stellar spectra
        star_wl_read = fits.open(self.data_dir+f_star_wl) # wavelength of stellar spectra
        self.star_interp = PchipInterpolator(star_wl_read[0].data, star_spec_read[0].data)

    def features(self,f_features) : 
        """
        Load berv and co from a .pkl planet file of an other existing planet, and then adapt them for our phase !!! 
        """
        with open(self.data_dir+f_features,'rb') as specfile:
            self.orders_spirou,self.wl_spirou,I_spirou,self.blaze_spirou,tellurics_spirou,T_obs_spirou,phase_spirou,\
            window_spirou,berv_spirou,vstar_spirou,airmass_spirou,self.SN_spirou = pickle.load(specfile)
        airmass = CubicSpline(phase_spirou,airmass_spirou)
        self.airmass = airmass(self.phases)
        self.berv = np.linspace(berv_spirou[0],berv_spirou[-1],len(self.phases))
        #print(tellurics_spirou)
        #Planet induced RV
        self.Vs = Tools.get_rvs(self.phases,Ks)
        self.Vc = V0 + self.Vs - self.berv  #Geocentric-to-barycentric correction

        #Compute planet RV
        self.Vp = Tools.rvp(self.phases,Kp,V_inj)

        plt.scatter(phase_spirou,airmass_spirou)
        plt.scatter(self.phases,self.airmass,s=3)
        plt.show()
    def load_tellurics(self,f_tellurics) : 
        tellurics_tot = np.loadtxt(self.data_dir+f_tellurics)
        self.tellurics_interp = PchipInterpolator(tellurics_tot[:,0],  tellurics_tot[:,1])

    def _planet_at_phase(self,phase,phase_index,order) : 
        wl,flux = ZOMEXFM.load_data(self.f_planet,phase_index,order)
        print(wl)
        print(flux)
        return PchipInterpolator(wl,flux)

    def _run_one_order(self,order) : #Order : 31 to 79
        print(self.orders_spirou)
        indice_order=np.where(self.orders_spirou==order)[0][0]
        no=np.where(self.orders==order)[0][0]

        print(indice_order)
        lmin = self.orders[no,1]
        lmax = self.orders[no,2]
        wl_order_high = [lmin*0.99]
        wl_next = wl_order_high[0]
        while wl_next <= lmax*1.01:
            wl_next += wl_next/(resolution*5)
            wl_order_high.append(wl_next)
        wl_order_high = np.array(wl_order_high)
        self.data_order_high = np.zeros((len(self.phases),len(wl_order_high)))
        self.data_order_high_tell = np.zeros((len(self.phases),len(wl_order_high)))

            ### calculate oversampled tellurics
        if consider_tellurics:
            self.tell_high = self.tellurics_interp(wl_order_high).clip(min=0) ### ensure that there are no negative values
        ### Then we create the final wavelength range for our data, at the given resolution
        ### We do as in SPIRou : the pixel size is half the resolution
        wl_order = [lmin]
        wl_next = lmin
        while wl_next <= lmax:
            wl_next += wl_next/(2*resolution)
            wl_order.append(wl_next)
        wl_order = np.array(wl_order)
        
        if consider_tellurics:
            self.tell_fin = self.tellurics_interp(wl_order).clip(min=0)
        #plt.plot(wl_order_high,self.star_interp(wl_order_high*10)/np.max(self.star_interp(wl_order_high*10)))
        data_order = np.zeros((len(self.phases),len(wl_order)))
        SN_order = np.zeros((len(self.phases)))

        ### We prepare the noise map at the given SNR
        noise_order = np.zeros(data_order.shape)
        for i in range(len(data_order)):
            noise_order[i] = np.random.normal(scale=1./np.mean(self.SN_spirou[indice_order])/SN_scaling,size=np.shape(data_order)[1])
            
        ### IF we want blaze, we interpolate the SPIRou blaze and add it to the noise
        if consider_blaze:
            #blaze_interp = interpolate.CubicSpline(self.wl_spirou[indice_order], self.blaze_spirou[indice_order][0])
            #blaze_order = blaze_interp(wl_order)
            #blaze_and_noise_order = 1+(noise_order)*np.max(np.sqrt(blaze_order))/np.sqrt(blaze_order)
            pass

        for i,phase in enumerate(self.phases) : 
            self.data_order_high[i] = self.star_interp(wl_order_high*10*(1-self.Vc[i]/c0)) ### *10 because phoenix is in Angstrom
            
            ### In SPIRou, the flux can vary a lot from one observation to another because of imperfect
            ### centering of the fiber. I don't know how does that trnaslate to the ELT. You can leave or suppress that
            if consider_flux_variations:
                #data_order_high[i]*= random.gauss(np.mean(I_spirou[indice_order]),np.std(np.mean(I_spirou[indice_order],axis=1)))
                pass
            ### Add the planet, doppler shifted by its orbital velocity
            #if window[t]>0:
                #if phase_dependency:
            plt.plot(wl_order_high,self.data_order_high[i])
            model_interp = self._planet_at_phase(phase,i,order)
            plt.plot(wl_order_high,model_interp(wl_order_high))
            self.data_order_high[i] *= (1.+100000*(Rp/Rs)**2 * model_interp(wl_order_high*(1-(self.Vc[i]+self.Vp[i])/c0))/self.data_order_high[i]) #EMISSION CASE
                #else:
                    #data_order_high[t] *= (1.-model_interp(wl_order_high*(1-(Vc[t]+Vp[t])/c0))*window[t]) # TRANSIT CASE
            plt.plot(wl_order_high,self.data_order_high[i])
            plt.show()
            self.data_order_high[i] /= np.max(self.data_order_high[i]) ### We normalize the data to 1, not a big deal

            ### Add tellurics
            if consider_tellurics:
                self.data_order_high_tell[i] = self.data_order_high[i]*self.tell_high**self.airmass[i]
                ### Broadening of the new data at the instrumental resolution
                wl_order_high_broad,data_order_high_broad = Tools.broaden(wl_order_high,self.data_order_high_tell[i],sigma*1000)
            else:
                wl_order_high_broad,data_order_high_broad = Tools.broaden(wl_order_high,self.data_order_high[i],sigma*1000)
            self.data_high_broad_interp = interpolate.CubicSpline(wl_order_high_broad, data_order_high_broad)
            
            ### Integrate the intensity over the pixel size. This is performed very naively here
            tmp = np.zeros(len(wl_order))
            for pp in pixel_array: 
                tmp += self.data_high_broad_interp(wl_order/(1.0+(pp)/c0))
            data_order[i] = tmp/len(pixel_array)
        mid = int(len(wl_order)/2)
        if consider_blaze:
            pass
            #data_order*= blaze_and_noise_order
            #SN_order =1./ np.std(blaze_and_noise_order[:,mid-int(mid/4):mid+int(mid/4)],axis=1)
        else:
            data_order *= 1+noise_order
            SN_order = 1./np.std(noise_order[:,mid-int(mid/4):mid+int(mid/4)],axis=1)

        ### Store the data
        #orders_final.append(orders_spirou[i])
        #data.append(data_order)
        #wl.append(wl_order)
        #SN.append(SN_order)
        #blaze.append(np.tile(blaze_order,(len(T_obs),1))) 
        ### To be consistent with SPIRou fits file we do this but it can be changed
        #tellurics.append((np.tile(tell_fin,(len(T_obs),1)).T**airmass).T)
        #plt.plot(wl_order,data_order[0])
        #plt.plot(wl_order,data_order[50])

        #plt.show()
    def run(self,l_orders="all") :
        if l_orders == "all" : 
            pass 
if __name__=="__main__" : 
    A = ZOMEXDS("./data/","orders_spirou.dat","try.h5")
    A.features("WASP33.pkl")
    A.load_star("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits","lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    A.load_tellurics("tellurics_ESO.txt")
    A._run_one_order(31)