#ZOMEX data simulator
#Inspired by ANDES Dynamic simulator from Florian Debras

import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline
from astropy.io import fits
from ZOMEX_FM import ZOMEXFM
import pickle
import h5py
import matplotlib.pyplot as plt
from ZOMEX_parameters import DataSimulationParameters as DS
from ZOMEX_parameters import OrbitalParameters as OP
from ZOMEX_parameters import Tools,Directions
    
class ZOMEXDS() : 
    """
    This is the Data Simulator for ZOMEX. It uses a ZOMEX-genereted planet (saved in a .h5 file), and simulates observed data, with a spectrometer,
    such as SPIRou. For so, it 
        1 - uses features from an other simulation, from SPIRou, such as the WASP33.pkl : it gets the berv, noise, blaze ans other from this simulation to create
        realist berv, noise, blaze data...
        2 - Loads a stellar spectra, from PHOENIX data for example.
        3 - Loads tellurics (absorption from water)
        4 Then, it loads the planet, and add all the different effects for each observed order (range of wavelength) : adding the star, performing the 
        right shifts depending on the orbital parameters, then make normalizations, blaze,telluric,noise, broadening.
        5 - Finally it saves the whole simulated spectrum in a .pkl file, in the same format as the one needed in ATMOSPHERIX data processing pipeline (Klein et al. 2024)

    """
    def __init__(self,f_orders,f_planet,emission_or_transit = "emission"):
        self.f_planet = f_planet
        self.emission_or_transit = emission_or_transit
        self.orders = np.loadtxt(Directions.data_dir+f_orders)  #SPIRou order wavelength
        print(self.orders)
        with h5py.File(self.f_planet, "r") as f:
            self.phases = f["phase"][:]/(np.pi*2) #from [0,2*pi] -> [0,1]
        self.T_obs = OP.T0 + self.phases * OP.Porb
        self.n_obs = len(self.phases)
    def load_star(self,f_star_wl,f_star_data) :
        """
        Load and interpolate star from files. We usually use PHOENIX spectras : 
        UNITS HAVE TO BE IN erg.s-1.cm-2.cm-1 !
        """ 
        star_spec_read = fits.open(Directions.data_dir+f_star_data) #stellar spectra
        star_wl_read = fits.open(Directions.data_dir+f_star_wl) # wavelength of stellar spectra
        self.star_interp = PchipInterpolator(star_wl_read[0].data, star_spec_read[0].data)

    def features(self,f_features) : 
        """
        Load berv and co from a .pkl planet file of an other existing planet, and then adapt them for our phase !!! 
        """
        with open(Directions.data_dir+f_features,'rb') as specfile:
            self.orders_spirou,self.wl_spirou,I_spirou,self.blaze_spirou,tellurics_spirou,T_obs_spirou,phase_spirou,\
            window_spirou,berv_spirou,vstar_spirou,airmass_spirou,self.SN_spirou = pickle.load(specfile)
        airmass = CubicSpline(phase_spirou,airmass_spirou)
        self.airmass = airmass(self.phases)
        self.berv = np.linspace(berv_spirou[0],berv_spirou[-1],self.n_obs)
        #print(tellurics_spirou)
        #Planet induced RV
        self.Vs = Tools.get_rvs(self.phases,OP.Ks)
        self.Vc = OP.V0 + self.Vs - self.berv  #Geocentric-to-barycentric correction

        #Compute planet RV
        self.Vp = Tools.rvp(self.phases,OP.Kp,OP.V_inj)

        #plt.scatter(phase_spirou,airmass_spirou)
        #plt.scatter(self.phases,self.airmass,s=3)
        #plt.show()

    def load_tellurics(self,f_tellurics) : 
        """
        Load and interpolate tellurics from a file f_tellurics
        """
        tellurics_tot = np.loadtxt(Directions.data_dir+f_tellurics)
        self.tellurics_interp = PchipInterpolator(tellurics_tot[:,0],  tellurics_tot[:,1])

    def _planet_at_phase(self,phase_index,order) : 
        """
        Given the self.f_planet .h5 file, this function loads and interpolate the flux for the right phase and order for our planet !
        UNITS HAVE TO BE IN erg.s-1.cm-2.cm-1, but it's normally done with ZOMEX-generated .h5 planet files.
        Inputs : 
            -> phase_index : index of the phase in the self.phase list of the phases ;
            -> order : the order : for SPIRou, it is an in between 31 and 79
        Returns an interpolator of the planet at the given phase and order ;
        """
        wl,flux = ZOMEXFM.load_data(self.f_planet,phase_index,order)
        return PchipInterpolator(wl,flux)

    def _run_one_order(self,order) : #Order : 31 to 79
        """
        This task processes the data simulation for one given order (at all phases) ;
        Inputs : 
            -> order : the order number. For SPIRou, it's between 31 and 79 ;
        """
        #print(self.orders_spirou)
        indice_order=np.where(self.orders_spirou==order)[0][0]
        no=np.where(self.orders==order)[0][0]

        #print(indice_order)
        lmin = self.orders[no,1]
        lmax = self.orders[no,2]
        wl_order_high = [lmin*0.99]
        wl_next = wl_order_high[0]
        while wl_next <= lmax*1.01:
            wl_next += wl_next/(DS.resolution*5)
            wl_order_high.append(wl_next)
        wl_order_high = np.array(wl_order_high)
        self.data_order_high = np.zeros((self.n_obs,len(wl_order_high)))
        self.data_order_high_tell = np.zeros((self.n_obs,len(wl_order_high)))

            ### calculate oversampled tellurics
        if DS.consider_tellurics:
            self.tell_high = self.tellurics_interp(wl_order_high).clip(min=0) ### ensure that there are no negative values
        ### Then we create the final wavelength range for our data, at the given resolution
        ### We do as in SPIRou : the pixel size is half the resolution
        wl_order = [lmin]
        wl_next = lmin
        while wl_next <= lmax:
            wl_next += wl_next/(2*DS.resolution)
            wl_order.append(wl_next)
        wl_order = np.array(wl_order)
        
        if DS.consider_tellurics:
            self.tell_fin = self.tellurics_interp(wl_order).clip(min=0)
        #plt.plot(wl_order_high,self.star_interp(wl_order_high*10)/np.max(self.star_interp(wl_order_high*10)))
        data_order = np.zeros((self.n_obs,len(wl_order)))
        SN_order = np.zeros((self.n_obs))

        ### We prepare the noise map at the given SNR
        noise_order = np.zeros(data_order.shape)
        for i in range(len(data_order)):
            noise_order[i] = np.random.normal(scale=1./np.mean(self.SN_spirou[indice_order])/DS.SN_scaling,size=np.shape(data_order)[1])
            
        ### IF we want blaze, we interpolate the SPIRou blaze and add it to the noise
        if DS.consider_blaze:
            blaze_interp = CubicSpline(self.wl_spirou[indice_order], self.blaze_spirou[indice_order][0])
            blaze_order = blaze_interp(wl_order)
            blaze_and_noise_order = 1+(noise_order)*np.max(np.sqrt(blaze_order))/np.sqrt(blaze_order)

        for i in range(self.n_obs) : 
            self.data_order_high[i] = self.star_interp(wl_order_high*10*(1-self.Vc[i]/DS.c0)) ### *10 because phoenix is in Angstrom
            
            ### In SPIRou, the flux can vary a lot from one observation to another because of imperfect
            ### centering of the fiber. I don't know how does that trnaslate to the ELT. You can leave or suppress that
            if DS.consider_flux_variations:
                #data_order_high[i]*= random.gauss(np.mean(I_spirou[indice_order]),np.std(np.mean(I_spirou[indice_order],axis=1)))
                pass
            ### Add the planet, doppler shifted by its orbital velocity
            #if window[t]>0:
                #if phase_dependency:
            #plt.plot(wl_order_high,self.data_order_high[i])
            model_interp = self._planet_at_phase(i,order)
            #print(np.mean(self.star_interp(wl_order_high))/np.mean(model_interp(wl_order_high)))
            #plt.plot(wl_order_high,model_interp(wl_order_high))
            self.data_order_high[i] *= (1.+100000*(OP.Rp/OP.Rs)**2 * model_interp(wl_order_high*(1-(self.Vc[i]+self.Vp[i])/DS.c0))/self.data_order_high[i]) #EMISSION CASE
                #else:
                    #data_order_high[t] *= (1.-model_interp(wl_order_high*(1-(Vc[t]+Vp[t])/c0))*window[t]) # TRANSIT CASE
            #plt.plot(wl_order_high,self.data_order_high[i])
            #plt.show()
            self.data_order_high[i] /= np.max(self.data_order_high[i]) ### We normalize the data to 1, not a big deal

            ### Add tellurics
            if DS.consider_tellurics:
                self.data_order_high_tell[i] = self.data_order_high[i]*self.tell_high**self.airmass[i]
                ### Broadening of the new data at the instrumental resolution
                wl_order_high_broad,data_order_high_broad = Tools.broaden(wl_order_high,self.data_order_high_tell[i],DS.sigma*1000)
            else:
                wl_order_high_broad,data_order_high_broad = Tools.broaden(wl_order_high,self.data_order_high[i],DS.sigma*1000)
            self.data_high_broad_interp = CubicSpline(wl_order_high_broad, data_order_high_broad)
            
            ### Integrate the intensity over the pixel size. This is performed very naively here
            tmp = np.zeros(len(wl_order))
            for pp in DS.pixel_array: 
                tmp += self.data_high_broad_interp(wl_order/(1.0+(pp)/DS.c0))
            data_order[i] = tmp/len(DS.pixel_array)
        mid = int(len(wl_order)/2)
        if DS.consider_blaze:
            data_order*= blaze_and_noise_order
            SN_order =1./ np.std(blaze_and_noise_order[:,mid-int(mid/4):mid+int(mid/4)],axis=1)
        else:
            data_order *= 1+noise_order
            SN_order = 1./np.std(noise_order[:,mid-int(mid/4):mid+int(mid/4)],axis=1)
        return (data_order,wl_order,SN_order,np.tile(blaze_order,(self.n_obs,1)),(np.tile(self.tell_fin,(self.n_obs,1)).T**self.airmass).T)   
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
    def run(self,l_orders="all",f_save = "zomex_simulated_data.pkl") :
        wl = []
        data = []
        SN = []
        blaze = []
        tellurics = []
        orders_final = []
        if l_orders == "all" : 
            for i in self.orders : 
                print(i)
                try :
                    data_order,wl_order,SN_order,blaze_order,tell_order = self._run_one_order(int(i[0]))
                    plt.plot(wl_order,data_order[0])
                except(KeyError) : 
                    print("This order doesn't exist. Skipping")
            ### Store the data
                orders_final.append(int(i[0]))
                data.append(data_order)
                wl.append(wl_order)
                SN.append(SN_order)
                blaze.append(blaze_order) 
                ### To be consistent with SPIRou fits file we do this but it can be changed
                tellurics.append(tell_order)
            plt.show()
        window = np.ones_like(self.phases)
        savedata = (orders_final,wl,data,blaze,tellurics,self.T_obs,self.phases,window,self.berv,OP.V0+self.Vs,self.airmass,SN)
        with open(Directions.data_dir+f_save, 'wb') as specfile:
            pickle.dump(savedata,specfile)
print("DONE")
if __name__=="__main__" : 
    A = ZOMEXDS("orders_spirou.dat","try.h5")
    A.features("WASP33.pkl")
    A.load_star("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits","lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    A.load_tellurics("tellurics_ESO.txt")
    A.run(l_orders="all")
    #A._run_one_order(31)