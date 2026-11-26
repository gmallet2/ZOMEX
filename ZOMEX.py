import numpy as np
import matplotlib.pyplot as plt
from ZOMEX_plot import PlotPlanet
import time
import os
for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
from scipy.interpolate import PchipInterpolator
from ZOMEX_spec import CreateSpectra
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
from ZOMEX_FM import ZOMEXFM
NS = 0
DS = 1
HS = 2

C = 299792.458  #km/s 

class Grid() : 
    """
    This class manages the grid of points on the planet surface, for the Hot Spot Model. Thoses points are such as they are equally separated when seen from the observer, so they will 
    all have the same weight when summing the spectrum coming from all points (just adding a limb darkening effect).
    We use to frames : the classical cartesian frame for calculations, and a quadrature circles frame (using and adapting the idea from Changeat et al. 2021), for creating the points at
    the right places. 
    IMPORTANT : this class is created once for each simulation, and then never changes, from one order to an other or one phase to an other. The only
    change is that it can be used to update phase and computing areas of the other ZOMEX classes at each phase change.
    Inputs : 
        -> N_mu : the number of quadrature frames ; The number of points is ~2 * N_mu^2
        -> alpha : the angle of the Hot Spot, in radians
        -> delta : the angle between the Hot Spot and the substellar point, in radians ;
    """
    def __init__(self, N_mu=100, alpha=np.radians(30),delta=np.radians(30),v_rotation = 5,v_res = 0.1):
        self.alpha = alpha
        self.delta = delta
        radius = np.linspace(0,1,N_mu-1)
        N_beta_max = int(2*np.pi / radius[1])
        N_beta = np.maximum(1,(N_beta_max*radius).astype(int))
        mu = np.repeat(np.sqrt(1-radius**2),N_beta)
        beta = np.concatenate([np.linspace(0,2*np.pi,n, endpoint=False) for n in N_beta])
        self.q_coords=np.column_stack((mu, beta)) 
        print("Number of points in the grid : "+str(self.q_coords.shape[0]))
        self._calculate_cart_coords()

        ###Manages the speed array for further interpolations
        self.R = v_res #Resolution of the velocity array. For SPIRou, 0.1 km.s-1 is very fine.
        self.v_rotation = v_rotation
        self.n = int(np.ceil(2 * self.v_rotation / self.R)) + 1
        self.velocity_grid = np.linspace(-self.v_rotation,self.v_rotation,self.n)
        self.delta_v = self.velocity_grid[1] - self.velocity_grid[0]


    def _calculate_cart_coords(self) : 
        """
        (Internal) Calculate cartesian coordinates (x,y,z) from quadrature circle (mu, beta) coordinates.
        """
        mu = self.q_coords[:,0]
        beta = self.q_coords[:,1]
        r = np.sqrt(1-mu**2)
        self.cart_coords = np.column_stack((mu,r*np.cos(beta),r*np.sin(beta)))

    def classify_points(self,phi) : 
        """
        Classify all the points in their respective area : Hot Spot, Night Side or Day Side. It's possible to adapt this code to add areas !
        the areas of the points are then saved in an array self.areas of the same number of rows as the array of points ;
        Inputs : 
            -> phi : the phase of the planet (obviously, the classification of the points change with the phase of the planet).
        """
        P_HS = np.array([np.cos(np.pi + phi + self.delta),np.sin(np.pi + phi + self.delta),0.0])
        P_N = np.array([np.cos(phi),np.sin(phi),0.0])
        Z_hotspot = np.clip(self.cart_coords @ P_HS,-1,1) >= np.cos(self.alpha)
        Z_nightside = np.clip(self.cart_coords @ P_N,-1,1) > 0
        areas = np.where(Z_hotspot,HS,DS)
        areas[Z_nightside] = NS
        return areas

    def compute_speeds(self) : 
        """
        Compute an array of radial speeds at each point, considering rotation of the planet ;
        Input : 
            -> v_max : the max radial speed (speed of rotation to be simple)
        Those speeds are then saved in self.radial_speed, of the same number of rows as the array of points ;
        """
        theta = np.arccos(self.q_coords[:,0])
        self.radial_speeds = self.v_rotation *np.sin(theta) * np.cos(self.q_coords[:,1])

    def compute_limb_darkening(self) : 
        """
        Compute an array of limb darkening coefficients at each point, considering gray case -> maybe more realistic case to be implemented
        Those coefficients are then saved in self.limb_darkening, of the same number of rows as the array of points ;
        """
        self.limb_darkening = 3/5 * (self.q_coords[:,0]+2/3)
        #self.limb_darkening=np.ones_like(self.q_coords[:,0])

    def plot_speeds(self) : 
        """Plot the speeds depending on the position on the planet"""
        _, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(self.cart_coords[:, 1],self.cart_coords[:, 2],c=self.radial_speeds,cmap='coolwarm',s=10)
        plt.colorbar(sc, ax=ax, label='Radial speed'),ax.set_xlabel('y'),ax.set_ylabel('z'),ax.set_aspect('equal')
        plt.show()
    def plot_ld(self) : 
        _, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(self.cart_coords[:, 1],self.cart_coords[:, 2],c=self.limb_darkening,cmap='plasma_r',s=10)
        plt.colorbar(sc, ax=ax, label='Limb darkening'),ax.set_xlabel('y'),ax.set_ylabel('z'),ax.set_aspect('equal')
        plt.show()
    def plot(self,phi) :
        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.cart_coords[:,1],self.cart_coords[:,2],c = self.classify_points(phi),s=4)
        PlotPlanet.plot_planet(ax,phi,self.delta,self.alpha)
        plt.show()

class ZOMEXEmissionModel() : 
    """
    This class manages to build emission spectrum for the very light and optimized Hot Spot Model. It can, more generally, be adapted for any model 
    with well delimited areas (see the Grid class on how to add new areas) : that's why it's called ZOMEX : ZOnal Modelization of EXoplanets ;
    Inputs : 
        -> N_mu : the number of quadrature frames ; The number of points is ~2 * N_mu^2, so choose it wisely
        -> alpha : the angle of the Hot Spot, in radians
        -> delta : the angle between the Hot Spot and the substellar point, in radians ;
        -> phi : the phase of the planet, in radians : 0° is when the observer sees only the night side ;
        -> v_rotation : the speed of rotation of the planet (km.s-1)
        -> the resolution in speed. Typically, it has to be spaller than the resolution of the spectrum (with SPIRou data, a spectrum has pixel of 2 km.s-1, so v_res = 0.1 is quite safe)
    The total time of calculation is N ~ N_mu**2 * 1/v_res (*n_phases if more than one phase obviously)
    """
    def __init__(self,grid : Grid,phi):

        ###Manages the grid, computation of the velocities and limb darkening etc...
        self.grid = grid
        self.grid.compute_speeds()
        self.update(phi)
        self.grid.compute_limb_darkening()


        self.interpolated = {}
        self.keys = [] #The keys corresponding to each model loaded.
        self.wl = None #The wavelength for calculation : they are the same for each model
        self.models = {} #Dictionnary of all the models loaded. Each item is a np.array() of the data.
        self.shift_arrays = {} 

    def update(self,new_phase) : 
        """
        Update to a new phase -> it calculates the new areas and the new classifications of each point !
        """
        self.phi = new_phase
        self.areas = self.grid.classify_points(self.phi)

    def add_model(self,corr_area,data_model) : 
        """
        Makes the link between an area of the planet (for example Day Side), and the corresponding model, created for example with petitRadtrans.
        Inputs : 
            -> corr_area : the number corresponding to the given area. This same number is used in the Grid Class to classify a point in that area.
            -> data_model : (np.array) : an array with 2 columns, the first one beeing the wavelengths, the second one the data. FOR NOW, IT HAS TO HAVE THE SAME
            WL AS THE OTHER MODELS ALREADY ENTERED IN THE ZOMEX SIMULATION.
        """
        self.models[str(corr_area)] = data_model[:,1]
        if len(self.keys) == 0:
            self.wl = data_model[:,0]
        else:
            if not np.array_equal(self.wl, data_model[:,0]):
                raise ValueError("Wavelength grids are different")
        self.keys.append(str(corr_area))
        self.interpolated[str(corr_area)] = PchipInterpolator(data_model[:,0],data_model[:,1])

    def shift_models(self):
        """
        This important function creates an array of interpolations of the models according to the velocity paramter. So, for each point, 
        we just have to interpolate the right spectrum with the right interpolator depending on the velocity at the point.
        """
        D = 1+self.grid.velocity_grid/C
        for i in self.keys:
            self.shift_arrays[i] =self.interpolated[i](self.wl[:,None]/D[None,:])

    def run(self) : 
        """
        Run the calculations, by doing a smart sum of the contributions of each point to the final spectrum, for each aera.
        """
        self.final_spectrum = np.zeros(self.wl.size)
        for i in self.keys:
            velocities = self.grid.radial_speeds[self.areas ==int(i)]
            weights = self.grid.limb_darkening[self.areas==int(i)]/np.sum(self.grid.limb_darkening)
            indices = np.rint((velocities +self.grid.v_rotation)/self.grid.delta_v).astype(int)
            indices = np.clip(indices, 0, self.grid.n-1)
            weights_of_shifts_arrays = np.bincount(indices,weights=weights,minlength=self.grid.n)
            self.final_spectrum += (self.shift_arrays[i] @ weights_of_shifts_arrays)    
    def plot_planet(self) : 
        self.grid.plot(self.phi)

class ZOMEXEmission() : 
    def __init__(self,grid : Grid,l_phases,l_orders,save_file=None):
        self.save_file = save_file
        self.grid = grid
        self.l_phases = l_phases
        if self.save_file != None : 
            with h5py.File(self.save_file, "a") as f:
                f.create_dataset("phase",data=self.l_phases)
        self.l_orders = l_orders
    def run_not_parallelized(self) : 
        for i in self.l_orders :
            print("Order : "+str(i))
            self.emission_model = ZOMEXEmissionModel(self.grid,phi=np.radians(0))
            self.emission_model.add_model(NS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
            self.emission_model.add_model(DS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
            self.emission_model.add_model(HS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
            self.emission_model.shift_models()
            #A.add_data(data)
            for j,phase in enumerate(self.l_phases) :
                #print("Phase : "+str(j)) 
                self.emission_model.update(phase)
                self.emission_model.run()
                if self.save_file != None : 
                    ZOMEXFM.save_data(self.save_file,j,i,self.emission_model.wl,self.emission_model.final_spectrum)
    def _task(self,order) : 
        print("Order : "+str(order))
        self.emission_model = ZOMEXEmissionModel(self.grid)
        self.emission_model.add_model(NS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
        self.emission_model.add_model(DS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
        self.emission_model.add_model(HS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
        self.emission_model.shift_models()
        #A.add_data(data)
        for j in self.l_phases :
            #print("Phase : "+str(j)) 
            self.emission_model.areas = self.emission_model.grid.classify_points(j)
            self.emission_model.run()
            #np.savetxt("./models/"+str(j)+"_"+str(order)+".txt",self.emission_model.final_spectrum) -> 3 Go pour 100 phases * 50 ordres ! 

        return self.emission_model.final_spectrum,self.emission_model.wl,order
    def run_parallelized(self) : 
        """
        STILL NOT FINISHED - DON'T USE
        """
        N_WORKERS = 4
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(self._task, i) for i in self.l_orders]
            for f in as_completed(futures):
                try:
                    spectrum,wl,order = f.result()
                    print(spectrum)
                    if order == 2 : 
                        plt.plot(wl,spectrum)
                    print("ended")
                except Exception as e:
                    print("ÉCHEC :", repr(e), flush=True)

if __name__ == "__main__" : 

    grid = Grid(N_mu=100,alpha=np.radians(30),delta=np.radians(30),v_rotation=5)
    
    t0 = time.time()
    import pickle
    with open("./data/WASP33.pkl",'rb') as specfile:
        orders_spirou,wl_spirou,I_spirou,blaze_spirou,tellurics_spirou,T_obs_spirou,phase_spirou,\
        window_spirou,berv_spirou,vstar_spirou,airmass_spirou,SN_spirou = pickle.load(specfile)
    phases = phase_spirou *2 * np.pi
    phases = np.linspace(phases[0],phases[-1],100)
    print(phases)
    #A = ZOMEXEmission(grid,phases,range(31,80,1),save_file="try_2.h5")
    #A.run_not_parallelized()
    print(time.time()-t0)
    
    t0 = time.time()
    CC = ZOMEXEmissionModel(grid,phi=np.radians(30))
    #CC.grid.plot_ld()
    #CC.grid.plot_speeds()
    #CC.plot_planet()
    CC.update(np.radians(100))
    CC.plot_planet()

    CC.add_model(NS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
    CC.add_model(DS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
    CC.add_model(HS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
    CC.shift_models()
    CC.run()


    grid_2 = Grid(N_mu=100,alpha=np.radians(30),delta=np.radians(30),v_rotation=5,v_res=0.01)
    AA = ZOMEXEmissionModel(grid_2,phi=np.radians(30))
    #AA.grid.plot_ld()
    #AA.grid.plot_speeds()
    #AA.plot_planet()
    AA.update(np.radians(100))
    #AA.plot_planet()

    AA.add_model(NS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
    AA.add_model(DS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
    AA.add_model(HS,CreateSpectra.load_from_txt("./models/Gliese_H2O/31.txt"))
    AA.shift_models()
    AA.run()
    print(time.time()-t0)
    plt.plot(CC.wl,CC.models["0"])
    plt.plot(CC.wl,CC.final_spectrum)
    plt.plot(AA.wl,AA.final_spectrum)

    #plt.plot(A.emission_model.wl,A.emission_model.final_spectrum)

    plt.legend(["original","B","C"])
    plt.show()