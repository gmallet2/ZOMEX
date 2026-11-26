### ZOMEX : ZOnal Modelization of EXoplanets

Zomex allows to modelize simple 3D models and data of exoplanets, without too much complexity : it's quite fast, because it supposes well-delimited zones.
The current versions implements for example the Hot Spot Model, where a planet is supposed to have a day side, a night side, and a Hot Spot. For those three areas,
we modelize, using PetitRadTrans, 3 emission (or transit) spectrum, with the temperatures, compositions, clouds etc that seem pertinent.

-> ZOMEX_model.py : it manages the creation of emission spectrum for areas with petitRadtrans.

-> ZOMEX.py : it's the main code ! It allows to : 

    1 - create a geometry : size and position of the different areas on the planet, phase(s) at which we observe the planet.

    2 - create, imported from ZOMEX_model.py, the emission spectra for the different areas

    3 - Modelize an emission spectrum at the right phase(s), adding the effect of wind/rotation, limb darkening ;

    4 - Save the emission spectrum, in a .h5 format, managed by ZOMEX_FM.py (File Manager)

-> ZOMEX_DS.py : Important code too. It allows, using the spectrum of a simulated of an exoplanet, at many phases, to simulate data (like the data of a spectropolarimeter, this code was used for simulating SPIRou data for example.) ;

-> ZOMEX_parameter.py : Put all the parameters of the planet you want to create/simulate in that file ;

-> ZOMEX_plot.py : allows to make various interesting plots, such as a plot of the planet, with the different areas, at a given phase ;

...Some other files, see the description in each file.