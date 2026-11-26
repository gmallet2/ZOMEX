#File loading and saving in H5 format !
import h5py
import numpy as np
"""
Structure of the file : 
Group data
    Group "31" : 
        Group "1" -> phase index 1
        Group "2é ->...
Group phase : array of the phases
"""
class ZOMEXFM:
    @classmethod
    def save_data(_, filename, phase_index, order, wl, data):
        """
        Save wavelength and spectrum for one order and phase.
        """

        with h5py.File(filename, "a") as f:

            data_group = f.require_group("data")
            order_group = data_group.require_group(str(order))

            # Save wavelength
            if "wl" not in order_group:
                order_group.create_dataset("wl", data=wl)

            # Save spectrum
            phase_name = str(phase_index)

            if phase_name in order_group:
                del order_group[phase_name]

            order_group.create_dataset(phase_name, data=data)

    @classmethod
    def load_data(_, filename, phase, order):
        with h5py.File(filename, "r") as f:

            order_path = f"data/{order}"

            if order_path not in f:
                raise KeyError(f"Order {order} does not exist.")

            order_group = f[order_path]

            phase_name = str(phase)

            if phase_name not in order_group:
                raise KeyError(
                    f"Phase {phase} does not exist for order {order}."
                )

            wl = order_group["wl"][:]
            data = order_group[phase_name][:]

        return wl, data
    @classmethod
    def load_phases(_,filename) : 
        """
        Returns the phase in radians ([0,2*pi])
        """
        with h5py.File(filename, "r") as f :
            return f["phase"][:]