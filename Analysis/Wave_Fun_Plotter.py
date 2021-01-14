if True:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    import sys
    import json
    import mpl_toolkits.mplot3d.axes3d as axes3d
    from scipy.special import sph_harm
    from os.path import expanduser
    path = expanduser("~/Research/Hydrogen_Molecule/TDSE")
    sys.path.append(path)
    import Module as Mod

    import warnings
    warnings.filterwarnings("error")

def Bound_State_Plotter(input_par, m_n_array):
    
    BS, BE = Mod.Bound_State_Reader(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
     
    for m_n in m_n_array:
        m, n = m_n[0], m_n[1]
        wave_function = np.zeros(len(grid), dtype=complex)
        print(n, np.absolute(BE[(m, n)]))
        for l in range(0, input_par["l_max_bound_state"] + 1):
            wave_function += BS[(n, l, m)]
        plt.plot(grid, np.absolute(wave_function))
    
    plt.xlim(0,50)
    plt.savefig("WF.png")

def Psi_Plotter(input_par, Psi):
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    l_list = range(10, 20)
    for key in idx_map_box:
        
        l = key[1]
        if l in l_list:
            plt.plot(grid, np.absolute(Psi[key]), label= str(key))
     
    plt.xlim(0, 10)
    plt.legend()
    plt.savefig("Psi_Final.png")
if __name__=="__main__":

    input_par = Mod.Input_File_Reader(input_file = "input.json")
    
    Psi = Mod.Psi_Reader(input_par, "Psi_Final")

    
    Psi_Plotter(input_par, Psi)

    # Bound_State_Plotter(input_par, m_n_array)

