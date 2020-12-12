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
    import H2_Module as Mod

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

def Continuum_State_Plotter(input_par, m_n_array):
    CS, CE, CSC = Mod.Continuum_State_Reader(input_par)
   
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    dr = input_par["grid_spacing"]
    r = grid
    r_val = r[-2]
    z = 1

    for m_n in m_n_array:
        m, n = m_n[0], m_n[1]
        wave_function = np.zeros(len(grid), dtype=complex)
        # print(n, np.absolute(CE[(m, n)]))
        
        for l in range(0, input_par["l_max_bound_state"] + 1):
            wave_function += CS[(l, m, n)]
        
        coul_wave_r = -1*np.linalg.norm(wave_function[-2])
        dcoul_wave_r = -1*np.linalg.norm((wave_function[-1]-wave_function[-3])/(2*dr))

        k = np.sqrt(2*np.absolute(CE[(m, n)]))
        norm = np.sqrt(np.abs(coul_wave_r)**2+(np.abs(dcoul_wave_r)/(k+z/(k*r_val)))**2)
        # norm = np.absolute(wave_function.max())
        plt.plot(grid, np.absolute(wave_function)/norm)

    # plt.xlim(0,50)
    plt.savefig("CWF.png")

if __name__=="__main__":

    input_par = Mod.Input_File_Reader(input_file = "input.json")
    
    m_n_array = []
    # for n in range(0, 1):
    
    m_n_array.append((0,50))
    m_n_array.append((0,1000))
    m_n_array.append((0,1500))

    Continuum_State_Plotter(input_par, m_n_array)

    # Bound_State_Plotter(input_par, m_n_array)

