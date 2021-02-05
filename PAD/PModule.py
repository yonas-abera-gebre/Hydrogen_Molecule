if True:
    import numpy as np
    from math import ceil, floor
    import sys
    import h5py
    import json
    from os.path import expanduser
    path = expanduser("~/Research/Hydrogen_Molecule/TDSE")
    sys.path.append(path)
    import Module as Mod


def Get_Psi(input_par, name):
    TDSE_file =  h5py.File(input_par["TDSE_File"])
    psi  = TDSE_file[name]
    psi = psi[:,0] + 1.0j*psi[:,1]
    norm = np.linalg.norm(psi)
    print("norm  = ",  norm)

    return psi

def Organize_Psi(input_par, psi):

    Psi_Dictionary = {}
    index_map_l_m, index_map_box =  Mod.Index_Map(input_par)
    
    r_ind_max = int(input_par["grid_size"] / input_par["grid_spacing"])
    r_ind_lower = 0
    r_ind_upper = r_ind_max
    
    for i in index_map_box:   
        Psi_Dictionary[i] = np.array(psi[r_ind_lower: r_ind_upper])
        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max
    return Psi_Dictionary

def Bound_State_Reader(input_par):
    Target_file = h5py.File(input_par["Target_File"])

    Bound_Dictionary = {}
    n_max = input_par["n_max"]
    n_values = np.arange(1, n_max + 1)  
    
    for l in range(n_max):
        for n in range(l + 1, n_max + 1):
            group_name = "BS_Psi_" + str(l) +"_" + str(n)
            Bound_Dictionary[(n, l)] = Target_file[group_name]
            Bound_Dictionary[(n, l)] = np.array(Bound_Dictionary[(n, l)][:,0] + 1.0j*Bound_Dictionary[(n, l)][:,1])
	
    return Bound_Dictionary

def Psi_Reader(input_par):
    TDSE_file =  h5py.File(input_par["TDSE_File"])

    Psi_Dictionary = {}
    psi  = TDSE_file["Psi_Final"]
    psi = psi[:,0] + 1.0j*psi[:,1]
    index_map_l_m, index_map_box =  Mod.Index_Map(input_par)
    
    r_ind_max = int(input_par["grid_size"] / input_par["grid_spacing"])
    r_ind_lower = 0
    r_ind_upper = r_ind_max
    
    for i in index_map_box:   
        Psi_Dictionary[i] = np.array(psi[r_ind_lower: r_ind_upper])
        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max
      
    return Psi_Dictionary

def Proj_Bound_States_Out(input_par, psi, bound_states):
    
    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    index_map_l_m, index_map_box =  Mod.Index_Map(input_par)
    n_max = input_par["n_max"]
    for i in index_map_box:   
        l = i[1]
        for n in range(l + 1, n_max + 1):
            psi[i] -= np.sum(bound_states[(n, l)]*psi[i])*bound_states[(n, l)]
    return psi

def Photo_Energy_Spectrum(input_par, k_array, COEF):
    PES = []
    idx_map_l_m, idx_map_box =  Mod.Index_Map(input_par)
    k_array = np.arange(0.05, 3.0, 0.05)
    for k in k_array:
        k = round(k, 3)
        pes = 0.0
        for key in idx_map_box:
            m, l  = key[0], key[1]   
            pes += np.abs(COEF[str(k)][str((l,m))])**2
        
        PES.append(pes)
    
    PES = np.array(PES)
    return PES