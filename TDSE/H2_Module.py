if True:
    import numpy as np
    from math import ceil, floor
    import sys
    import h5py
    import json
    import H2_Module as Mod

def Make_Grid(grid_start, grid_end, h):
    grid = np.arange(h, grid_end + h, h)
    return grid

def Coulomb_Eff_Potential(grid, l):
    return -1.0*np.power(grid, -1.0) + 0.5*l*(l+1)*np.power(grid, -2.0)

def Input_File_Reader(input_file = "input.json"):
    with open(input_file) as input_file:
        input_paramters = json.load(input_file)
    return input_paramters
    
def Index_Map_M_Block(input_par):
    l_max = input_par["l_max"]
    m_max = input_par["m_max"]
    index_map_m_l = {}
    index_map_box = {}
    block = 0
    for m in np.arange(0, m_max + 1):
        for l in np.arange(m, l_max + 1):
            index_map_m_l[block] = (m,l)
            index_map_box[(m,l)] = block
            block += 1

        if m > 0:
            m = -1*m
            for l in np.arange(abs(m), l_max + 1):
                index_map_m_l[block] = (m,l)
                index_map_box[(m,l)] = block
                block += 1

    return index_map_m_l, index_map_box


def Target_File_Reader(input_par):
    file = h5py.File(input_par["Target_File"], 'r')
    energy = {}
    wave_function = {}
    for m in range(0, input_par["m_max_bound_state"] + 1):
        even_count = 1
        odd_count = 1
        for i in range(0, input_par["n_max"]):
            
            energy_temp = file["Energy_" + str(m) + "_" + str(i)]
            energy_temp = np.array(energy_temp[:,0] + 1.0j*energy_temp[:,1])
            wave_function_temp = file["Psi_" + str(m) + "_" + str(i)]
            wave_function_temp = np.array(wave_function_temp[:,0] + 1.0j*wave_function_temp[:,1])

            pairity = Wave_Function_Pairity(wave_function_temp, input_par)
            
            if pairity == 0:
                energy[(m, even_count, 0)] = energy_temp
                wave_function[(m, even_count, 0)] = wave_function_temp
                even_count += 1

            if pairity == 1:
                energy[(m, odd_count, 1)] = energy_temp
                wave_function[(m, odd_count, 1)] = wave_function_temp
                odd_count += 1

    
    return energy, wave_function
def Wave_Function_Pairity(wave_function, input_par):
    WF = {}
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    r_ind_max = len(grid)
    r_ind_lower = 0
    r_ind_upper = r_ind_max
    for l in range(input_par["l_max"] + 1):
        WF[l] = np.array(wave_function[r_ind_lower: r_ind_upper])
        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max


    odd = 0.0
    even = 0.0
    for l in range(0, input_par["l_max"], 2):
        even += abs(np.sum(WF[l]))
        odd += abs(np.sum(WF[l + 1]))

    if odd > even:
        return 1
    else:
        return 0 
def Matrix_Build_Status(input):

    build_status = {}

    build_status["Int_Mat_X_Stat"] = False 
    build_status["Int_Mat_Y_Stat"] = False 
    build_status["Int_Mat_Z_Stat"] = False 

    build_status["Int_Mat_Right_Stat"] = False
    build_status["Int_Mat_Left_Stat"] = False

    build_status["Dip_Acc_Mat_X_Stat"] = False
    build_status["Dip_Acc_Mat_Y_Stat"] = False
    build_status["Dip_Acc_Mat_Z_Stat"] = False
    
    build_status["Dip_Mat_X_Stat"] = False
    build_status["Dip_Mat_Y_Stat"] = False
    build_status["Dip_Mat_Z_Stat"] = False

    build_status["Int_Ham_Temp"] = False
    build_status["Int_Ham_Left_Temp"] = False

    return build_status