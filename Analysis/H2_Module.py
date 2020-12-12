import numpy as np
import sys
import json
import h5py


def Bound_State_Reader(input_par):
    Target_file = h5py.File(input_par["Target_File"])

    Bound_States = {}
    Bound_Energies = {}
    n_max = 50
    grid = Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    for m in range(0, input_par["m_max_bound_state"] + 1):
       
        for n in range(n_max):
            group_name = "BS_Energy_" + str(m) +"_" + str(n)
            energy = Target_file[group_name]
            Bound_Energies[(m, n)] = np.array(energy[:,0] + 1.0j*energy[:,1])

            group_name = "BS_Psi_" + str(m) +"_" + str(n)

            wave_function = Target_file[group_name]
            wave_function = np.array(wave_function[:,0] + 1.0j*wave_function[:,1])

            r_ind_max = len(grid)
            r_ind_lower = 0
            r_ind_upper = r_ind_max

            for l in range(0, input_par["l_max_bound_state"] + 1):

                Bound_States[(n, l, m)] = wave_function[r_ind_lower: r_ind_upper]
                r_ind_lower = r_ind_upper
                r_ind_upper = r_ind_upper + r_ind_max
     
    
    return Bound_States, Bound_Energies

def Continuum_State_Reader(input_par):
    CW_Psi = {}
    CW_Energy = {}

    CW_File = h5py.File(input_par["Target_File"])
    CSC = 2000
    
  
    grid = Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    for m in range(0, input_par["m_max_bound_state"] + 1):
        for i in range(CSC):

            energy_dataset_name = "CS_Energy_" + str(m) + "_" + str(i)
            wave_function_dataset_name = "CS_Psi_" + str(m) + "_" + str(i)

            wave_function =  CW_File[wave_function_dataset_name]
            wave_function = np.array(wave_function[:,0] + 1.0j*wave_function[:,1])
            # wave_function /= np.linalg.norm(wave_function)

            CW_Energy[m,i] = CW_File[energy_dataset_name]
            CW_Energy[m,i] = np.array(CW_Energy[m,i][:,0] + 1.0j*CW_Energy[m,i][:,1])

            r_ind_max = len(grid)
            r_ind_lower = 0
            r_ind_upper = r_ind_max

            for l in range(0, input_par["l_max_bound_state"] + 1):

                CW_Psi[(l, m, i)] = wave_function[r_ind_lower: r_ind_upper]
                # CW_Psi[(l, m, i)] /= np.linalg.norm(np.array(CW_Psi[(l, m, i)]))
                # CW_Psi[(l, m, i)] /= np.array(CW_Psi[(l, m, i)]).max()

                r_ind_lower = r_ind_upper
                r_ind_upper = r_ind_upper + r_ind_max
            
            
    return CW_Psi, CW_Energy, CSC

def Make_Grid(grid_start, grid_end, h):
    grid = np.arange(h, grid_end + h, h)
    return grid

def Input_File_Reader(input_file = "input.json"):
    with open(input_file) as input_file:
        input_paramters = json.load(input_file)
    return input_paramters

