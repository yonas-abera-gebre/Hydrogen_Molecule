import numpy as np
import sys
import json
import h5py


def Make_Grid(grid_start, grid_end, h):
    grid = np.arange(h, grid_end + h, h)
    return grid

def Input_File_Reader(input_file = "input.json"):
    with open(input_file) as input_file:
        input_paramters = json.load(input_file)
    return input_paramters

def Continuum_State_Reader(input_par):
    CW_Psi = {}
    CW_Energy = {}
    CW_File = h5py.File("Hydrogen.h5")
    # CW_File = h5py.File(input_par["Target_File"])
    CSC = 100# int(np.array(CW_File["CSC"])[0][0])

    for l in range(0, input_par["l_max"] + 1):
        for i in range(CSC):

            dataset_name = "CS_Energy_" + str(l) + "_" + str(i)
            CW_Energy[l,i] = CW_File[dataset_name]
            CW_Energy[l,i] = np.array(CW_Energy[l,i][:,0] + 1.0j*CW_Energy[l,i][:,1]).real[0]

            dataset_name = "CS_Psi_" + str(l) + "_" + str(i)
            CW_Psi[l,i] = CW_File[dataset_name]
            CW_Psi[l,i] = np.array(CW_Psi[l,i][:,0] + 1.0j*CW_Psi[l,i][:,1])
            CW_Psi[l,i] /= np.array(CW_Psi[l,i]).max()
            # CW_Psi[l,i] /= np.linalg.norm(CW_Psi[l,i])


    return CW_Energy, CW_Psi, 