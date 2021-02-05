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


def Population_Calculator(Psi, FF_WF, input_par):
    Pop = {}
    M_N_Pop = {}
    Par = {}
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    
    l_range = input_par["l_max_bound_state"]*len(grid) + len(grid)

    pop_total = 0.0

    for m in range(-1*input_par["m_max_bound_state"] , input_par["m_max_bound_state"] + 1):
        for n in range(input_par["n_max"]):

            Par[(m,n)] = Mod.Wave_Function_Pairity(FF_WF[(m,n)], input_par)
            pop = 0.0
            for l in range(abs(m), input_par["l_max_bound_state"] + 1):
                x = FF_WF[(m,n)]
                y = Psi[(m,l)]
                pop += np.vdot(FF_WF[(m,n)][l*len(grid): l*len(grid) + len(grid)], Psi[(m,l)])

            M_N_Pop[(m,n)] = pop
            M_N_Pop[(m,n)]  = np.power(np.absolute(M_N_Pop[(m,n)]),2.0) 
            pop_total += M_N_Pop[(m,n)]

    print(pop_total)
    return M_N_Pop, Par

if __name__=="__main__":

    input_par = Mod.Input_File_Reader(input_file = "input.json")
    index_map_l_m, index_map_box =  Mod.Index_Map(input_par)

    Psi = Mod.Psi_Reader(input_par, "Psi_Final")
    energy, FF_WF = Mod.Target_File_Reader_WO_Parity(input_par)

    # for k in energy.keys():
    #     print(k, energy[k])

    M_N_Pop, Par = Population_Calculator(Psi, FF_WF, input_par)

    for k in M_N_Pop.keys():
        # if M_N_Pop[k] > 10e-5:
        print(k, M_N_Pop[k])#, energy[k], Par[k])