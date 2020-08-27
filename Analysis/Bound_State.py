if True:
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import h5py
    import sys
    import json
    import H2_Module as Mod

def Field_Free_Wavefunction_Reader(Target_File, input_par, n):
    FF_WF = {}
    psi = Target_File["Psi_" + "1" + "_" + str(n)]
    psi = psi[:,0] + 1.0j*psi[:,1]
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    r_ind_max = len(grid)
    r_ind_lower = 0
    r_ind_upper = r_ind_max

    for l in range(input_par["l_max"] + 1):
        FF_WF[(l)] = np.array(psi[r_ind_lower: r_ind_upper])

        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max

    return(FF_WF)

def Data():
    R_0 = np.concatenate((np.arange(1, 3.2, 0.2), np.arange(3.5, 5.0, 0.5)))
    R_0 = np.concatenate((R_0, np.arange(6, 12.0, 2)))
    R_0 = np.concatenate((np.arange(0.1, 1.0, 0.1), R_0))
    E_0_Sym = np.array([8.00009, 2.98983, 1.47686, 0.68692, 0.27701, -0.02039, -0.17633, -0.3161, -0.38553, -0.46073,  -0.53612, -0.57586, -0.59588, -0.60449, -0.60631, -0.60406, -0.59939, -0.59333, -0.58656, -0.57949, -0.5585, -0.5467, -0.53057, -0.50863, -0.49252, -0.48358])
    E_1_Sym = np.array([9.49933, 4.49726, 2.82738, 1.98901, 1.48334, 1.14165, 0.89601, 0.70597, 0.55839, 0.43324,  0.24214, 0.09909, -0.01209, -0.10034, -0.17124, -0.22866, -0.27541, -0.31364, -0.34503, -0.37088, -0.41253, -0.44678, -0.46094, -0.48785, -0.49005, -0.4815])
    E_2_Sym = np.array([9.49997, 4.4987, 2.85174, 2.02431, 1.53657, 1.20847, 0.9821, 0.80952, 0.68104, 0.57583,  0.42405, 0.31859, 0.24173, 0.18369, 0.13864, 0.1029, 0.07403, 0.05036, 0.03071, 0.01422, -0.01661, -0.03858, -0.07451, -0.14506, -0.17217, -0.17151])
    E_3_Sym = np.array([9.77758, 4.77697, 3.10937, 2.27456, 1.77297, 1.43727, 1.19722, 1.01558, 0.87467, 0.76011,  0.58827, 0.46487, 0.37231, 0.30066, 0.24386, 0.19794, 0.16021, 0.12878, 0.10228, 0.07567, 0.01437, -0.03583, -0.05372, -0.08063, -0.09604, -0.10329])
    E_4_Sym = np.array([9.77775, 4.77739, 3.11086, 2.27732, 1.77706, 1.44341, 1.20493, 1.02591, 0.88651, 0.77481,  0.60675, 0.48598, 0.3946, 0.32267, 0.26418, 0.21532, 0.17354, 0.13709, 0.10475, 0.07972, 0.03671, 0.00473, -0.01788, -0.06046, -0.08669, -0.09932])

    idx = np.argmin(E_0_Sym)
    print(R_0[idx])
    print(E_0_Sym[idx])
    
    plt.plot(R_0, E_0_Sym)
    # plt.plot(R_0, E_1_Sym)
    # plt.plot(R_0, E_2_Sym)
    # plt.plot(R_0, E_3_Sym)
    # plt.plot(R_0, E_4_Sym)
    # plt.xlim(1, 6)
    plt.savefig("Energy.png")

if __name__=="__main__":
    Data()

    # file = sys.argv[1]
    
    # input_par = Mod.Input_File_Reader(file + "input.json")
    # grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    
    # Target_File = h5py.File(file + "/" + input_par["Target_File"])

    # for n in range(0, 5):
    #     FF_WF = Field_Free_Wavefunction_Reader(Target_File, input_par, n)
    #     psi = np.zeros(len(FF_WF[0]), dtype=complex)
    #     for k in FF_WF.keys():
    #         psi += FF_WF[k]
    #     plt.plot(grid, np.absolute(psi))

    # plt.xlim(0, 35)
    # plt.savefig("Bound_States_1.png")
