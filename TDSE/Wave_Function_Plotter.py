if True:
    import numpy as np
    import matplotlib.pyplot as plt
    import Module as Mod 
    from sympy.physics.wigner import wigner_3j as wigner3j




def Radial_Function_Plotter(wave_function, qn_list, grid):
    
    for qn in qn_list:
        psi = np.zeros(len(grid), dtype=complex)
        r_ind_max = len(grid)
        r_ind_lower = 0
        r_ind_upper = r_ind_max
        for l in range(input_par["l_max_bound_state"] + 1):
            psi += np.array(wave_function[qn[0], qn[1], qn[2]][r_ind_lower: r_ind_upper])
            r_ind_lower = r_ind_upper
            r_ind_upper = r_ind_upper + r_ind_max

        plt.plot(grid, np.absolute(psi), label = str(qn))


    plt.legend()
    plt.xlim(0, 50)
    plt.savefig("wave_function.png")
    plt.clf()


if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    energy, wave_function = Mod.Target_File_Reader(input_par)

    qn_list = [[-1,1,0], [1,1,0], [-1,1,1], [1,1,1]]
    Radial_Function_Plotter(wave_function, qn_list, grid)
