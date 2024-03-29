if True:
    import TISE
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import sys
    import Module as Mod 
    import cProfile
    import Propagate as Prop
    import Potential as Pot
    import Interaction as Int

if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

def Eigen_State_Solver(input_par):
    solver = input_par["solver"]
    if solver == "File":
        if rank == 0:
            print("Reading eigenstates from " + input_par["Target_File"] + "\n")
        energy, wave_function = Mod.Target_File_Reader(input_par)
        return energy, wave_function

    elif solver == "SLEPC":
        if rank == 0:
            print("Calculating the Eigenstates and storing them in " + input_par["Target_File"] + "\n")
    
        TISE.TISE(input_par)   
        energy, wave_function = Mod.Target_File_Reader(input_par)
        return energy, wave_function

    else:
        print("\nArgumet for the solver must be 'File' or 'SLEPC' ")
   

def Inital_State(input_par, wave_function):
    inital_state = input_par["inital_state"]
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    psi = np.zeros(int(len(grid) * len(idx_map_l_m)), dtype=complex)

    for amp, m_n_par in zip(inital_state["amplitudes"], inital_state["m,n,parity_values"]):
    
        psi_m_n = amp * wave_function[(m_n_par[0], m_n_par[1], m_n_par[2])]
        
        box_idx =  idx_map_box[(m_n_par[0], abs(m_n_par[0]))] * grid.size
        low_idx = int(grid.size*abs(m_n_par[0]))
        high_idx = int(min(input_par["l_max"], input_par["l_max_bound_state"])*grid.size + grid.size)

        psi_m_n = psi_m_n[low_idx:high_idx]
        psi[box_idx: box_idx + len(psi_m_n)] += psi_m_n

    psi = psi / np.linalg.norm(psi)   

    return psi 

if __name__=="__main__":

    input_par = Mod.Input_File_Reader(input_file = "input.json")
  
    energy, wave_function = Eigen_State_Solver(input_par)


   
    psi_inital = Inital_State(input_par, wave_function)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    # plt.plot(np.absolute(psi_inital))
    # grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    # ax.set_xticks(np.arange(0*len(grid), 10*len(grid), len(grid)))
    # xlabel = idx_map_box.keys()
    # ax.set_xticklabels(xlabel)
    # plt.tight_layout()
    # plt.xlim(-100, 10*len(grid))
    # plt.savefig("psi.png")
    # plt.clf()



    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    # if rank == 0:
    #     print("Running Propagator \n \n")
        
    Prop.Crank_Nicolson_Time_Propagator(input_par, psi_inital)
    
    # pr.disable()
    # if rank == 0:
    #     pr.print_stats(sort='time')

    