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
    import H2_Module as Mod 
    import cProfile
    import Propagate as Prop
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
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    index_map_l_m, index_map_box = Mod.Index_Map_M_Block(input_par)
    psi_size = len(grid) * len(index_map_l_m)
    psi = np.zeros(int(psi_size), dtype=complex)

    for amp, m_n_p in zip(inital_state["amplitudes"], inital_state["m,n,parity_values"]):
        psi_n_l = amp * wave_function[(m_n_p[0], m_n_p[1], m_n_p[2])]
        psi_idx = int(m_n_p[0] * len(psi_n_l))
        psi[psi_idx : psi_idx + len(psi_n_l)] += psi_n_l

    psi = psi / np.linalg.norm(psi)   

    return psi 

if __name__=="__main__":

    input_par = Mod.Input_File_Reader(input_file = "input.json")
    if rank == 0:
        print("Getting eigen states \n")

    energy, wave_function = Eigen_State_Solver(input_par)
    if rank == 0:
        print("Making Psi \n")

    psi_inital = Inital_State(input_par, wave_function)
    
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    if rank == 0:
        print("Running Propagator \n \n")
        
    Prop.Crank_Nicolson_Time_Propagator(input_par, psi_inital)
    
    # pr.disable()
    # if rank == 0:
    #     pr.print_stats(sort='time')

    