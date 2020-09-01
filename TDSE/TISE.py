if True:
    import sys
    import time
    import json
    import H2_Module as Mod 
    from Potential import H2_Plus_Potential
    from math import ceil, floor
    import numpy as np

if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    import slepc4py 
    from slepc4py import SLEPc
    petsc4py.init(comm=PETSc.COMM_WORLD)
    slepc4py.init(sys.argv)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


def Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, input_par, m, Viewer):
    if rank == 0:
        print("Diagonalizing")
        
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian) ##pass the hamiltonian to the 
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    EV_Solver.setTolerances(input_par["tolerance"], PETSc.DECIDE)
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.SMALLEST_REAL)
    size_of_matrix = PETSc.Mat.getSize(Hamiltonian)
    dimension_size = int(size_of_matrix[0]) * 0.1
    EV_Solver.setDimensions(number_of_eigenvalues, PETSc.DECIDE, dimension_size) 
    EV_Solver.solve() 

    if rank == 0:
        print("Number of eigenvalues requested and converged")
        print(number_of_eigenvalues, EV_Solver.getConverged(), "\n")
   
    for i in range(number_of_eigenvalues):
        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)
        
        if rank == 0:
            print(round(eigen_state.real, 5))

        eigen_vector.setName("Psi_" + str(m) + "_"  + str(i)) 
        Viewer.view(eigen_vector)
        
        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        energy.setValue(0,eigen_state)
        
        energy.setName("Energy_" + str(m) + "_"  + str(i)) 
        energy.assemblyBegin()
        energy.assemblyEnd()
        Viewer.view(energy)

def Build_Hamiltonian_Second_Order(input_par, grid, m):
    grid_size = grid.size 
    matrix_size = grid_size * (input_par["l_max_bound_state"] + 1)
    nnz = int(input_par["l_max_bound_state"] / 2 + 1) + 4

    h2 = input_par["grid_spacing"]*input_par["grid_spacing"]
    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()

    R_o = input_par["R_o"]

    for i  in range(istart, iend):
        l_block = floor(i/grid_size)
        grid_idx = i % grid_size
        r = grid[grid_idx]
        Hamiltonian.setValue(i, i, 1.0/h2 + H2_Plus_Potential(r, l_block, l_block, m, R_o))
        if grid_idx >=  1:
            Hamiltonian.setValue(i, i-1, (-1.0/2.0)/h2)
        if grid_idx < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-1.0/2.0)/h2)
    
        if l_block % 2 == 0:
            l_prime_list = list(range(0, input_par["l_max_bound_state"] + 1, 2))
        else:
            l_prime_list = list(range(1, input_par["l_max_bound_state"] + 1, 2))

        l_prime_list.remove(l_block)
        for l_prime in l_prime_list:
            col_idx = grid_size*l_prime + grid_idx
            Hamiltonian.setValue(i, col_idx, H2_Plus_Potential(r, l_block, l_prime, m, R_o))

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Build_Hamiltonian_Fourth_Order(input_par, grid, m): 
    grid_size = grid.size 
    matrix_size = grid_size * (input_par["l_max_bound_state"] + 1)
    nnz = int(input_par["l_max_bound_state"] / 2 + 1) + 6
    
    h2 = input_par["grid_spacing"]*input_par["grid_spacing"]
    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()

    R_o = input_par["R_o"]

    for i  in range(istart, iend):
        l_block = floor(i/grid_size)
        grid_idx = i % grid_size
        r = grid[grid_idx]

        Hamiltonian.setValue(i, i, (15.0/ 12.0)/h2 + H2_Plus_Potential(r, l_block, l_block, m, R_o))  
        if grid_idx >=  1:
            Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
        if grid_idx >= 2:
            Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
        if grid_idx < grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
        if grid_idx < grid.size - 2:
            Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)


        if l_block % 2 == 0:
            l_prime_list = list(range(0, input_par["l_max_bound_state"] + 1, 2))
        else:
            l_prime_list = list(range(1, input_par["l_max_bound_state"] + 1, 2))

        l_prime_list.remove(l_block)
        for l_prime in l_prime_list:
            col_idx = grid_size*l_prime + grid_idx
            Hamiltonian.setValue(i, col_idx, H2_Plus_Potential(r, l_block, l_prime, m, R_o))

    for i in np.arange(0, matrix_size, grid_size):
        l_block = floor(i/grid_size)

        Hamiltonian.setValue(i, i, (20.0/24.0)/h2 + H2_Plus_Potential(grid[0], l_block, l_block, m, R_o)) 
        Hamiltonian.setValue(i, i+1, (-6.0/24.0)/h2)
        Hamiltonian.setValue(i, i+2, (-4.0/24.0)/h2)
        Hamiltonian.setValue(i, i+3, (1.0/24.0)/h2) 

        j = i + (grid_size - 1)
        Hamiltonian.setValue(j, j, (20.0/24.0)/h2 + H2_Plus_Potential(grid[grid_size - 1], l_block, l_block, m, R_o)) 
        Hamiltonian.setValue(j, j-1, (-6.0/24.0)/h2)
        Hamiltonian.setValue(j, j-2, (-4.0/24.0)/h2)
        Hamiltonian.setValue(j, j-3, (1.0/24.0)/h2)

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def TISE(input_par):
    
    if rank == 0:
        start_time = time.time()
        print("Calculating Bound States for H2+ \n ")
        print("R_0 = " + str(input_par["R_o"]) + "\n" )
        print()

    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    for m in range(0, input_par["m_max_bound_state"] + 1):
        if rank == 0:
            print("Calculating the B-States for m = " + str(m) + "\n")

        Hamiltonian = eval("Build_Hamiltonian_" + input_par["order"] + "_Order(input_par, grid, m)")
        Eigen_Value_Solver(Hamiltonian, input_par["n_max"], input_par, m, ViewHDF5)
     
        if rank == 0:
            print("Finished calculation for m = " + str(m) + "\n" , "\n")

    if rank == 0:
        total_time = (time.time() - start_time) / 60
        print("Total time taken for calculating Bound States is " + str(round(total_time, 3)))

    ViewHDF5.destroy()

if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    TISE(input_par)
    
