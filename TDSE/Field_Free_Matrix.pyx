if True:
    import numpy as np
    import sys
    from numpy import pi
    from math import floor
    import H2_Module as Mod 
    from Potential import H2_Plus_Potential
         
if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


def Build_FF_Hamiltonian_Second_Order(input_par):

    cdef double h2
    cdef int grid_size, l_blk, grid_idx, ECS_idx, m_blk, l_prime
    
    h2 = input_par["grid_spacing"] * input_par["grid_spacing"]
    index_map_m_1, index_map_box = Mod.Index_Map_M_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_box)

    if input_par["ECS_region"] < 1.00 and input_par["ECS_region"] > 0.00:
        ECS_idx = np.where(grid > grid[-1] * input_par["ECS_region"])[0][0]
    elif input_par["ECS_region"] == 1.00:
        ECS_idx = grid_size
        if rank == 0:
            print("No ECS applied for this run \n")
    else:
        if rank == 0:
                print("ECS region has to be between 0.0 and 1.00\n")
                exit() 

    nnz = int((input_par["l_max"] + 1)/2) + 4
    FF_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = FF_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        grid_idx = i % grid_size 
        m_blk = index_map_m_1[floor(i/grid_size)][0]
        l_blk = index_map_m_1[floor(i/grid_size)][1]

        if grid_idx < ECS_idx:
            FF_Hamiltonian.setValue(i, i, 1.0/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_idx], -2.0) + H2_Plus_Potential(grid[grid_idx], l_blk, l_blk, m_blk, input_par["R_o"]))
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, -0.5/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, -0.5/h2)

        elif grid_idx > ECS_idx:
            FF_Hamiltonian.setValue(i, i,  -1.0j/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_idx], -2.0) + H2_Plus_Potential(grid[grid_idx], l_blk, l_blk, m_blk, input_par["R_o"]))
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, 0.5j/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, 0.5j/h2)

        else:
            FF_Hamiltonian.setValue(i, i, np.exp(-1.0j*pi/4.0)/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_idx], -2.0) + H2_Plus_Potential(grid[grid_idx], l_blk, l_blk, m_blk, input_par["R_o"]))
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, -1/(1 + np.exp(1.0j*pi/4.0))/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, -1*np.exp(-1.0j*pi/4.0)/ (1+np.exp(1.0j*pi/4.0))/h2)

        if l_blk % 2 == 0:
            l_prime_list = list(range(0, input_par["l_max"] + 1, 2))
        else:
            l_prime_list = list(range(1, input_par["l_max"] + 1, 2))

        l_prime_list.remove(l_blk)

        
        for l_prime in l_prime_list:
            if abs(m_blk) > l_prime:
                continue
            col_idx = grid_size*index_map_box[(m_blk, l_prime)] + grid_idx
            FF_Hamiltonian.setValue(i, col_idx, H2_Plus_Potential(grid[grid_idx], l_blk, l_prime, m_blk, input_par["R_o"]))

    FF_Hamiltonian.assemblyBegin()
    FF_Hamiltonian.assemblyEnd()
    return FF_Hamiltonian

def Build_FF_Hamiltonian_Fourth_Order(input_par):

    cdef float h2
    cdef int grid_size, ECS, l_blk, grid_idx, ECS_idx

    h2 = input_par["grid_spacing"] * input_par["grid_spacing"]
    index_map_m_1, index_map_box = Mod.Index_Map_M_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_box)

    if input_par["ECS_region"] < 1.00 and input_par["ECS_region"] > 0.00:
        ECS_idx = np.where(grid > grid[-1] * input_par["ECS_region"])[0][0]
    elif input_par["ECS_region"] == 1.00:
        ECS_idx = grid_size
        if rank == 0:
            print("No ECS applied for this run \n")
    else:
        if rank == 0:
            print("ECS region has to be between 0.0 and 1.00\n")
            exit() 

    def Fourth_Order_Stencil():
        x_2 = 0.25*(2j - 3*np.exp(3j*pi/4)) / (1 + 2j + 3*np.exp(1j*pi/4))
        x_1 = (-2j + 6*np.exp(3j*pi/4)) / (2 + 1j + 3*np.exp(1j*pi/4))
        x = 0.25*(-2 + 2j - 9*np.exp(3j*pi/4))
        x__1 = (2 + 2j - 6*np.sqrt(2)) / (3 + 1j + 3*np.sqrt(2))
        x__2 = 0.25*(-2 -2j + 3*np.sqrt(2)) / (3 - 1j + 3*np.sqrt(2))
        return (x__2, x__1, x, x_1, x_2)
    
    ECS_Stencil = Fourth_Order_Stencil()

    nnz = int((input_par["l_max"] + 1)/2) + 6

    FF_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = FF_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        grid_idx = i % grid_size 
        m_blk = index_map_m_1[floor(i/grid_size)][0]
        l_blk = index_map_m_1[floor(i/grid_size)][1]
 
        if grid_idx < ECS_idx:
            FF_Hamiltonian.setValue(i, i, (15.0/ 12.0)/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_idx], -2.0) + H2_Plus_Potential(grid[grid_idx], l_blk, l_blk, m_blk, input_par["R_o"]))
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
            if grid_idx < grid_size - 2:
                FF_Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)
        
        if grid_idx == ECS_idx:
            FF_Hamiltonian.setValue(i, i, ECS_Stencil[2]/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_idx], -2.0) + H2_Plus_Potential(grid[grid_idx], l_blk, l_blk, m_blk, input_par["R_o"]))
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, ECS_Stencil[1]/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, ECS_Stencil[0]/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, ECS_Stencil[3]/h2)
            if grid_idx < grid_size - 2:
                FF_Hamiltonian.setValue(i, i+2, ECS_Stencil[4]/h2)

        if grid_idx > ECS_idx:
            FF_Hamiltonian.setValue(i, i, (15.0/ 12.0)* -1.0j/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_idx], -2.0) + H2_Plus_Potential(grid[grid_idx], l_blk, l_blk, m_blk, input_par["R_o"]))
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, (-2.0/3.0) * -1.0j/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, (1.0/24.0) * -1.0j/h2)
            if grid_idx < grid_size - 1:
                FF_Hamiltonian.setValue(i, i+1, (-2.0/3.0) * -1.0j/h2)
            if grid_idx < grid_size - 2:
                FF_Hamiltonian.setValue(i, i+2, (1.0/24.0) * -1.0j/h2)


        if l_blk % 2 == 0:
            l_prime_list = list(range(0, input_par["l_max"] + 1, 2))
        else:
            l_prime_list = list(range(1, input_par["l_max"] + 1, 2))

        l_prime_list.remove(l_blk)

        
        for l_prime in l_prime_list:
            if abs(m_blk) > l_prime:
                continue
            col_idx = grid_size*index_map_box[(m_blk, l_prime)] + grid_idx
            FF_Hamiltonian.setValue(i, col_idx, H2_Plus_Potential(grid[grid_idx], l_blk, l_prime, m_blk, input_par["R_o"]))


    for i in np.arange(0, matrix_size, grid_size):
        l_blk = index_map_m_1[floor(i/grid_size)][1]
        m_blk = index_map_m_1[floor(i/grid_size)][0]

        FF_Hamiltonian.setValue(i, i, (20.0/24.0)/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[0], -2.0) + H2_Plus_Potential(grid[0], l_blk, l_blk, m_blk, input_par["R_o"]))
        FF_Hamiltonian.setValue(i, i+1, (-6.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+2, (-4.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+3, (1.0/24.0)/h2)

        j = i + (grid_size - 1)
        FF_Hamiltonian.setValue(j,j, (20.0/24.0) * -1.0j/h2 + 0.5*l_blk*(l_blk+1)*pow(grid[grid_size - 1], -2.0) + H2_Plus_Potential(grid[grid_size - 1], l_blk, l_blk, m_blk, input_par["R_o"]))
        FF_Hamiltonian.setValue(j,j - 1, (-6.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 2, (-4.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 3, (1.0/24.0) * -1.0j/h2)

    
    FF_Hamiltonian.assemblyBegin()
    FF_Hamiltonian.assemblyEnd()
    return FF_Hamiltonian
if __name__=="__main__":
    input_par = Mod.Input_File_Reader(input_file = "input.json")
    FF_Hamiltonian =  Build_FF_Hamiltonian_Second_Order(input_par)