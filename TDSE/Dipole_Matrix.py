if True:
    import numpy as np
    import sys
    from numpy import pi
    from math import floor
    import json
    from sympy.physics.wigner import gaunt, wigner_3j
    import Module as Mod 
    import Potential as Pot
    
         
if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


def Dipole_Z_Matrix(input_par):
    index_map_l_m, index_map_box = Mod.Index_Map_M_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    matrix_size = grid.size * len(index_map_l_m)
    h = abs(grid[1] - grid[0]) 

    Dipole_Matrix = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Matrix.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid.size)][1]
        m_block = index_map_l_m[floor(i/grid.size)][0]
        grid_idx = i % grid.size 
        
        if l_block < input_par["l_max"]:
            columon_idx = grid.size*index_map_box[(m_block, l_block + 1)] + grid_idx
            CG_Coeff = (l_block+1) / np.sqrt((2*l_block+1)*(2*l_block+3))
            Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)
            
        if abs(m_block) < l_block and l_block > 0:
            columon_idx = grid.size*index_map_box[(m_block, l_block - 1)] + grid_idx
            CG_Coeff = (l_block) / np.sqrt((2*l_block-1)*(2*l_block+1))
            Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)
           
    Dipole_Matrix.assemblyBegin()
    Dipole_Matrix.assemblyEnd()
    return Dipole_Matrix

def Dipole_X_Matrix(input_par):
    index_map_l_m, index_map_box = Mod.Index_Map_M_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    matrix_size = grid.size * len(index_map_l_m)
    h = abs(grid[1] - grid[0]) 

    with open(sys.path[0] + "/wigner_3j.json") as file:
        wigner_3j_dict = json.load(file)

    Dipole_Matrix = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Matrix.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid.size)][1]
        m_block = index_map_l_m[floor(i/grid.size)][0]
        grid_idx = i % grid.size 
        
        if input_par["m_max"] == 0:
            Dipole_Matrix.setValue(i,i,0.0)
            continue

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            factor = pow(-1.0, m_block)*np.sqrt((2*l_block+1)*(2*l_prime+1)/2)*wigner_3j_dict[str((l_block,0,l_prime,0,1,0))]

            m_prime = m_block - 1
            CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
            columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
            Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

            m_prime = m_block + 1
            CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
            columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
            Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

        if l_block > 0:
            l_prime = l_block - 1
            factor = pow(-1.0, m_block)*np.sqrt((2*l_block+1)*(2*l_prime+1)/2)*wigner_3j_dict[str((l_block,0,l_prime,0,1,0))]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
                columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
                Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

            if m_block < l_prime:
                m_prime = m_block + 1
                CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
                columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
                Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

    Dipole_Matrix.assemblyBegin()
    Dipole_Matrix.assemblyEnd()
    return Dipole_Matrix

def Dipole_Y_Matrix(input_par):
    index_map_l_m, index_map_box = Mod.Index_Map_M_Block(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])
    matrix_size = grid.size * len(index_map_l_m)
    h = abs(grid[1] - grid[0]) 

    with open(sys.path[0] + "/wigner_3j.json") as file:
        wigner_3j_dict = json.load(file)

    Dipole_Matrix = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Matrix.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = index_map_l_m[floor(i/grid.size)][1]
        m_block = index_map_l_m[floor(i/grid.size)][0]
        grid_idx = i % grid.size 
        
        if input_par["m_max"] == 0:
            Dipole_Matrix.setValue(i,i,0.0)
            continue
        
        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            factor = 1.0j*pow(-1.0, m_block)*np.sqrt((2*l_block+1)*(2*l_prime+1)/2)* wigner_3j_dict[str((l_block,0,l_prime,0,1,0))]

            m_prime = m_block - 1
            CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
            columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
            Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

            m_prime = m_block + 1
            CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
            columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
            Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

        if l_block > 0:
            l_prime = l_block - 1
            factor = 1.0j*pow(-1.0, m_block)*np.sqrt((2*l_block+1)*(2*l_prime+1)/2)* wigner_3j_dict[str((l_block,0,l_prime,0,1,0))]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
                columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
                Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

            if m_block < l_prime:
                m_prime = m_block + 1
                CG_Coeff = factor*(wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l_block,-1*m_block,l_prime,m_prime,1,1))])
                columon_idx = grid.size*index_map_box[(m_prime, l_prime)] + grid_idx
                Dipole_Matrix.setValue(i, columon_idx, grid[grid_idx]*CG_Coeff)

    Dipole_Matrix.assemblyBegin()
    Dipole_Matrix.assemblyEnd()
    return Dipole_Matrix

