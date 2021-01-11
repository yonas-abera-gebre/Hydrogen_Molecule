if True:
    import numpy as np
    import sys
    from numpy import pi
    from math import floor
    import json
    from sympy.physics.wigner import gaunt, wigner_3j
    import Module as Mod 
    
         
if True:
    import petsc4py
    from petsc4py import PETSc
    petsc4py.init(sys.argv)
    petsc4py.init(comm=PETSc.COMM_WORLD)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    
def X_and_Y_Matrix_Coeff_Calculator(input_par):

    Dip_Acc_X_Coeff = {}
    Dip_Acc_Y_Coeff = {}
    
    cdef int l, m, m_prime, l_prime
    
    with open(sys.path[0] + "/wigner_3j.json") as file:
        wigner_3j_dict = json.load(file)


    for l in range(input_par["l_max"]+1):
        
        for m in range(-1*l, l+1):
        
            l_prime = l + 1
            m_prime = m + 1
            
            Dip_Acc_X_Coeff[l,m,l_prime, m_prime] = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)*wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])   
            Dip_Acc_Y_Coeff[l,m,l_prime, m_prime] =  1.0j*pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)* wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])
            
            m_prime = m - 1
            
            Dip_Acc_X_Coeff[l,m,l_prime, m_prime] = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)*wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])
            Dip_Acc_Y_Coeff[l,m,l_prime, m_prime] =  1.0j*pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)* wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])
            
            if l > 0:
                l_prime = l - 1
                if -1*m < l_prime:
                    m_prime = m - 1
                
                    Dip_Acc_X_Coeff[l,m,l_prime, m_prime] = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)*wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])
                    Dip_Acc_Y_Coeff[l,m,l_prime, m_prime] =  1.0j*pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)* wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])
            
                if m < l_prime:
                    m_prime = m + 1
                    
                    Dip_Acc_X_Coeff[l,m,l_prime, m_prime] = pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)*wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] - wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])
                    Dip_Acc_Y_Coeff[l,m,l_prime, m_prime] =  1.0j*pow(-1.0, m)*np.sqrt((2*l+1)*(2*l_prime+1)/2)* wigner_3j_dict[str((l,0,l_prime,0,1,0))]*(wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,-1))] + wigner_3j_dict[str((l,-1*m,l_prime,m_prime,1,1))])

           
    
    return Dip_Acc_X_Coeff, Dip_Acc_Y_Coeff

def Z_Matrix_Coeff_Calculator(input_par):
    Dip_Acc_Z_Upper_Coeff = {}
    Dip_Acc_Z_Lower_Coeff = {}

    cdef int l, m

    with open(sys.path[0] + "/wigner_3j.json") as file:
        wigner_3j_dict = json.load(file)

    for l in range(input_par["l_max"]+1):
        
        for m in range(-1*l, l+1):
            Dip_Acc_Z_Upper_Coeff[l,m] = (l+1)/np.sqrt((2*l+1)*(2*l+3))
            if abs(m) < l and l > 0:
                Dip_Acc_Z_Lower_Coeff[l,m] = (l)/np.sqrt((2*l-1)*(2*l+1))
        
    return Dip_Acc_Z_Upper_Coeff, Dip_Acc_Z_Lower_Coeff

def Dipole_Acceleration_Z_Matrix(input_par):

    cdef int l_block, m_block, grid_size, grid_idx, columon_idx

    index_map_m_l, index_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_m_l)

    Dipole_Acceleration_Matrix = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Acceleration_Matrix.getOwnershipRange() 
   
    Dip_Acc_Z_Upper_Coeff, Dip_Acc_Z_Lower_Coeff  = Z_Matrix_Coeff_Calculator(input_par)
    grid2 = np.power(grid, 2.0)
    for i in range(istart, iend):
        l_block = index_map_m_l[floor(i/grid_size)][1]
        m_block = index_map_m_l[floor(i/grid_size)][0]
        grid_idx = i % grid_size 
        
        if l_block < input_par["l_max"]:
            columon_idx = grid_size*index_map_box[(m_block, l_block + 1)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_Z_Upper_Coeff[l_block,m_block]/grid2[grid_idx])
            
        if abs(m_block) < l_block and l_block > 0:
            columon_idx = grid_size*index_map_box[(m_block, l_block - 1)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_Z_Lower_Coeff[l_block,m_block]/grid2[grid_idx])
           
    Dipole_Acceleration_Matrix.assemblyBegin()
    Dipole_Acceleration_Matrix.assemblyEnd()
    return Dipole_Acceleration_Matrix

def Dipole_Acceleration_X_Matrix(input_par):
    
    cdef int l_block, m_block, grid_size, columon_idx, grid_idx

    index_map_m_l, index_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_m_l)

    Dipole_Acceleration_Matrix = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Acceleration_Matrix.getOwnershipRange() 
    
    Dip_Acc_X_Coeff, Dip_Acc_Y_Coeff  = X_and_Y_Matrix_Coeff_Calculator(input_par)
    grid2 = np.power(grid, 2.0)

    for i in range(istart, iend):
        l_block = index_map_m_l[floor(i/grid_size)][1]
        m_block = index_map_m_l[floor(i/grid_size)][0]
        grid_idx = i % grid_size 
        
        if input_par["m_max"] == 0:
            Dipole_Acceleration_Matrix.setValue(i,i,0.0)
            continue

        if l_block < input_par["l_max"]:

            columon_idx = grid_size*index_map_box[(m_block - 1, l_block + 1)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_X_Coeff[l_block,m_block,l_block + 1, m_block - 1]/grid2[grid_idx] )

            columon_idx = grid_size*index_map_box[(m_block + 1, l_block + 1)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_X_Coeff[l_block,m_block,l_block + 1, m_block + 1]/grid2[grid_idx]  )

        if l_block > 0:
            if -1*m_block < l_block - 1:
                columon_idx = grid_size*index_map_box[(m_block - 1, l_block - 1)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_X_Coeff[l_block,m_block,l_block - 1, m_block - 1]/grid2[grid_idx] )

            if m_block < l_block - 1:
                columon_idx = grid_size*index_map_box[(m_block + 1, l_block - 1)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_X_Coeff[l_block,m_block,l_block - 1, m_block + 1]/grid2[grid_idx] )

    Dipole_Acceleration_Matrix.assemblyBegin()
    Dipole_Acceleration_Matrix.assemblyEnd()
    return Dipole_Acceleration_Matrix

def Dipole_Acceleration_Y_Matrix(input_par):

    cdef int l_block, m_block, grid_size, columon_idx, grid_idx

    index_map_m_l, index_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])
    grid_size = grid.size
    matrix_size = grid_size * len(index_map_m_l)

    Dipole_Acceleration_Matrix = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Acceleration_Matrix.getOwnershipRange() 
    
    Dip_Acc_X_Coeff, Dip_Acc_Y_Coeff  = X_and_Y_Matrix_Coeff_Calculator(input_par)
    grid2 = np.power(grid, 2)
    for i in range(istart, iend):
        l_block = index_map_m_l[floor(i/grid_size)][1]
        m_block = index_map_m_l[floor(i/grid_size)][0]
        grid_idx = i % grid_size 
        
        if input_par["m_max"] == 0:
            Dipole_Acceleration_Matrix.setValue(i,i,0.0)
            continue

        if l_block < input_par["l_max"]:
    
            columon_idx = grid_size*index_map_box[(m_block - 1, l_block + 1)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_Y_Coeff[l_block,m_block,l_block + 1,  m_block - 1]/grid2[grid_idx])

            m_prime = m_block + 1
            columon_idx = grid_size*index_map_box[(m_block + 1, l_block + 1)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_Y_Coeff[l_block,m_block,l_block + 1, m_block + 1]/grid2[grid_idx])

        if l_block > 0:
            if -1*m_block < l_block - 1:
                columon_idx = grid_size*index_map_box[(m_block - 1, l_block - 1)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_Y_Coeff[l_block,m_block,l_block - 1, m_block - 1]/grid2[grid_idx])

            if m_block < l_block - 1:
                columon_idx = grid_size*index_map_box[(m_block + 1, l_block - 1)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(i, columon_idx, Dip_Acc_Y_Coeff[l_block,m_block,l_block - 1, m_block + 1]/grid2[grid_idx])

    Dipole_Acceleration_Matrix.assemblyBegin()
    Dipole_Acceleration_Matrix.assemblyEnd()
    return Dipole_Acceleration_Matrix

