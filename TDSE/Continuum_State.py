if True:
    import sys
    import H2_Module as Mod 
    import matplotlib.pyplot as plt

    from Potential import Ele_Ele_Interaction as EEI

    from numpy import sin, log, pi, angle
    import numpy as np
    from scipy import special

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


def Coulomb_Function(r, l, k, Z = 2):
    phase = angle(special.gamma(l + 1 - 1J*Z/k))
    return sin(k*r + (Z/k)*log(2*k*r) - l*pi/2 + phase)

def Right_Side(grid, lo, mo, l_prime, k, R0):
    col_fun = Coulomb_Function(grid, lo, k)
    Right_Vec =  PETSc.Vec().createMPI(len(grid), comm=PETSc.COMM_WORLD)

    for i, r in enumerate(grid):
        pot = EEI(r,l_prime, lo, mo, R0)
     
        if l_prime == lo:
            pot += 2/r

        Right_Vec.setValue(i, col_fun[i] * pot / k)

    Right_Vec.assemblyBegin()
    Right_Vec.assemblyEnd()

    return Right_Vec

def Left_Side_Matrix(grid, lo, mo, l_prime, k, R0):
    grid_size = grid.size 
    l_max = 3
    matrix_size_row = grid_size 
    matrix_size_col = grid_size * l_max

    nnz = l_max* 4

    h2 = abs(grid[1] - grid[0])
    h2 = h2*h2
    LSM = PETSc.Mat().createAIJ([matrix_size_row, matrix_size_col], nnz=nnz, comm=PETSc.COMM_WORLD)

    istart, iend = LSM.getOwnershipRange()

    for i  in range(istart, iend):
        r = grid[i]
        for l in range(l_max):
            col_idx = i + l*grid_size

            if l != l_prime:
                LSM.setValue(i, col_idx, -1*EEI(r,l_prime, l, mo, R0))
            
            else:
                if i >=  1:
                    LSM.setValue(i, col_idx-1, (1.0/2.0)/h2)
                if i < grid_size - 1:
                    LSM.setValue(i, col_idx+1, (1.0/2.0)/h2)

                diag_ele = k*k/2 - 1/h2 - 0.5*l*(l+1)*pow(r,-2) - EEI(r, l, l, mo, R0)
                LSM.setValue(i, col_idx, diag_ele)

    LSM.assemblyBegin()
    LSM.assemblyEnd()
    return LSM

def AXEB_Solver(Left_Side_Matrix, Right_Side_Vec):

    R_Vec = PETSc.Vec().createMPI(Left_Side_Matrix.getSize()[1], comm=PETSc.COMM_WORLD)
    R_Vec.assemblyBegin()
    R_Vec.assemblyEnd()
    
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    
    # ksp.setOptionsPrefix("prop_")

    LSMT = PETSc.Mat.transpose(Left_Side_Matrix)
    LSMT.axpy(1.0, LSM)
    ksp.setType(PETSc.KSP.Type.LSQR)
    ksp.setOperators(Left_Side_Matrix, LSMT)
    
    # ksp.setTolerances(1.e-12, PETSc.DEFAULT, PETSc.DEFAULT, PETSc.DEFAULT)
  


    print(Left_Side_Matrix.getSize())
    print(R_Vec.getSize())
    print(Right_Side_Vec.getSize())

    ksp.solve(Right_Side_Vec, R_Vec)

    return R_Vec

if __name__=="__main__":
    grid = Mod.Make_Grid(0, 5, 0.1)
    Right_Vec = Right_Side(grid, 0, 0, 0, 1, 2)

    LSM = Left_Side_Matrix(grid, 0, 0, 0, 1, 2)

    
    R_Vec = AXEB_Solver(LSM, Right_Vec)

    exit()

    plt.plot(grid, Right_Vec)
    plt.xlim(0,5)
    plt.savefig("cont_state.png")