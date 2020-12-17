if True:
    import sys
    import matplotlib.pyplot as plt
    import matplotlib
    from sympy.physics.wigner import wigner_3j as wigner3j
    from numpy import sin, log, pi, angle
    import numpy as np
    import mpmath as mp
    from scipy import special

def Ele_Ele_Int(r, l, l_prime, m, R_o):
    R_o = R_o / 2.0
    
    potential_value = 0.0
    
    if abs(m) > l or abs(m) > l_prime:
        return 0.0
     
    if r <= R_o:
        for lamda in range(0, l + l_prime + 2, 2):
            coef = wigner3j(l,lamda,l_prime,0,0,0) * wigner3j(l,lamda,l_prime,-m,0,m)
            potential_value += pow(r, lamda)/pow(R_o, lamda + 1) * coef   
    else:
         for lamda in range(0, l + l_prime + 2, 2):
            coef = wigner3j(l,lamda,l_prime,0,0,0) * wigner3j(l,lamda,l_prime,-m,0,m)
            potential_value += pow(R_o,lamda)/pow(r,lamda + 1) * coef

    potential_value = -2.0 * pow(-1.0, m)* np.sqrt((2.0*l+1.0)*(2.0*l_prime+1.0)) * potential_value 

    return potential_value

def Coulomb_Function_Limit(r, lo, k, Z=2):
    phase = angle(special.gamma(lo + 1 - 1J*Z/k))
    return sin(k*r + (Z/k)*log(2*k*r) - lo*pi/2 + phase)

def Coulomb_Function(grid, lo, k, Z=2):
    cf = np.zeros(len(grid))
    for i, r in enumerate(grid):
        cf[i] = mp.coulombf(lo, -1*Z/k, k*r)

    return cf

def Right_Side(grid, lo, mo, l_prime, k, R0):
    col_fun = Coulomb_Function(grid, lo, k)
    Right_Vec =  np.zeros(len(grid))

    for i, r in enumerate(grid):
        pot = Ele_Ele_Int(r, l_prime, lo, mo, R0)

        if l_prime == lo:
            pot += 2/r

        Right_Vec[i]  = col_fun[i] * pot / k

    return Right_Vec

def Left_Side_Matrix(grid, lo, mo, l_prime, k, R0):
    grid_size = grid.size 
    l_max = 20

    if lo%2 == 0:
        l_list = list(range(0, l_max, 2))
    else:
        l_list = list(range(1, l_max, 2))

    matrix_size_row = grid_size 
    matrix_size_col = grid_size*len(l_list)

    h2 = abs(grid[1] - grid[0])
    h2 = h2*h2
    LSM = np.zeros((matrix_size_row, matrix_size_col), dtype=float)


    for i, r in enumerate(grid):
        for j, l in enumerate(l_list):
            col_idx = i + j*grid_size

            if l != l_prime:
                LSM[i, col_idx] =  -1*Ele_Ele_Int(r,l_prime, l, mo, R0)
            else:
                
                if i >=  1:
                    LSM[i, col_idx-1] =  (1.0/2.0)/h2
                if i < grid_size - 1:
                    LSM[i, col_idx+1] =  (1.0/2.0)/h2

                diag_ele = k*k/2 - 1/h2 - 0.5*l*(l+1)*pow(r,-2) - Ele_Ele_Int(r, l, l, mo, R0)
                LSM[i, col_idx] =  diag_ele

    return LSM

def Solve_R_Vector(grid, lo, mo, l_prime, k, R0):
    
    Right_Vec = Right_Side(grid, lo, mo, l_prime, k, R0)
    print("Made the right vector")
    LSM = Left_Side_Matrix(grid, lo, mo, l_prime, k, R0)
    print("Made the LSM")
    R = np.linalg.lstsq(LSM, Right_Vec, rcond=None)[0]
    print("Solved Ax=b")
    Plot_Result(grid, LSM, Right_Vec, R, lo)
    print("Plotted results")
    return R

def Plot_Result(grid, LSM, Right_Vec, soln, lo):
    if lo%2 == 0:
        l_list = list(range(0, 20, 2))
    else:
        l_list = list(range(1, 20, 2))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(np.absolute(LSM[:,list(range(5*len(grid)))]))#, norm=matplotlib.colors.LogNorm())
    ax.set_xticks(np.arange(0.5*len(grid), 5*len(grid), len(grid)))
    xlabel = l_list
    ax.set_xticklabels(xlabel)
    plt.colorbar(orientation='vertical')
    plt.tight_layout()
    plt.savefig("Left_Side_Matrix.png")
    plt.clf()

    plt.plot(grid, Right_Vec)
    plt.tight_layout()
    # plt.xlim(0,10)
    plt.savefig("Right_Vector.png")
    plt.clf()

    
    count = 0 
    for i in range(0, 5*len(grid), len(grid)):
        norm = np.linalg.norm(soln[i:i+len(grid)])
        plt.plot(grid, soln[i:i+len(grid)], label = str(l_list[count]))
        count += 1

    plt.legend(loc = 'lower right')
    plt.tight_layout()
    plt.xlim(0,25)
    # plt.title("l_prime = 3")
    plt.tight_layout()
    plt.savefig("Soln.png")
    plt.clf()


if __name__=="__main__":

    grid = np.arange(0.1, 100+0.1, 0.1)

    lo = 5
    l_prime = 3
    mo = 0
    k = 0.5
    Ro = 0.5

    Solve_R_Vector(grid, lo, mo, l_prime, k, Ro)

    

