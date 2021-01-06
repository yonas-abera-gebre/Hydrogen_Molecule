if True:
    import sys
    import matplotlib.pyplot as plt
    import matplotlib
    from sympy.physics.wigner import wigner_3j as wigner3j
    from numpy import sin, log, pi, angle, sqrt
    import numpy as np
    import mpmath as mp
    from scipy import special
    from scipy.special import sph_harm


def Potential(r, l_prime, l, m, A=1.0):
    pot_term = 0.0

    if r<=A:
        for L in range(0, l_prime+l + 2, 2):
            pot_term += pow(r,L)/pow(A,L+1)*wigner3j(l_prime,L,l,0,0,0)*wigner3j(l_prime,L,l,-m,0,m)
    else:
        for L in range(0, l_prime+l + 2, 2):
            pot_term += pow(A,L)/pow(r,L+1)*wigner3j(l_prime,L,l,0,0,0)*wigner3j(l_prime,L,l,-m,0,m)

    pot_term *= -2*pow(-1.0,m)*sqrt((2*l+1)*(2*l_prime+1))

    return pot_term

def Coulomb_Fun(grid, lo, k, z=2):
    coulomb_fun = np.zeros(len(grid))
    for i, r in enumerate(grid):
        coulomb_fun[i] = mp.coulombf(lo, -z/k, k*r)

    return coulomb_fun

def Coulomb_Fun_Limit(grid, lo, k, z=2):
    phase = angle(special.gamma(lo + 1 - 1j*z/k))
    return sin(k*grid + (z/k)*log(2*k*grid) - lo*pi/2 + phase)

def Right_Side_Vector(grid, lo, l_prime, mo, k, z=2):
    right_vector = np.zeros(len(grid))

    for i, r in enumerate(grid):
        pot_term = Potential(r, l_prime, lo, mo)

        if l_prime == lo:
            pot_term += 2.0/r

        right_vector[i] = pot_term * mp.coulombf(lo, -z/k, k*r) / k

    return right_vector

def Left_Side_Matrix(grid, lo, l_prime, mo, k):
    l_max = 20
    if lo % 2 == 0:
        l_sum_list = np.arange(0, l_max, 2)
    else:
        l_sum_list = np.arange(1, l_max, 2) 

    num_of_rows = len(grid)
    num_of_cols = len(grid)*len(l_sum_list)

    LSM = np.zeros((num_of_rows, num_of_cols), dtype=float)
    h2 = abs(grid[1] - grid[0])
    h2 = h2*h2

    for i, r in enumerate(grid):
        for j, l in enumerate(l_sum_list):
            col_idx = j*len(grid) + i 

            if l != l_prime:
                LSM[i, col_idx] = -1.0*Potential(r, l_prime, l, mo)
            else:
                if i >=  1:
                    LSM[i, col_idx-1] =  (1.0/2.0)/h2
                if i < grid.size - 1:
                    LSM[i, col_idx+1] =  (1.0/2.0)/h2

                LSM[i, col_idx] = k*k/2.0 - (1.0)/h2 - l*(l+1)/(2*pow(r, 2)) - Potential(r, l_prime, l, mo)
    
    return LSM

def Solver(grid, lo, l_prime, mo, k):
    right_vector = Right_Side_Vector(grid, lo, l_prime, mo, k)
    print("Made the right vector")
    LSM = Left_Side_Matrix(grid, lo, l_prime, mo, k)
    print("Made the LSM")
    R = np.linalg.lstsq(LSM, right_vector, rcond=None)[0]
    print("Solved Ax=b")

    l_label = 0
    for i in range(0, 4*len(grid), len(grid)):
        plt.plot(grid, R[i:i+len(grid)], label="l = " + str(l_label))
        # plt.legend(loc = 'lower right')
        # plt.xlim(0,50)
        plt.tight_layout()
        l_label += 2
    
    plt.legend()
    plt.savefig("Soln6.png")


    plt.clf()
    soln_check = LSM.dot(R)
    plt.plot(grid, soln_check)
    plt.plot(grid, right_vector)
    plt.savefig("Soln_Check6.png")


    plt.clf()
    Chi(grid, R)

def Chi(grid, R):
    phi =0
    theta = 0
    theta, phi = np.meshgrid(theta, phi)
    chi = np.zeros(len(grid))
    l = 0
    m = 0
    for i in range(0, 10*len(grid), len(grid)):
        chi += R[i:i+len(grid)]*sph_harm(m, l, phi, theta)[0][0].real
        l +=2

    plt.clf()
    plt.plot(grid, chi)
    # plt.xlim(0,50)
    plt.savefig("Chi6.png")
    return chi  

if __name__=="__main__":

    grid = np.arange(0.01, 50, 0.01)
    lo = 6
    k = 1
    l_prime = 6
    mo = 0

    # right_vector_0 = Right_Side_Vector(grid, lo, 0, mo, k)
    # right_vector_2 = Right_Side_Vector(grid, lo, 2, mo, k) + 2/grid
    # right_vector_4 = Right_Side_Vector(grid, lo, 4, mo, k)
    # right_vector_6 = Right_Side_Vector(grid, lo, 6, mo, k)
    # right_vector_9 = Right_Side_Vector(grid, lo, 9, mo, k)
    # right_vector_10 = Right_Side_Vector(grid, lo, 10, mo, k)


    # # plt.plot(grid, right_vector_0, label = "l'= 0")
    # plt.plot(grid, right_vector_2, label = "l'= 2")
    # plt.plot(grid, right_vector_4, label = "l'= 4")
    # plt.plot(grid, right_vector_6, label = "l'= 6")
    # plt.plot(grid, right_vector_9, label = "l'= 9")
    # plt.plot(grid, right_vector_10, label = "l'= 10")
    # plt.xlim(0, 10)
    # plt.legend()
    # plt.savefig("RV.png")

    Solver(grid, lo, l_prime, mo, k)

    # l = 0
    # l_prime = 2
    # m = 0
    

    # for L in range(0, l + l_prime + 2, 2):
    #     w = wigner3j(l_prime,L,l,-m,0,m)
    #     print(w)


