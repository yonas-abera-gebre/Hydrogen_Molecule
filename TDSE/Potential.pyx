from math import sqrt

cdef double factorial_new(int x):
    cdef double y = 1.0
    cdef int i
    if x == 0:
        return 1.0
    elif x == 1:
        return 1.0
    for i in range(1, x+1):
        y *= i
    return y

cdef double clebsch(int j1, int j2, int j3, int m1, int m2, int m3):
    cdef int vmin, vmax, v
    cdef double S, C
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    
    if m3 != m1 + m2:
        return 0
    vmin = max([-j1 + j2 + m3, -j1 + m1, 0])
    vmax = min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3])
    
    C = sqrt((2.0 * j3 + 1.0)) * sqrt(factorial_new(j3 + j1 - j2)/factorial_new(j1 + j2 + j3 + 1)) 
    C *= sqrt(factorial_new(j3 - j1 + j2)/factorial_new(j1 - m1))
    C *= sqrt(factorial_new(j1 + j2 - j3)/factorial_new(j1 + m1))
    C *= sqrt(factorial_new(j3 + m3)/factorial_new(j2 - m2)) 
    C *= sqrt(factorial_new(j3 - m3)/factorial_new(j2 + m2))

    S = 0
   
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial_new(v) * \
            (factorial_new(j2 + j3 + m1 - v)/factorial_new(j3 - j1 + j2 - v)) * (factorial_new(j1 - m1 + v) / factorial_new(j3 + m3 - v))/\
            factorial_new(v + j1 - j2 - m3)
    
    C = C * S
    return C

cdef double wigner3j(int j1, int j2, int j3, int m1, int m2, int m3):
    cdef double ret_val
    ret_val = clebsch(j1, j2, j3, m1, m2, -m3)
    ret_val *= (-1.)**(-j1+j2+m3)
    ret_val *= (2*j3+1)**(-1/2.)
    return ret_val



cpdef double H2_Plus_Potential(double r, int l, int l_prime, int m, double R_o):
    R_o = R_o / 2.0
    
    cdef double potential_value = 0.0
    
    if abs(m) > l or abs(m) > l_prime:
        if l == l_prime:
            return 0.5*l*(l+1)*pow(r,-2) 
        else:
            return 0.0
            
    if r <= R_o:
        for lamda in range(abs(l-l_prime), l + l_prime + 2, 2):
            coef = wigner3j(l,lamda,l_prime,0,0,0) * wigner3j(l,lamda,l_prime,-m,0,m)
            potential_value += pow(r, lamda)/pow(R_o, lamda + 1) * coef   
    else:
         for lamda in range(abs(l-l_prime), l + l_prime + 2, 2):
            coef = wigner3j(l,lamda,l_prime,0,0,0) * wigner3j(l,lamda,l_prime,-m,0,m)
            potential_value += pow(R_o,lamda)/pow(r,lamda + 1) * coef

    potential_value = -2.0 * pow(-1.0, m)* sqrt((2.0*l+1.0)*(2.0*l_prime+1.0)) * potential_value 

    if l == l_prime:
        potential_value += 0.5*l*(l+1)*pow(r,-2)

    return potential_value

