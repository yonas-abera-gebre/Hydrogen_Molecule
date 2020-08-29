import numpy as np
from sympy.physics.wigner import gaunt, wigner_3j
import json
import H2_Module as Mod 
import matplotlib.pyplot as plt

def Nucleus_Electron_Interaction(grid, l, l_prime, m, R_o):

    R_o = R_o / 2.0
    R_o_idx =  np.nonzero(grid > R_o)[0][0]
    
    grid_low = grid[:R_o_idx]
    grid_high = grid[R_o_idx:]

    potential_low = np.zeros(len(grid_low))
    potential_high = np.zeros(len(grid_high))

    for lamda in range(0, l + l_prime + 2, 2):
        coef = float(wigner_3j(l,lamda,l_prime,0,0,0)) * float(wigner_3j(l,lamda,l_prime,-m,0,m))
        potential_low +=  np.power(grid_low, lamda)/pow(R_o, lamda+1) * coef
        potential_high +=  pow(R_o, lamda) / np.power(grid_high, lamda + 1) * coef
 
    potential = -2.0 * pow(-1.0, m)* np.sqrt((2.0*l+1.0)*(2.0*l_prime+1.0)) * np.concatenate((potential_low, potential_high)) 
    
    if l == l_prime:
        potential += 0.5*l*(l+1)*np.power(grid, -2.0)

    return list(potential)

def H2_Plus_Potential(r, l, l_prime, m, R_o):
    R_o = R_o / 2.0
    potential_value = 0.0
    if r <= R_o:
        for lamda in range(0, l + l_prime + 2, 2):
            coef = float(wigner_3j(l,lamda,l_prime,0,0,0)) * float(wigner_3j(l,lamda,l_prime,-m,0,m))
            potential_value += pow(r, lamda)/pow(R_o, lamda + 1) * coef   
    else:
         for lamda in range(0, l + l_prime + 2, 2):
            coef = float(wigner_3j(l,lamda,l_prime,0,0,0)) * float(wigner_3j(l,lamda,l_prime,-m,0,m))
            potential_value += pow(R_o,lamda)/pow(r,lamda + 1) * coef

    potential_value = -2.0 * pow(-1.0, m)* np.sqrt((2.0*l+1.0)*(2.0*l_prime+1.0)) * potential_value 

    return potential_value

def Potential(input_par):
    pot_dict = {}
    for m in range(input_par["m_max"] + 1):
        for l in range(input_par["l_max"] + 1):
            if l % 2 == 0:
                l_prime_list = range(0, input_par["l_max"] + 1, 2)
            else:
                l_prime_list = range(1, input_par["l_max"] + 1, 2)

            for l_prime in l_prime_list:
                pot_dict[str((m, l, l_prime))] = Nucleus_Electron_Interaction(grid, l, l_prime, m, input_par["R_o"])
   
    with open("Nuclear_Electron_Int.json", 'w') as file:
        json.dump(pot_dict, file)


if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    Potential(input_par)

    