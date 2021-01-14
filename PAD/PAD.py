if True:
    import sys
    import json 
    import matplotlib.pyplot as plt
    import matplotlib
    from numpy import sin, log, pi, angle, sqrt
    import numpy as np
    import mpmath as mp
    from math import floor
    from scipy.special import sph_harm
    from scipy import special
    from os.path import expanduser
    path = expanduser("~/Research/Hydrogen_Molecule/TDSE")
    sys.path.append(path)
    import Module as Mod


def Cont_State_Save(l_array, k_array, grid):
    z = 2
    CS = {}
    for l in l_array:
        for k in k_array:
            print(l, k)
            coulomb_fun = np.zeros(len(grid))
            for i, r in enumerate(grid):
                coulomb_fun[i] = mp.coulombf(l, -z/k, k*r)

            key = str((l, float(k)))          
            CS[key] = list(coulomb_fun)

    # with open("CSD.json", 'w') as file:
    #     json.dump(CS, file)
    # print("Finished")

def Cont_State_Read():
    with open("CSA.json") as file:
        CSA = json.load(file)
    with open("CSB.json") as file:
        CSB = json.load(file)
    with open("CSC.json") as file:
        CSC = json.load(file)
    with open("CSD.json") as file:
        CSD = json.load(file)

    CSB.update(CSA)
    CSC.update(CSB)
    CSD.update(CSC)
    CS  = {}

    for key in CSD.keys():
        l_low = key.index("(")
        l_high = key.index(",")
        l = key[l_low+1:l_high]
        k_low = key.index(",")
        k_high = key.index(")")
        k = float(key[k_low+1:k_high])
        k = round(k, 3)
        name = str(l) + str(k)
        CS[name] = CSD[key]

 
    return CS

def Coulomb_Fun(grid, lo, k, z=2):
    coulomb_fun = np.zeros(len(grid))
    for i, r in enumerate(grid):
        coulomb_fun[i] = mp.coulombf(lo, -z/k, k*r)

    return coulomb_fun

def Coulomb_Fun_Limit(grid, lo, k, z=2):
    phase = angle(special.gamma(lo + 1 - 1j*z/k))
    return sin(k*grid + (z/k)*log(2*k*grid) - lo*pi/2 + phase)

def Proj_Bound_States_Out(input_par, Psi, bound_states):
    
    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"])

    for key in idx_map_box:
        m, l  = key[0], key[1]
        if  l > input_par["l_max_bound_state"]:
            continue
       
        l_idx = int(grid.size*l)
        for n in range(input_par["n_max"]):
            bound_states_l_comp = bound_states[(m, n)][l_idx:l_idx+grid.size]
            Psi[key] -= np.sum(bound_states_l_comp.conj()*Psi[key])*bound_states_l_comp
        
    return Psi

def Coefficent_Calculator(input_par, k_array, CS, Psi, z=2):
    COEF = {}

    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)

    for key in idx_map_box:
        m, l  = key[0], key[1]
        if l == 35:
            continue

        COEF_Minor = {}
       
        for k in k_array:
            k = round(k, 3)
            name = str(l) + str(k)
            CF = np.array(CS[name])
            phase = angle(special.gamma(l + 1 - 1j*z/k))
            coef = np.exp(-1.j*phase)* 1.j**l *np.sum(CF.conj()*Psi[key])    
            COEF_Minor[k] = coef

        COEF[key] = COEF_Minor

    return COEF

def Coef_Organizer(input_par, COEF, k_array):
    COEF_Organized = {}
    
    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    for i, k in enumerate(k_array):
        k = round(k, 3)
        COEF_Minor = {}
        for key in idx_map_box:
            m, l  = key[0], key[1]
            if l == 35:
                continue
            COEF_Minor[str((l,m))] = COEF[key][k]

        COEF_Organized[k] = COEF_Minor

    return COEF_Organized

def K_Sphere(coef_dic, input_par, phi, theta):
    idx_map_l_m, idx_map_box = Mod.Index_Map(input_par)
    theta, phi = np.meshgrid(theta, phi)
    out_going_wave = np.zeros(phi.shape, dtype=complex)
    for key in idx_map_box:
        m, l  = key[0], key[1]
        if l == 35:
            continue
        coef = coef_dic[str((l,m))]#[0] + 1j*coef_dic[str((l,m))][1]
        out_going_wave += coef*sph_harm(m, l, phi, theta)

    return out_going_wave

def closest(lst, k): 
    return lst[min(range(len(lst)), key = lambda i: abs(float(lst[i])-k))] 

def PAD_Momentum(COEF, input_par):

    resolution = 0.05
    x_momentum = np.arange(-2.5 ,2.5 + resolution, resolution)
    z_momentum = np.arange(-2.5 ,2.5 + resolution, resolution)
    resolution = 0.05
    y_momentum = np.arange(-2. ,2. + resolution, resolution)
    
    

    pad_value = np.zeros((z_momentum.size,x_momentum.size))

    for i, px in enumerate(x_momentum):
        print(round(px,3))
        for j, pz in enumerate(z_momentum):
            pad_value_temp = 0.0

            for l, py in enumerate(y_momentum):

                k = np.sqrt(px*px + py*py + pz*pz)
                if k == 0:
                    continue

                if px > 0 and py > 0:
                    phi = np.arctan(py/px)
                elif px > 0 and py < 0:
                    phi = np.arctan(py/px) + 2*pi
                elif px < 0 and py > 0:
                    phi = np.arctan(py/px) + pi
                elif px < 0 and py < 0:
                    phi = np.arctan(py/px) + pi
                elif px == 0 and py == 0:
                    phi = 0
                elif px == 0 and py > 0:
                    phi = pi / 2
                elif px == 0 and py < 0:
                    phi = 3*pi / 2
                elif py == 0 and px > 0:
                    phi = 0
                elif py == 0 and px < 0:
                    phi = pi

                theta = np.arccos(pz/k)
                coef_dic = COEF[closest(list(COEF.keys()), k)]
                pad_value_temp +=  np.abs(K_Sphere(coef_dic, input_par, phi, theta))**2

            pad_value[j, i] = pad_value_temp[0][0]

    return pad_value, x_momentum, z_momentum

if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    k_array = np.arange(0.05, 3, 0.05)



    energy, bound_states = Mod.Target_File_Reader_WO_Parity(input_par)
    Psi = Mod.Psi_Reader(input_par, "Psi_Final")
    Psi =  Proj_Bound_States_Out(input_par, Psi, bound_states)

    CS = Cont_State_Read()
    COEF = Coefficent_Calculator(input_par, k_array, CS, Psi)

    COEF_Organized = Coef_Organizer(input_par, COEF, k_array)

    pad_value, x_momentum, y_momentum = PAD_Momentum(COEF_Organized, input_par)
    pad_value = pad_value / pad_value.max()
    plt.imshow(pad_value, cmap='jet')#, interpolation="spline16")#, interpolation='nearest')
    plt.savefig("PAD_Final.png")