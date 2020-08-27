if True:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    import sys
    import json
    import mpl_toolkits.mplot3d.axes3d as axes3d
    from scipy.special import sph_harm
    import H2_Module as Mod

    import warnings
    warnings.filterwarnings("error")


def Field_Free_Wavefunction_Reader(Target_File, input_par, n):
    FF_WF = {}
    psi = Target_File["Psi_" + str(n)]
    psi = psi[:,0] + 1.0j*psi[:,1]
    grid = Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"])

    r_ind_max = len(grid)
    r_ind_lower = 0
    r_ind_upper = r_ind_max

    for l in range(input_par["l_max"] + 1):
        FF_WF[(l)] = np.array(psi[r_ind_lower: r_ind_upper])

        # FF_WF[(l)] = FF_WF[(l)]/ np.linalg.norm(FF_WF[(l)])

        r_ind_lower = r_ind_upper
        r_ind_upper = r_ind_upper + r_ind_max

    return(FF_WF)

def Wavefunction_Plotter(input_par, FF_WF):
    grid = list(Mod.Make_Grid(input_par["grid_spacing"], input_par["grid_size"], input_par["grid_spacing"]))
   
    resolution = 0.1
    X = np.arange(-10. ,10. + resolution, resolution) 
    Z = np.arange(-10. ,10. + resolution, resolution)
    Y = np.arange(-5. ,5. + resolution, resolution)


    X_Array, Z_Array = np.meshgrid(X, Z)

    Psi = np.zeros(X_Array.shape, dtype=complex)
    
    m = 0

    for i, x in enumerate(X):
        print(x)
        for j, z in enumerate(Z):
            for k, y in enumerate(Y):
                
                r = np.sqrt(x*x + y*y + z*z)
                r = closest(grid, r)
                r_idx = grid.index(r)
            
                if z/r > 1:
                    theta = np.arccos(1)
                elif z/r < -1:
                    theta = np.arccos(-1)
                else:
                    theta = np.arccos(z/r)
                
                if x > 0 and y > 0:
                    phi = np.arctan(y/x)
                elif x > 0 and y < 0:
                    phi = np.arctan(y/x) + 2*np.pi
                elif x < 0 and y > 0:
                    phi = np.arctan(y/x) + np.pi
                elif x < 0 and y < 0:
                    phi = np.arctan(y/x) + np.pi
                elif x == 0 and y == 0:
                    phi = 0
                elif x == 0 and y > 0:
                    phi = np.pi / 2
                elif x == 0 and y < 0:
                    phi = 3*np.pi / 2
                elif y == 0 and x > 0:
                    phi = 0
                elif y == 0 and x < 0:
                    phi = np.pi

                for l in range(input_par["l_max"] + 1):
                    Psi[j, i] += FF_WF[l][r_idx]*sph_harm(m, l, phi, theta)
                      
            
    return X_Array, Z_Array, Psi

def closest(lst, k): 
    return lst[min(range(len(lst)), key = lambda i: abs(float(lst[i])-k))] 
    
if __name__=="__main__":

    file = sys.argv[1]
    input_par = Mod.Input_File_Reader(file + "input.json")
    Target_File = h5py.File(file + "/" + input_par["Target_File"])

    n = 2
    FF_WF = Field_Free_Wavefunction_Reader(Target_File, input_par, n)
    X_Array, Z_Array, Psi = Wavefunction_Plotter(input_par, FF_WF)

    fig, ax = plt.subplots()

    Psi = np.absolute(Psi)

    psi_min = Psi.min()
    psi_max = Psi.max()

    
    c = ax.pcolormesh(X_Array, Z_Array, Psi, cmap='RdBu', vmin=psi_min, vmax=psi_max)
    ax.axis([X_Array.min(), X_Array.max(), Z_Array.min(), Z_Array.max()])
    fig.colorbar(c, ax=ax)
    plt.savefig("PSI_0_2.png")
