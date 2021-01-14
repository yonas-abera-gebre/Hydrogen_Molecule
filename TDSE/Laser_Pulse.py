if True:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from math import pi, sqrt, log, ceil
    import Module as Mod 

def Gaussian(time, tau, center = 0):
    argument = -1*np.log(2)*np.power(2 *(time - center)/tau, 2.0)
    return np.exp(argument)

def Sin(time, tau, center = 0):
    return np.power(np.sin(pi*time / tau), 2.0)

def Freq_Shift(omega, envelop_fun, num_of_cycles, freq_shift = 1):
    if freq_shift == 1:
        if envelop_fun == Sin:
            mu = 4*pow(np.arcsin(np.exp(-1/4)),2.0)
        if envelop_fun == Gaussian:
            mu = 8*np.log(2.0)/pow(pi,2)
        omega = omega*2.0/(1.0 + np.sqrt(1+ mu/pow(num_of_cycles,2))) 
        return omega
    elif freq_shift == 0:
        return omega
    else:
        print(freq_shift)
        print("freq shift should be 1 for true and 0 for false")
        exit()

def Cycle_to_time(cycle, omega):
    return 2*pi*cycle/omega

def Laser_Vectors(polarization, poynting):
    polarization = polarization / np.linalg.norm(polarization)
    poynting = poynting / np.linalg.norm(poynting)
    
    if np.dot(polarization, poynting) != 0.0:
        print("polarization of laser and the poynting direction are not orthagonal \n")
        exit()

    ellipticity_Vector = np.cross(poynting, polarization)    
    ellipticity_Vector = ellipticity_Vector / np.linalg.norm(ellipticity_Vector)

    return polarization, poynting, ellipticity_Vector

def Pulse(intensity, envelop_fun, omega, num_of_cycles, CEP, time_spacing, polarization, poynting, ellipticity, freq_shift = 1, cycle_delay = 0):
    omega = Freq_Shift(omega, envelop_fun, num_of_cycles, freq_shift)
    tau = Cycle_to_time(num_of_cycles, omega)
    tau_delay = Cycle_to_time(cycle_delay, omega)

    amplitude = pow(intensity / 3.51e16, 0.5) / omega
    polarization, poynting, ellipticity_Vector = Laser_Vectors(polarization, poynting)

    Electric_Field = {}
    Vector_Potential = {}

    for i in range(3):
    
        if envelop_fun == Gaussian:
            gaussian_length = 5
            time = np.arange(0, gaussian_length* tau, time_spacing)
            envelop = amplitude * envelop_fun(time, tau, (gaussian_length * tau)/2)
            
        if envelop_fun == Sin:  
            time = np.arange(0, tau + time_spacing, time_spacing)
            envelop = amplitude * envelop_fun(time, tau)

        Vector_Potential[i] = envelop * 1/np.sqrt(1+pow(ellipticity,2.0)) * (polarization[i] * np.sin(omega*(time - tau/2) + CEP) + ellipticity * ellipticity_Vector[i] * np.cos(omega*(time - tau/2) + CEP))
        Electric_Field[i] =  -1.0 * np.gradient(Vector_Potential[i], time_spacing)

        
        if tau_delay != 0:
            time_delay = np.arange(0, tau_delay + time_spacing, time_spacing)
            Vector_Potential[i] = np.pad(Vector_Potential[i], (len(time_delay),0), 'constant', constant_values=(0,0))
            Electric_Field[i] = np.pad(Electric_Field[i], (len(time_delay),0), 'constant', constant_values=(0,0))
            
            if envelop_fun == Sin:  
                time = np.linspace(0, tau + tau_delay, len(Vector_Potential[i]))

    return time, Electric_Field, Vector_Potential

def Build_Laser_Pulse(input_par):

    number_of_lasers = len(input_par["laser"]["pulses"])
    time_spacing = input_par["time_spacing"]

    laser = []
    total_polarization = np.zeros(3)
    total_poynting = np.zeros(3)
    laser_time = np.zeros(1)

    elliptical_pulse = False
    freq_shift = input_par["laser"]["freq_shift"]
    free_prop_steps = input_par["laser"]["free_prop_steps"]

    for i in range(number_of_lasers):
        current_pulse = input_par["laser"]["pulses"][i]
        intensity = current_pulse["intensity"]
        envelop_fun = current_pulse["envelop"]
        omega = current_pulse["omega"]
        CEP = current_pulse["CEP"]
        num_of_cycles = current_pulse["num_of_cycles"]
        polarization = np.array(current_pulse["polarization"])
        poynting = np.array(current_pulse["poynting"])
        ellipticity = current_pulse["ellipticity"]
        cycle_delay = current_pulse["cycle_delay"]


        if ellipticity != 0:
            elliptical_pulse = True
        time, Electric_Field, Vector_Potential = Pulse(intensity, eval(envelop_fun), omega, num_of_cycles, CEP, time_spacing, polarization, poynting, ellipticity, freq_shift, cycle_delay)
    
        
        if len(time) > len(laser_time):
            laser_time = time

        total_polarization += polarization
        total_poynting += poynting
        laser.append({})
        
        laser[i] = {}


        if(input_par["gauge"] == "Length"):
            laser[i]['x'] = Electric_Field[0]
            laser[i]['y'] = Electric_Field[1]
            laser[i]['z'] = Electric_Field[2]
        elif(input_par["gauge"] == "Velocity"):
            laser[i]['x'] = Vector_Potential[0]
            laser[i]['y'] = Vector_Potential[1]
            laser[i]['z'] = Vector_Potential[2]
        else:
            print("Gauge not specified")  

    laser_pulse = {}
    laser_pulse['x'] = np.zeros(len(laser_time))
    laser_pulse['y'] = np.zeros(len(laser_time))
    laser_pulse['z'] = np.zeros(len(laser_time))

    for i in range(number_of_lasers):
        laser_pulse['x'][:len(laser[i]['x'])] +=  laser[i]['x']
        laser_pulse['y'][:len(laser[i]['y'])] +=  laser[i]['y']
        laser_pulse['z'][:len(laser[i]['z'])] +=  laser[i]['z']
        

    laser_pulse['Right'] = 0.5*(laser_pulse['x'] - 1.0j*laser_pulse['y'])
    laser_pulse['Left'] = 0.5*(laser_pulse['x'] + 1.0j*laser_pulse['y'])

    total_polarization /= np.linalg.norm(total_polarization)
    total_poynting /= np.linalg.norm(total_poynting)
    
    free_prop_idx = len(laser_time) - 1
 
    if free_prop_steps != 0:
        laser_pulse['x'] = np.pad(laser_pulse['x'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['y'] = np.pad(laser_pulse['y'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['z'] = np.pad(laser_pulse['z'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['Right'] = np.pad(laser_pulse['Right'], (0, free_prop_steps), 'constant', constant_values=(0,0))
        laser_pulse['Left'] = np.pad(laser_pulse['Left'], (0, free_prop_steps), 'constant', constant_values=(0,0))

    laser_end_time = laser_time[-1] + free_prop_steps*time_spacing
    laser_time = np.linspace(0, laser_end_time, len(laser_pulse['x']))
    
    Pulse_Plotter(laser_time, laser_pulse, "Pulse.png")

    return laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse, free_prop_idx

def Pulse_Plotter(laser_time, laser_pulse, plot_name):
    
    plt.plot(laser_time, laser_pulse['x'], label = 'x')
    plt.plot(laser_time, laser_pulse['y'], label = 'y')
    plt.plot(laser_time, laser_pulse['z'], label = 'z')
    
    plt.legend()
    plt.savefig(plot_name)
    plt.clf()

if __name__=="__main__":

    input_par = Mod.Input_File_Reader("input.json")
    laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse, free_prop_idx = Build_Laser_Pulse(input_par)
    
    print(len(laser_time))
    print(free_prop_idx)

    if free_prop_idx != len(laser_time) - 1:
        save_idx = list(range(free_prop_idx, len(laser_time), int((len(laser_time) - free_prop_idx)/10)))
    else:
        save_idx = [free_prop_idx]
        
    print(save_idx)
    print(save_idx[-1])
    
    # polarization = np.array([1,0,0])
    # poynting = np.array([0,0,1])
    # time, Electric_Field, Vector_Potential = Pulse(5.0e13, Sin, 0.375, 2, 0, 0.1, polarization, poynting, -1, 0, 2)
    # plt.plot(time, Vector_Potential[0])
    # plt.savefig("pulse.png")
    # plt.clf()

    
    

