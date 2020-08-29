if True:
    import numpy as np
    import sys
    import h5py
    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt
    from math import pi, sqrt, log, ceil
    import scipy.integrate as Int
    import H2_Module as Mod 

def Gaussian(time, tau, center = 0):
    argument = -1*np.log(2)*np.power(2 *(time - center)/tau, 2.0)
    return np.exp(argument)

def Sin(time, tau, center = 0):
    return np.power(np.sin(pi*time / tau), 2.0)

def Pulse(intensity, envelop_fun, omega, num_of_cycles, CEP, time_spacing, polarization, poynting, ellipticity, frequency_shift = 1, cycle_delay = 0):
    
    if envelop_fun == Sin:
        mu = 4*pow(np.arcsin(np.exp(-1/4)),2.0)
    if envelop_fun == Gaussian:
        mu = 8*np.log(2.0)/pow(pi,2)

    if frequency_shift == 1:
        omega = omega*2.0/(1.0 + np.sqrt(1+ mu/pow(num_of_cycles,2))) 
    if frequency_shift != 1 and frequency_shift !=0:
        print(frequency_shift)
        print("freq shift should be 1 for true and 0 for false")
        exit()

    tau = 2*pi*num_of_cycles/omega
    tau_delay = 2*pi*cycle_delay/omega
    
    intensity = intensity / 3.51e16
    amplitude = pow(intensity, 0.5) / omega
    polarization = polarization / np.linalg.norm(polarization)
    poynting = poynting / np.linalg.norm(poynting)
    if np.dot(polarization, poynting) != 0.0:
        print("polarization of laser and the poynting direction are not orthagonal")
        exit()
    ellipticity_Vector = np.cross(poynting, polarization)    
    ellipticity_Vector = ellipticity_Vector / np.linalg.norm(ellipticity_Vector)

    Electric_Field = {}
    Vector_Potential = {}

    for i in range(len(polarization)):
    
        if envelop_fun == Gaussian:
            gaussian_length = 5
            time = np.arange(0, gaussian_length* tau, time_spacing)
            pulse_duration = gaussian_length * tau
            envelop = amplitude * envelop_fun(time, tau, (gaussian_length * tau)/2)
            
        if envelop_fun == Sin:  
            time = np.arange(0, tau + time_spacing, time_spacing)
            pulse_duration = tau
            envelop = amplitude * envelop_fun(time, tau)

        Vector_Potential[i] = envelop * 1/np.sqrt(1+pow(ellipticity,2.0)) * (polarization[i] * np.sin(omega*(time - tau/2) + CEP) + ellipticity * ellipticity_Vector[i] * np.cos(omega*(time - tau/2) + CEP))
        Electric_Field[i] =  -1.0 * np.gradient(Vector_Potential[i], time_spacing)

        if tau_delay != 0:
            time_delay = np.arange(0, tau_delay + time_spacing, time_spacing)
            Vector_Potential[i] = np.pad(Vector_Potential[i], (len(time_delay),0), 'constant', constant_values=(0,0))
            Electric_Field[i] = np.pad(Electric_Field[i], (len(time_delay),0), 'constant', constant_values=(0,0))
            
            
            if envelop_fun == Sin:  
                time = np.linspace(0, tau + tau_delay, len(Vector_Potential[i]))
            
    return time, Electric_Field, Vector_Potential, pulse_duration

def Build_Laser_Pulse(input_par):

    number_of_lasers = len(input_par["laser"]["pulses"])
    time_spacing = input_par["time_spacing"]
    laser = []
    total_polarization = np.zeros(3)
    total_poynting = np.zeros(3)
    laser_time = np.zeros(1)
    elliptical_pulse = False
    frequency_shift = input_par["laser"]["frequency_shift"]
    for i in range(number_of_lasers):
        intensity = input_par["laser"]["pulses"][i]["intensity"]
        envelop_fun = input_par["laser"]["pulses"][i]["envelop"]
        omega = input_par["laser"]["pulses"][i]["omega"]
        CEP = input_par["laser"]["pulses"][i]["CEP"]
        num_of_cycles = input_par["laser"]["pulses"][i]["num_of_cycles"]
        polarization = np.array(input_par["laser"]["pulses"][i]["polarization"])
        poynting = np.array(input_par["laser"]["pulses"][i]["poynting"])
        ellipticity = input_par["laser"]["pulses"][i]["ellipticity"]
        cycle_delay = input_par["laser"]["pulses"][i]["cycle_delay"]
        if ellipticity != 0:
            elliptical_pulse = True
        time, Electric_Field, Vector_Potential, pulse_duration = Pulse(intensity, eval(envelop_fun), omega, num_of_cycles, CEP, time_spacing, polarization, poynting, ellipticity, frequency_shift, cycle_delay)
    
        
        if len(time) > len(laser_time):
            laser_time = time

        total_polarization += polarization
        total_poynting += poynting
        laser.append({})
        
        laser[i]["pulse_duration"] = pulse_duration
        laser[i]["laser_pulse"] = {}


        if(input_par["gauge"] == "Length"):
            laser[i]["laser_pulse"]['x'] = Electric_Field[0]
            laser[i]["laser_pulse"]['y'] = Electric_Field[1]
            laser[i]["laser_pulse"]['z'] = Electric_Field[2]
        elif(input_par["gauge"] == "Velocity"):
            laser[i]["laser_pulse"]['x'] = Vector_Potential[0]
            laser[i]["laser_pulse"]['y'] = Vector_Potential[1]
            laser[i]["laser_pulse"]['z'] = Vector_Potential[2]
        else:
            print("Gauge not specified")  

    laser_pulse = {}
    laser_pulse['x'] = np.zeros(len(laser_time))
    laser_pulse['y'] = np.zeros(len(laser_time))
    laser_pulse['z'] = np.zeros(len(laser_time))

    for i in range(number_of_lasers):
        laser_pulse['x'][:len(laser[i]["laser_pulse"]['x'])] +=  laser[i]["laser_pulse"]['x']
        laser_pulse['y'][:len(laser[i]["laser_pulse"]['y'])] +=  laser[i]["laser_pulse"]['y']
        laser_pulse['z'][:len(laser[i]["laser_pulse"]['z'])] +=  laser[i]["laser_pulse"]['z']
        

    laser_pulse['Right'] = 0.5*(laser_pulse['x'] - 1.0j*laser_pulse['y'])
    laser_pulse['Left'] = 0.5*(laser_pulse['x'] + 1.0j*laser_pulse['y'])

    total_polarization = total_polarization / np.linalg.norm(total_polarization)
    total_poynting = total_poynting / np.linalg.norm(total_poynting)
    
    Pulse_Plotter(laser_time, laser_pulse, "Pulse.png")
    return laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse

def Window_Function(time, sigma):
    arg = -1.0*np.power(time, 2.0) / (2*pow(sigma,2.0))
    return np.exp(arg) / pow(2*pi*pow(sigma, 2.0), 0.5)

def Husimi_Trans(time, pulse, tau, omega):
    return_val = []
    for w in omega:
        omega_array = []
        for t in tau:
            w_fun  = Window_Function(time - t, 1)
            integrand = w_fun * pulse * np.exp(-1.0J * w * (time - t))
            integrand = Int.simps(integrand, time, abs(time[1] - time[0]))
            integrand = np.vdot(integrand, integrand)
            
            omega_array.append(integrand.real)
        return_val.append(omega_array) 
    return return_val

def Joel_TDSE_Output_Reader(directory_path=None):
    if directory_path == None:
        TDSE_file = h5py.File("TDSE.h5")
        Pulse_file = h5py.File("Pulse.h5")
    else:
        TDSE_file = h5py.File(file + "/TDSE.h5")
        Pulse_file = h5py.File(file + "/Pulse.h5")

    pulse_array = Pulse_file["Pulse"]
    time = pulse_array["time"][:]
    num_dims = TDSE_file["Parameters"]["num_dims"][0]
    num_electrons = TDSE_file["Parameters"]["num_electrons"][0]
    num_pulses = TDSE_file["Parameters"]["num_pulses"][0]
    checkpoint_frequency = TDSE_file["Parameters"]["write_frequency_observables"][0]
    energy = TDSE_file["Parameters"]["energy"][0]
    count = 0
    for dim_idx in range(num_dims):
        if count == 0:
            pulse = pulse_array["field_" + str(dim_idx)][:]
        else:
            pulse += pulse_array["field_" + str(dim_idx)][:]
    return time, pulse

def Pulse_Plotter(laser_time, laser_pulse, plot_name):
    
    plt.plot(laser_time, laser_pulse['x'], label = 'x')
    plt.plot(laser_time, laser_pulse['y'], label = 'y')
    plt.plot(laser_time, laser_pulse['z'], label = 'z')
    
    # plt.legend()
    plt.savefig(plot_name)
    # plt.clf()

if __name__=="__main__":
    polarization = np.array([1,0,0])
    poynting = np.array([0,0,1])
    # input_par = Mod.Input_File_Reader("input.json")
    # time_1, Electric_Field_1, Vector_Potential_1, pulse_duration = Pulse(5.0e13, Sin, 0.375, 10, 0, 0.1, polarization, poynting, -1, 0, 2)

    # laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse = Build_Laser_Pulse(input_par)
    

