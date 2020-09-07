from Potential import wigner3j
import H2_Module as Mod 
import time

input_par = Mod.Input_File_Reader(input_file = "input.json")
index_map_m_1, index_map_box = Mod.Index_Map_M_Block(input_par)

start_time = time.time()
for i in index_map_box.keys():
    m = i[0]
    l = i[0]


    if l % 2 == 0:
        l_prime_list = list(range(0, input_par["l_max"] + 1, 2))
    else:
        l_prime_list = list(range(1, input_par["l_max"] + 1, 2))

    for l_prime in l_prime_list:
        for lamda in range(abs(l-l_prime), l + l_prime + 2, 2):
            if abs(m) > l or abs(m) > l_prime:
                continue
            coef = wigner3j(l,lamda,l_prime,0,0,0) * wigner3j(l,lamda,l_prime,-m,0,m)
            print(l, l_prime,lamda,m,coef)


print((time.time() - start_time)/60)
