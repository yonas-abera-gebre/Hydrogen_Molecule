import Module as Mod

if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    energy, wave_function = Mod.Target_File_Reader(input_par)

    for k in energy.keys():
        print(k, energy[k])
    # index_map_m_l, index_map_box = Mod.Index_Map(input_par)

    # print(index_map_box)