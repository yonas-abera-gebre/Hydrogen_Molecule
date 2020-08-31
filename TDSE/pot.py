import Potential as Pot
import H2_Module as Mod 

if __name__=="__main__":
    input_par = Mod.Input_File_Reader("input.json")
    Pot.Potential(input_par)
