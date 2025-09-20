import matplotlib.pyplot as plt
from collections import defaultdict
import time

def nested_dict():
    return defaultdict(nested_dict)

def dictify(d):
    if isinstance(d, dict):
        return {k: dictify(v) for k, v in d.items()}
    else:
        return d


def drop_rep1(data):
    for movement, sets in data.items():
        for set_key, reps in sets.items():
            if "Rep:0:" in reps:
                del reps["Rep:0:"]
    return data


def getROM(dict):
    #Absolute Difference Method
    rom_abs = {}
    for movement, sets in dict.items():  
        rom_abs[movement] = {}
        
        for set_key, states in sets.items():
            rom_abs[movement][set_key] = {}
        
            down_reps = states.get('State:down:', {})
            up_reps = states.get('State:up:', {})
       
            for rep_key in down_reps.keys():
                if rep_key in up_reps:
                    down_val = down_reps[rep_key]
                    up_val = up_reps[rep_key]
                    rom_abs[movement][set_key][rep_key] = [abs(u - d) for u, d in zip(up_val, down_val)]
    return rom_abs

def plotROM(rom_dict,plot = "plot"):
    for movement, sets in rom_dict.items():
        plt.figure() 

        for set_key, reps in sets.items(): 
            reps_nums = []
            rom_values = []

            for rep_key, rom in reps.items():
                rep_num = int(rep_key.replace('Rep:', '').replace(':', ''))
                reps_nums.append(rep_num)
                rom_values.append(rom[0])

            if plot == "plot":
                plt.plot(reps_nums, rom_values, label=set_key)
            elif plot == "scatter":
                plt.scatter(reps_nums, rom_values, label=set_key)

        plt.xlabel('Reps')
        plt.ylabel('ROM (absolute difference)')
        plt.title(f'Range of Motion - {movement}')
        plt.legend()
        plt.savefig(f"Rom_{movement}.png")  

def plotTimes(times_dict,plot = "plot"):
    for movement, sets in times_dict.items():
        plt.figure() 

        for set_key, reps in sets.items(): 
            reps_nums = []
            times_values = []

            for rep_key, times in reps.items():
                rep_num = int(rep_key.replace('Tempo_Rep:', '').replace(':', ''))
                reps_nums.append(rep_num)
                times_values.append(times)

            if plot == "plot":
                plt.plot(reps_nums, times_values, label=set_key)
            elif plot == "scatter":
                plt.scatter(reps_nums, times_values, label=set_key)

        plt.xlabel('Reps')
        plt.ylabel('Elapsed time per rep')
        plt.title(f'Elapsed time per rep - {movement}')
        plt.legend()
        plt.savefig(f"Times_{movement}.png")  



def counter(start_t,rest_period):
     elapsed = time.time() - start_t
     remaining = int(rest_period - elapsed)
     return remaining
    

