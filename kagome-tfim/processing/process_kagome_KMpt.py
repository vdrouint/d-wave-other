import numpy as np
import dimod
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx
import minorminer
import minorminer.layout as mml
import dwave.inspector
import matplotlib as mpl
import math
from datetime import datetime
from pathlib import Path  
import pandas as pd
import os
import pickle
import random
import h5py


####
#onto the code
####

####
#CAREFUL - put these files from kagome_embedding.ipynb in the correct folder
####
#load the graph and the dictionary
final_qubit_dictionary_old = np.load("dict_qbit_to_lattice_periodic.npy", allow_pickle=True).flat[0]
Kag_graph = pickle.load(open('Kag_graph_periodic.pickle', 'rb'))
unit_cells = np.load("unitcells.npy", allow_pickle=True).flat[0]
unit_cells_down = np.load("unitcells_down.npy", allow_pickle=True).flat[0]

sublatt_dict_truncer = (np.load("sublatt_dict_truncer.npy", allow_pickle=True))
sublatt_dict_truncer = sublatt_dict_truncer.tolist()
#filtering final_qubit_dictionary
final_qubit_dictionary_new = {}
for i in range(len(final_qubit_dictionary_old.keys())):
    key = list(final_qubit_dictionary_old.keys())[i]
    if key in list(sublatt_dict_truncer.keys()):
        final_qubit_dictionary_new.update({key:final_qubit_dictionary_old[key]})

final_qubit_dictionary = final_qubit_dictionary_new.copy()


####
#CAREFUL - find your own path and J values  
####
# folder_base = "../../data_prathus_runs/"
folder_base = '/Users/prathunarasimhan/Desktop/APQ 7_31 raw'
# all_folders = [folder_base + "data_prathus_runs/APQ new/"]
all_folders = [folder_base]
allJvals = [0.83]
#you can put more folders in there, but you likely wont need it

# folder_global = folder_base + "raw"

# where_to_save = "/Users/prathunarasimhan/Desktop/processed FT"
# where_to_save = '/Volumes/KAGOME/processedFT2/'
where_to_save = '//Users/prathunarasimhan/Desktop/FTBulk/'
folder_create = Path(where_to_save)
print(folder_create)
folder_create.mkdir(parents=True, exist_ok=True)


K = (4*np.pi/3)*np.array([1, 0])
M = (4*np.pi/3)*(np.array([1, 0]) + np.array([np.cos(2*np.pi/6), np.sin(2*np.pi/6)]))/2


for mf in range(len(all_folders)):

    print("start of folder" + all_folders[mf])
    folder_global = all_folders[mf]
    Jmax = allJvals[mf]

    dict_h = {}
    mainfolder = list(Path(folder_global).glob('*'))
    #mainfolder = list(Path('./data/raw_apq_zx').glob('*'))
    for foldername in mainfolder:
        h1overj = str(foldername).split('/')[-1].split('=')[-1]
        if h1overj != ".DS_Store" and h1overj != ".DS_Store:Zone.Identifier":
            # print("h1/j=",h1overj)
#             hJtag = "{:.3f}".format(float(h1)/Jmax)
            hJtag = h1overj
            h1 = float(h1overj) * Jmax
            dict_h.update({hJtag:h1})
    print("various h/J in folder: " + folder_global)
    # print(dict_h.keys())

    ####
    #CAREFUL - choose the right format for your folders to be accessed
    #what h/J do you want?
    #what s_p do you want?
    ####
    # hoverJ_to_process = [val for val in dict_h.keys()]
    hoverJ_to_process = [str(val) for val in [0,0.5,1,1.25,2,2.5,3,3.5,4]]
    # s_vals = [0.15,0.4,0.65]
    # s_vals = np.round(sorted(list(np.arange(0.15,0.651,0.05)) + [0.275,0.325,0.375,0.425, 0.475, 0.525]),2)
    s_vals = [0.15,0.25,0.35,0.4,0.45,0.55,0.65]
    s_vals_strs = [str(s_val) for s_val in s_vals]
    s_vals = s_vals_strs.copy()
    print(hoverJ_to_process)
    print(s_vals)

    for hview in hoverJ_to_process:
        hJview = hview
        for sview in s_vals:
            folder = folder_global + "/hoverj=" + hJview + "/s=" + sview
            print(folder)

            mainfolder = list(Path(folder).glob('*'))
            #mainfolder = list(Path('./data/raw_apq_zx').glob('*'))

            reps = 0
            folders_to_open = []
            if len(mainfolder) != 0:
                print("success in finding this folder")
            else:
                print('problem')
            for foldername in mainfolder:
                if (str(foldername)[-4:] == ".npz" and str(foldername).split('.')[-2].split('_')[-1] != 'shimdata'):
                    reps += 1
                    folders_to_open.append(foldername)
            print(len(folders_to_open))


            all_configs = []
            all_energies = []
            for k in range(len(folders_to_open)):
                file=np.load(folders_to_open[k])
                resps = file['resp']

                for i in range(len(resps)):
                    frequency = resps[i][2]
                    for freq in range(frequency):
                        all_configs.append(resps[i][0])
                        all_energies.append(resps[i][1])

            all_configs = np.array(all_configs)
            file2=np.load(folders_to_open[0])
            qubit_variables = np.array(file2["final_nodes"])

            #nodes period is the variables run on the QPU
            n_vars = len(qubit_variables)
            n_sites = len(bulk_sites)
            interaction_vector_K = np.zeros(n_vars) + 1j*np.zeros(n_vars)
            interaction_vector_M = np.zeros(n_vars) + 1j*np.zeros(n_vars)

            for i in range(n_vars):
                if qubit_variables[i] in final_qubit_dictionary:
                    rdist = final_qubit_dictionary[qubit_variables[i]]
                    interaction_vector_K[i] = np.exp(1j*np.dot(K, rdist))
                    interaction_vector_M[i] = np.exp(1j*np.dot(M, rdist))
                else:
                    interaction_vector_K[i] = 0.0
                    interaction_vector_M[i] = 0.0

            value_K = np.zeros(len(all_configs)) + 1j*np.zeros(len(all_configs))
            value_M = np.zeros(len(all_configs)) + 1j*np.zeros(len(all_configs))
            for rep in range(len(all_configs)):
                vec = resps[rep][0]
                value_K[rep] = np.dot(vec, interaction_vector_K) #this is a complex number
                value_M[rep] = np.dot(vec, interaction_vector_M) #this is a complex number

            ####
            #saving
            ####

            filename_hs = where_to_save + "KMpts_h=" + hview + "_s=" + sview + ".hdf5"

            try:
                os.remove(filename_hs)
            except OSError:
                pass
            with h5py.File(Path(filename_hs), "w") as f:
                f.create_dataset("real_sigma_K", data = np.real(value_K))
                f.create_dataset("real_sigma_M", data = np.real(value_M))
                f.create_dataset("real_sigma_K_3", data = np.real(value_K**3))
                f.create_dataset("real_sigma_M_3", data = np.real(value_M**3))
                f.create_dataset("real_sigma_K_6", data = np.real(value_K**6))
                f.create_dataset("real_sigma_M_6", data = np.real(value_M**6))

                f.create_dataset("imag_sigma_K", data = np.imag(value_K))
                f.create_dataset("imag_sigma_M", data = np.imag(value_M))
                f.create_dataset("imag_sigma_K_3", data = np.imag(value_K**3))
                f.create_dataset("imag_sigma_M_3", data = np.imag(value_M**3))
                f.create_dataset("imag_sigma_K_6", data = np.imag(value_K**6))
                f.create_dataset("imag_sigma_M_6", data = np.imag(value_M**6))

            print("done with h/J=" + hview + " and s=" + sview)
