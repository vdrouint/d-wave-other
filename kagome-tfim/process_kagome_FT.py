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


#load the graph and the dictionary

#final_qubit_dictionary = np.load("./data/dict_qbit_to_lattice.npy", allow_pickle=True).item()
final_qubit_dictionary = np.load("./data/dict_qbit_to_lattice_periodic.npy", allow_pickle=True).flat[0]
Kag_graph = pickle.load(open('./data/Kag_graph_periodic.pickle', 'rb'))

folder_global = "../data_prathus_runs/APQ1/raw"

dict_h = {}
Jmax = 0.66
mainfolder = list(Path(folder_global).glob('*'))
#mainfolder = list(Path('./data/raw_apq_zx').glob('*'))
for foldername in mainfolder:
    h1 = str(foldername).split('/')[-1].split('=')[-1]
    if h1 != ".DS_Store" and h1 != ".DS_Store:Zone.Identifier":
        #print("h1=",h1)
        hJtag = "{:.3f}".format(float(h1)/Jmax)
        dict_h.update({hJtag:h1})
print("various h/J in folder: " + folder_global)
print(dict_h.keys())