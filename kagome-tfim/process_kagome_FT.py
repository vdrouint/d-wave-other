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


#####
#function definitions
#####

def momentum_grid_kagome(N):
        # Reciprocal lattice vectors
        b1 = 2*np.pi * np.array([1, -1/np.sqrt(3)])
        b2 = 2*np.pi * np.array([0, 2/np.sqrt(3)])

        # Generate a mesh of points in the range -0.5 to 0.5 for each reciprocal lattice vector
        points1 = np.linspace(-1, 1, N)
        points2 = np.linspace(-1, 1, N)
        mesh1, mesh2 = np.meshgrid(points1, points2)

        m1 = mesh1.flatten()
        m2 = mesh2.flatten()
        kx_vals = np.zeros(N**2)
        ky_vals = np.zeros(N**2)
        for i in range(N**2):
                kx_vals[i] = m1[i]*b1[0] + m2[i]*b2[0]
                ky_vals[i] = m1[i]*b1[1] + m2[i]*b2[1]

        total_K = []
        for i in range(N**2):
                total_K.append([kx_vals[i], ky_vals[i]])
        total_K = np.array(total_K)
        kx_vals = kx_vals.reshape(N,N)
        ky_vals = ky_vals.reshape(N,N)



        return total_K, kx_vals, ky_vals

def momentum_grid(N):
        # Reciprocal lattice vectors
        b1 = 2*np.pi * np.array([1,0])
        b2 = 2*np.pi * np.array([0, 1])

        # Generate a mesh of points in the range -0.5 to 0.5 for each reciprocal lattice vector
        points1 = np.linspace(-0.5, 0.5, N)
        points2 = np.linspace(-0.5, 0.5, N)
        mesh1, mesh2 = np.meshgrid(points1, points2)

        m1 = mesh1.flatten()
        m2 = mesh2.flatten()
        kx_vals = np.zeros(N**2)
        ky_vals = np.zeros(N**2)
        for i in range(N**2):
                kx_vals[i] = m1[i]*b1[0] + m2[i]*b2[0]
                ky_vals[i] = m1[i]*b1[1] + m2[i]*b2[1]

        total_K = []
        for i in range(N**2):
                total_K.append([kx_vals[i], ky_vals[i]])
        total_K = np.array(total_K)
        kx_vals = kx_vals.reshape(N,N)
        ky_vals = ky_vals.reshape(N,N)



        return total_K, kx_vals, ky_vals

def path_Kagome(N):
    # Define the reciprocal lattice vectors
    b1 = (2*np.pi/3)*np.array([1, -1/np.sqrt(3)])
    b2 = (2*np.pi/3)*np.array([1, 1/np.sqrt(3)])

    # Define the high-symmetry points
    Gamma = np.array([0, 0])
    #K = (4*np.pi/9)*np.array([1, 1/np.sqrt(3)])
    #G = (2*np.pi/3)*np.array([1, 0])
    K = (4*np.pi/3)*np.array([1, 0])
    G = (4*np.pi/3)*(np.array([1, 0]) + np.array([np.cos(2*np.pi/6), np.sin(2*np.pi/6)]))/2

    # Define the number of points between Gamma-K and K-Gamma'
    # Generate the path
    path = [Gamma]
    path += [Gamma + (i/N)*(K-Gamma) for i in range(1, N+1)]
    path += [K]
    path += [K + (i/N)*(G-K) for i in range(1, N+1)]
    path += [G]
    path += [G + (i/N)*(Gamma-G) for i in range(1, N+1)]

    # Convert the path to Cartesian coordinates
    path_cart = path
    kx_vals = []
    ky_vals = []
    for point in path_cart:
        kx_vals.append(point[0])
        ky_vals.append(point[1])

    return path_cart, kx_vals, ky_vals

def momentum_grid_kagome_scale(N, ascale):
        # Reciprocal lattice vectors
        b1 = 2*np.pi/ascale * np.array([1, -1/np.sqrt(3)])
        b2 = 2*np.pi/ascale * np.array([0, 2/np.sqrt(3)])

        # Generate a mesh of points in the range -0.5 to 0.5 for each reciprocal lattice vector
        points1 = np.linspace(-1, 1, N)
        points2 = np.linspace(-1, 1, N)
        mesh1, mesh2 = np.meshgrid(points1, points2)

        m1 = mesh1.flatten()
        m2 = mesh2.flatten()
        kx_vals = np.zeros(N**2)
        ky_vals = np.zeros(N**2)
        for i in range(N**2):
                kx_vals[i] = m1[i]*b1[0] + m2[i]*b2[0]
                ky_vals[i] = m1[i]*b1[1] + m2[i]*b2[1]

        total_K = []
        for i in range(N**2):
                total_K.append([kx_vals[i], ky_vals[i]])
        total_K = np.array(total_K)
        kx_vals = kx_vals.reshape(N,N)
        ky_vals = ky_vals.reshape(N,N)



        return total_K, kx_vals, ky_vals

def Sq_for_record(all_record_configs, qubit_dictionary, qubit_variables, L_Kpoints):
    #L_Kpoints needs to be odd to see the q=0 component

    total_K, kx_vals, ky_vals = momentum_grid_kagome(L_Kpoints)

    structure_factor = np.zeros(len(total_K)) + 1j*np.zeros(len(total_K))
    # structure_factor_mK = np.zeros(len(total_K)) + 1j*np.zeros(len(total_K))
    S_spin = np.zeros(len(total_K))
    S2_spin = np.zeros(len(total_K)) + 1j*np.zeros(len(total_K))
    S3_spin = np.zeros(len(total_K))

    num_reads = len(all_record_configs)

    for k in range(num_reads):
        response_analyzed = {}
        for j in range(len(qubit_variables)):
            response_analyzed.update({qubit_variables[j]:all_record_configs[k][j]})
        avg = 0.0
        #calculate_avg = False
        #if calculate_avg == True:
        #    for q1 in qubit_dictionary.keys():
        #        avg += response_analyzed[q1]
        #    avg = avg / len(qubit_dictionary)

        for kp in range(len(total_K)):
            val_K = 0
            val_mK = 0
            for q1 in qubit_dictionary.keys():
                kpoint = total_K[kp]
                r1 = qubit_dictionary[q1]
                val_K += np.exp(1j*np.dot(kpoint, r1))*(response_analyzed[q1])
                val_mK += np.exp(-1j*np.dot(kpoint, r1))*(response_analyzed[q1])
            val_K = val_K/len(qubit_dictionary)
            val_mK = val_mK/len(qubit_dictionary)
            structure_factor[kp] += val_K
            # structure_factor_mK[kp] += val_mK
            S_spin[kp] += abs(val_K)
            S2_spin[kp] += val_K*val_mK
            S3_spin[kp] += abs(val_K*val_mK)
        if k % (num_reads // 10) == 0:
            print("Done with reads: ", k)


    structure_factor = structure_factor / num_reads
    # structure_factor_mK = structure_factor / num_reads
    S_spin = S_spin / num_reads
    S2_spin = S2_spin / num_reads
    S3_spin = S3_spin / num_reads

    return total_K, kx_vals, ky_vals, S_spin, S2_spin, S3_spin, structure_factor
    

def line_BZ(all_record_configs, qubit_dictionary, qubit_variables, L_Kpoints):
    #L_Kpoints needs to be odd to see the q=0 component

    total_K, kx_vals, ky_vals = path_Kagome(L_Kpoints)

    structure_factor = np.zeros(len(total_K)) + 1j*np.zeros(len(total_K))
    # structure_factor_mK = np.zeros(len(total_K)) + 1j*np.zeros(len(total_K))
    S_spin = np.zeros(len(total_K))
    S2_spin = np.zeros(len(total_K)) + 1j*np.zeros(len(total_K))
    S3_spin = np.zeros(len(total_K))

    num_reads = len(all_record_configs)

    for k in range(num_reads):
        response_analyzed = {}
        for j in range(len(qubit_variables)):
            response_analyzed.update({qubit_variables[j]:all_record_configs[k][j]})
        avg = 0.0
        #calculate_avg = False
        #if calculate_avg == True:
        #    for q1 in qubit_dictionary.keys():
        #        avg += response_analyzed[q1]
        #    avg = avg / len(qubit_dictionary)

        for kp in range(len(total_K)):
            val_K = 0
            val_mK = 0
            for q1 in qubit_dictionary.keys():
                kpoint = total_K[kp]
                r1 = qubit_dictionary[q1]
                val_K += np.exp(1j*np.dot(kpoint, r1))*(response_analyzed[q1])
                val_mK += np.exp(-1j*np.dot(kpoint, r1))*(response_analyzed[q1])
            val_K = val_K/len(qubit_dictionary)
            val_mK = val_mK/len(qubit_dictionary)
            structure_factor[kp] += val_K
            # structure_factor_mK[kp] += val_mK
            S_spin[kp] += abs(val_K)
            S2_spin[kp] += val_K*val_mK
            S3_spin[kp] += abs(val_K*val_mK)
        if k % (num_reads // 10) == 0:
            print("Done with reads: ", k)

    structure_factor = structure_factor / num_reads
    # structure_factor_mK = structure_factor / num_reads
    S_spin = S_spin / num_reads
    S2_spin = S2_spin / num_reads
    S3_spin = S3_spin / num_reads

    return total_K, kx_vals, ky_vals, S_spin, S2_spin, S3_spin, structure_factor


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
####
#onto the code
####

#load the graph and the dictionary

#final_qubit_dictionary = np.load("./data/dict_qbit_to_lattice.npy", allow_pickle=True).item()
final_qubit_dictionary = np.load("./data/dict_qbit_to_lattice_periodic.npy", allow_pickle=True).flat[0]
Kag_graph = pickle.load(open('./data/Kag_graph_periodic.pickle', 'rb'))


#folder_base = "../data_prathus_runs/APQ1/"
#Jmax = 0.66
# folder_base = "../data_prathus_runs/periodic Kagome fwd/"
all_folders = ["../data_prathus_runs/APQ1/", "../data_prathus_runs/periodic Kagome fwd2/", "../data_prathus_runs/periodic Kagome fwd3/", "../data_prathus_runs/periodic Kagome fwd4/", "../data_prathus_runs/periodic Kagome fwd5/"]
# Jmax = 0.66
allJvals = [0.66, 0.66, 0.83, 1.0, 1.4]

# folder_global = folder_base + "raw"

where_to_save = "../data_prathus_runs/processed_FT/"
folder_create = Path(where_to_save[:-1])
folder_create.mkdir(parents=True, exist_ok=True)

for mf in range(len(all_folders)):

    print("start of folder" + all_folders[mf])
    folder_global = all_folders[mf] + "raw"
    Jmax = allJvals[mf]

    if folder_global == "../data_prathus_runs/APQ1/raw":

        dict_h = {}
        mainfolder = list(Path(folder_global).glob('*'))
        #mainfolder = list(Path('./data/raw_apq_zx').glob('*'))
        for foldername in mainfolder:
            h1 = str(foldername).split('/')[-1].split('=')[-1]
            if h1 != ".DS_Store" and h1 != ".DS_Store:Zone.Identifier":
                #print("h1=",h1)
                hJtag = "{:.3f}".format(float(h1)/Jmax)
                dict_h.update({hJtag:h1})
        # print("various h/J in folder: " + folder_global)
        # print(dict_h.keys())


        hoverJ_to_process = ['0.000', '2.004', '3.908', '5.211', '0.501', '2.505', '1.503', '3.006', '3.507', '1.002']
        #hoverJ_to_process = ['0.000']
        s_vals = ['s=0.7', 's=0.2']
        Nreads = 200

        for hview in hoverJ_to_process:
            for sview in s_vals:
                folder = folder_global + "/h1=" + dict_h[hview] + "/" + sview

                mainfolder = list(Path(folder).glob('*'))
                #mainfolder = list(Path('./data/raw_apq_zx').glob('*'))

                reps = 0
                folders_to_open = []
                if len(mainfolder) != 0:
                    print("success in finding this folder")
                else:
                    print('problem')
                for foldername in mainfolder:
                    if str(foldername)[-4:] == ".npz":
                        #print(foldername)
                        reps += 1
                        folders_to_open.append(foldername)

                # def open_file(path):
                    
                #     file=np.load(path)
                    
                #     resp=file['resp']
                #     paramsarray=file['paramsarray']
                #     missingqs=file['missingqs']
                #     twochains=file['twochains']
                #     nodes=file['final_nodes']
                    
                #     return resp,paramsarray,missingqs,twochains,nodes

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

                #order of configs to access
                order = list(range(len(all_configs)))
                random.shuffle(order)

                ######
                #do fourier transform (3D)
                ######
                print("start 3D FT")

                numKpoints = 19
                feed_in_configs = all_configs[order[:Nreads]]
                total_K, kx_vals, ky_vals, S_spin, S2_spin, S3_spin, structure_factor = Sq_for_record(feed_in_configs, final_qubit_dictionary, qubit_variables, numKpoints)

                ####
                #saving
                ####

                filename_hs = where_to_save + "3D_FT_h=" + hview + "_s=" + sview + ".hdf5"

                try:
                    os.remove(filename_hs)
                except OSError:
                    pass
                with h5py.File(Path(filename_hs), "w") as f:
                    f.create_dataset("kx", data = kx_vals)
                    f.create_dataset("ky", data = ky_vals)
                    f.create_dataset("avg_abs_sigma", data = S_spin)
                    f.create_dataset("avg_sigmasigma", data = S2_spin)
                    f.create_dataset("avg_abs_sigmasigma", data = S3_spin)
                    f.create_dataset("avg_sigma", data = structure_factor)


                ######
                #do fourier transform (3D)
                ######
                print("start 1D FT")

                numKpoints = 40
                feed_in_configs = all_configs[order[:Nreads]]
                total_K, kx_vals, ky_vals, S_spin, S2_spin, S3_spin, structure_factor = line_BZ(feed_in_configs, final_qubit_dictionary, qubit_variables, numKpoints)

                ####
                #saving
                ####

                filename_hs = where_to_save + "line_FT_h=" + hview + "_s=" + sview + ".hdf5"

                try:
                    os.remove(filename_hs)
                except OSError:
                    pass
                with h5py.File(Path(filename_hs), "w") as f:
                    f.create_dataset("kx", data = kx_vals)
                    f.create_dataset("ky", data = ky_vals)
                    f.create_dataset("avg_abs_sigma", data = S_spin)
                    f.create_dataset("avg_sigmasigma", data = S2_spin)
                    f.create_dataset("avg_abs_sigmasigma", data = S3_spin)
                    f.create_dataset("avg_sigma", data = structure_factor)
                    
                print("done with h/J=" + hview + " and s=" + sview)

    else:

        dict_h = {}
        all_hoverJ_vals = []
        all_h_str = []
        mainfolder = list(Path(folder_global).glob('*'))
        #mainfolder = list(Path('./data/raw_apq_zx').glob('*'))
        for foldername in mainfolder:
            h1 = str(foldername).split('/')[-1].split('=')[-1]
            if h1 != ".DS_Store" and h1 != ".DS_Store:Zone.Identifier":
                #print("h1=",h1)
                hJtag = "{:.3f}".format(float(h1)/Jmax)
                dict_h.update({hJtag:h1})
                all_hoverJ_vals.append(float(h1)/Jmax)
                all_h_str.append(h1)
        # print("various h/J in folder: " + folder_global)
        # print(dict_h.keys())


        # hoverJ_to_process = ['0.000', '2.004', '3.908', '5.211', '0.501', '2.505', '1.503', '3.006', '3.507', '1.002']
        hoverJ_to_process = ['0.0', '1.0', '2.0', '3.0', '4.0']


        Nreads = 200

        for hview in hoverJ_to_process:
            index = find_nearest(all_hoverJ_vals, float(hview))
            hval = all_h_str[index]

            folder = folder_global + "/h1=" + hval + "/"
            mainfolder = list(Path(folder).glob('*'))
            #mainfolder = list(Path('./data/raw_apq_zx').glob('*'))

            reps = 0
            folders_to_open = []
            if len(mainfolder) != 0:
                print("success in finding this folder")
            else:
                print('problem')
            for foldername in mainfolder:
                if str(foldername)[-4:] == ".npz":
                    #print(foldername)
                    reps += 1
                    folders_to_open.append(foldername)

            # def open_file(path):
                
            #     file=np.load(path)
                
            #     resp=file['resp']
            #     paramsarray=file['paramsarray']
            #     missingqs=file['missingqs']
            #     twochains=file['twochains']
            #     nodes=file['final_nodes']
                
            #     return resp,paramsarray,missingqs,twochains,nodes

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

            #order of configs to access
            order = list(range(len(all_configs)))
            random.shuffle(order)

            ######
            #do fourier transform (3D)
            ######
            print("start 3D FT")

            numKpoints = 19
            feed_in_configs = all_configs[order[:Nreads]]
            total_K, kx_vals, ky_vals, S_spin, S2_spin, S3_spin, structure_factor = Sq_for_record(feed_in_configs, final_qubit_dictionary, qubit_variables, numKpoints)

            ####
            #saving
            ####

            filename_hs = where_to_save + "3D_FT_J=" + str(Jmax) + "_h=" + hval +  ".hdf5"

            try:
                os.remove(filename_hs)
            except OSError:
                pass
            with h5py.File(Path(filename_hs), "w") as f:
                f.create_dataset("kx", data = kx_vals)
                f.create_dataset("ky", data = ky_vals)
                f.create_dataset("avg_abs_sigma", data = S_spin)
                f.create_dataset("avg_sigmasigma", data = S2_spin)
                f.create_dataset("avg_abs_sigmasigma", data = S3_spin)
                f.create_dataset("avg_sigma", data = structure_factor)


            ######
            #do fourier transform (3D)
            ######
            print("start 1D FT")

            numKpoints = 40
            feed_in_configs = all_configs[order[:Nreads]]
            total_K, kx_vals, ky_vals, S_spin, S2_spin, S3_spin, structure_factor = line_BZ(feed_in_configs, final_qubit_dictionary, qubit_variables, numKpoints)

            ####
            #saving
            ####

            filename_hs = where_to_save + "line_FT_J=" + str(Jmax) + "_h=" + hval +  ".hdf5"

            try:
                os.remove(filename_hs)
            except OSError:
                pass
            with h5py.File(Path(filename_hs), "w") as f:
                f.create_dataset("kx", data = kx_vals)
                f.create_dataset("ky", data = ky_vals)
                f.create_dataset("avg_abs_sigma", data = S_spin)
                f.create_dataset("avg_sigmasigma", data = S2_spin)
                f.create_dataset("avg_abs_sigmasigma", data = S3_spin)
                f.create_dataset("avg_sigma", data = structure_factor)
                
            print("done with J=" + str(Jmax) + " and h=" + hval)


