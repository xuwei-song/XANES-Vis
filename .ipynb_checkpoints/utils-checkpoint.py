#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import namedtuple
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

# 定义一个NamedTuple来保存参考光谱数据
Reference = namedtuple('Reference', ['data', 'name'])

def read_and_process_dat_file(filename, delimiter=' ', normalization_threshold=6585):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data_list = [line.strip().split(delimiter) for line in lines if line.strip()]
        data_array = np.array([[float(num) for num in row] for row in data_list])

    max_vals = np.max(data_array[:, 1:], axis=0)
    data_array[:, 1:] = data_array[:, 1:] / max_vals

    data_array = data_array[data_array[:, 0] <= normalization_threshold]

    return data_array

def process_reference_spectra(file2, file3):
    reference_Mn5O8 = read_and_process_dat_file(file2)
    reference_MnO2 = read_and_process_dat_file(file3)

    print(f"The data has {reference_Mn5O8.shape[0]} rows and {reference_Mn5O8.shape[1]} columns.")
    print(f"The data has {reference_MnO2.shape[0]} rows and {reference_MnO2.shape[1]} columns.")

    plt.figure(figsize=(10, 6))
    plt.plot(reference_Mn5O8[:,0], reference_Mn5O8[:,1], label='reference_Mn5O8 Data')
    plt.plot(reference_MnO2[:,0], reference_MnO2[:,1], label='reference_MnO2 Data')
    plt.xlabel('Energy')
    plt.ylabel('Normalized I0')
    plt.legend()
    plt.show()

def define_references(data):
    reference_MnCO3 = data[:, :2]  # 取第一张作为reference_MnCO3
    reference_Mn2O3 = np.column_stack((data[:, 0], data[:, -1]))  # 取最后一张作为reference_Mn2O3
    return Reference(reference_MnCO3[:, 1], 'MnCO3'), Reference(reference_Mn2O3[:, 1], 'Mn2O3')

def pca_reduction(X):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.T).T
#     print(X_reduced.shape)
    return X_reduced.T

def kmeans_clustering(X_reduced, n_clusters=4):
    kmeans = KMeans(n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_reduced)
    return labels

def ls_linear_combination_fit(data, references):
    def residuals(coeffs, data, references):
        fit = sum(c * ref.data for c, ref in zip(coeffs, references))
        penalty = 1e3 * (np.sum(coeffs) - 1)
        return np.append(data - fit, penalty)

    initial_guess = np.ones(len(references)) / len(references)
    bounds = (0, 1)
    result = least_squares(residuals, initial_guess, bounds=bounds, args=(data, references))

    fit = sum(c * ref.data for c, ref in zip(result.x, references))
    rss = np.sum((data - fit) ** 2)

    return result.x, rss

def process_cluster_means(cluster_means, references):
    coefficients = []
    rss_values = []
    for i in range(len(cluster_means)):
        coeffs, rss = ls_linear_combination_fit(cluster_means[i], references)
        coefficients.append(coeffs)
        rss_values.append(rss)

    coefficients = np.array(coefficients)
    R_values = np.array(rss_values)
    return coefficients, R_values

def plot_results(coefficients, R_values, references):
    print(coefficients)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    r = np.arange(len(coefficients))  # 聚类数目

    for i in range(coefficients.shape[1]):  # 参考光谱数目
        ref_name = references[i].name  # 获取当前参考光谱的名称
        ax1.bar(r + i * bar_width, coefficients[:, i] * 100, width=bar_width, label=ref_name)

    ax1.set_xlabel('Clusters', fontweight='bold')
    ax1.set_ylabel('%', fontweight='bold')
    ax1.set_xticks([r + bar_width * (coefficients.shape[1] - 1) / 2 for r in range(len(coefficients))])
    ax1.set_xticklabels([f'Cl. {i+1}' for i in range(len(coefficients))])

    ax2 = ax1.twinx()
    ax2.plot(r, R_values, color='orange', marker='o', label='R')

    ax2.set_ylabel('R', fontweight='bold')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()
    
def arctan(x, a, b, c, d):
    return a * np.arctan(b * (x - c)) + d

def extract_xanes_features(energy, intensity):
    # Normalize intensity
    intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    
    # Find edge energy (A)
    edge_index = np.argmax(np.gradient(intensity_norm))
    edge_energy = energy[edge_index]
    
    # Find white line (B) - look for the maximum after the edge
    wl_index = edge_index + np.argmax(intensity_norm[edge_index:])
    wl_intensity = intensity_norm[wl_index]
    wl_energy = energy[wl_index]
    
    # Find pit (C) - look for the minimum after the white line
    pit_index = wl_index + np.argmin(intensity_norm[wl_index:])
    pit_intensity = intensity_norm[pit_index]
    pit_energy = energy[pit_index]
    
    # Calculate edge slope
    popt, _ = curve_fit(arctan, energy, intensity_norm)
    edge_slope = popt[1]
    
    # Calculate curvature at B and C
    curvature_wl = np.gradient(np.gradient(intensity_norm))[wl_index]
    curvature_pit = np.gradient(np.gradient(intensity_norm))[pit_index]
    
    # Return features as a numpy array instead of a dictionary
    return np.array([
        edge_energy, edge_slope,
        wl_energy, wl_intensity, curvature_wl,
        pit_energy, pit_intensity, curvature_pit
    ])

def process_multiple_spectra(data):
    energy = data[:, 0]
    spectra = data[:, 1:]
    
    # Initialize an array to store features for all spectra
    all_features = np.zeros((spectra.shape[1], 8))  # 8 features per spectrum
    
    for i in range(spectra.shape[1]):
        all_features[i] = extract_xanes_features(energy, spectra[:, i])
    
    return all_features


# %%




