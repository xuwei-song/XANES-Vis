# %%
from utils import *

# %%
file1 = 'MnCO3.dat'
data = read_and_process_dat_file(file1)
features = process_multiple_spectra(data)

# %%
# Save features to a file
np.savetxt('xanes_features.csv', features, delimiter=',', header='edge_energy,edge_slope,wl_energy,wl_intensity,curvature_wl,pit_energy,pit_intensity,curvature_pit', comments='')

print(f"Features extracted and saved. Shape of features array: {features.shape}")

# Optionally, plot a few spectra with their features
num_plots = min(5, data.shape[1] - 1)  # Plot up to 5 spectra
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 4*num_plots), sharex=True)
for i in range(num_plots):
    axs[i].plot(data[:, 0], data[:, i+1], label=f'Spectrum {i+1}')
    axs[i].plot(features[i, 0], data[np.argmin(np.abs(data[:, 0] - features[i, 0])), i+1], 'ro', label='Edge (A)')
    axs[i].plot(features[i, 2], features[i, 3], 'go', label='White Line (B)')
    axs[i].plot(features[i, 5], features[i, 6], 'bo', label='Pit (C)')
    axs[i].set_ylabel('Normalized XANES')
    axs[i].legend()
axs[-1].set_xlabel('Energy (eV)')
plt.suptitle('XANES Feature Extraction for Multiple Spectra')
plt.tight_layout()
plt.show()

# %%
