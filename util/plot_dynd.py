import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Read the data from the file
filename = './ED_16_sites_QFI_test/output/time_corr_rand0_Sz_Gamma_step16246.dat'

beta=10

# Skip comment lines and load data
data = []
with open(filename, 'r') as file:
    for line in file:
        if line.startswith('#') or line.startswith('//'):
            continue
        values = [float(val) for val in line.strip().split()]
        data.append(values)

# Convert to numpy array for easier manipulation
data = np.array(data)
omega = data[:, 0]
spectral_function_real = data[:, 1]
spectral_function_imag = data[:, 2]

# Combine real and imaginary parts to create complex spectral function
spectral_function = spectral_function_real + 1j * spectral_function_imag


omega_f = np.fft.fftfreq(len(omega), d=(omega[1]-omega[0])/(2*np.pi))
spectral_function_f = np.fft.fft(spectral_function, norm="ortho")

# Plot the spectral function
plt.figure(figsize=(10, 6))
plt.plot(omega_f, spectral_function_f.real, label='Real Part', color='blue')
plt.plot(omega_f, spectral_function_f.imag, label='Imaginary Part', color='red')
plt.xlabel('Frequency (ω)', fontsize=14)
plt.ylabel('Spectral Function', fontsize=14)
plt.title('Spectral Function vs Frequency', fontsize=16)
plt.legend()
plt.show()  