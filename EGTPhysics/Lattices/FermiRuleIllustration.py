import numpy as np
import matplotlib.pyplot as plt

# Fixed value of π_x (as in the figure)
pi_x = 10.0

# Range of π_y - π_x
delta_pi = np.linspace(-70, 50, 500)
pi_y = pi_x + delta_pi  # So π_y - π_x = delta_pi

# Different Ky values to plot
Ky_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
colors = ['blue', 'green', 'red', 'gold', 'black']
markers = ['d', 'o', 'o', 's', '.']
linestyles = ['-', '-', '-', '-', '--']
labels = [r'$K = 0.1$', r'$K = 1.0$', r'$K = 10.0$', r'$K = 100.0$', r'$K = 1000.0$']

plt.figure(figsize=(10, 6))

for Ky, color, marker, ls, label in zip(Ky_values, colors, markers, linestyles, labels):
    W = 1 / (1 + np.exp((pi_y - pi_x) / Ky))
    plt.plot(delta_pi, W, label=label, color=color, marker=marker, linestyle=ls, markersize=4)

# Plot settings
plt.xlabel(r'$\pi_y - \pi_x$', fontsize=14)
plt.ylabel(r'$W(s_y \leftarrow s_x)$', fontsize=14)
plt.title('Fermi Update Rule for Various $K$', fontsize=15)
plt.grid(True)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("fermi_update_plot.png", dpi=100)
plt.show()
