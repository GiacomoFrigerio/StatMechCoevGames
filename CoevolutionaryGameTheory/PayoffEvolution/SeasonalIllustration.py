import matplotlib.pyplot as plt
import numpy as np

# Parameters for illustrative example
T1, S1 = 0.9, 0.1
T2, S2 = 1.5, -0.5
tau = 50
total_steps = 200

# Time axis
time = np.arange(total_steps)

# Plot
fig, ax = plt.subplots(figsize=(10, 2))

# Color blocks showing switching
for i in range(0, total_steps, tau):
    ax.axvspan(i, i+tau, color='lightblue' if ((i // tau) % 2 == 0) else 'lightcoral', alpha=0.4)

# Label phases clearly
ax.text(tau/2, 0.5, 'Harmony Game\n$(T_1, S_1)$', fontsize=12, ha='center', va='center')
ax.text(1.5 * tau, 0.5, 'Social Dilemma\n$(T_2, S_2)$', fontsize=12, ha='center', va='center')
ax.text(2.5 * tau, 0.5, 'Harmony Game\n$(T_1, S_1)$', fontsize=12, ha='center', va='center')
ax.text(3.5 * tau, 0.5, 'Social Dilemma\n$(T_2, S_2)$', fontsize=12, ha='center', va='center')

# Clean up axis
ax.set_ylim(0, 1)
ax.set_xlim(0, total_steps)
ax.set_yticks([])
ax.set_xlabel('Time steps')
ax.set_title('Qualitative Illustration of Seasonal Payoff Switching ($\\tau = {}$)'.format(tau))

plt.tight_layout()
plt.savefig("PayoffEvolution", dpi=60)
plt.show()
