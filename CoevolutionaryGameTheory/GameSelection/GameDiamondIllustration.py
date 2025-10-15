import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 8))

# Diamond vertices
T = np.array([2, 3, 2, 1, 2])
S = np.array([2, 1, 0, 1, 2])

# Fill quadrants with specified colors and alpha=0.3
ax.fill([2, 3, 2], [2, 1, 1], color='orange', alpha=0.2)   # SD
ax.fill([2, 3, 2], [0, 1, 1], color='red', alpha=0.2)      # PD
ax.fill([2, 1, 2], [0, 1, 1], color='blue', alpha=0.2)     # SH
ax.fill([2, 1, 2], [2, 1, 1], color='green', alpha=0.2)    # HG

# Diamond outline
ax.plot(T, S, 'k-', lw=1.5)

# Labels in each quadrant
ax.text(1.65, 1.25, 'HG', fontsize=20, ha='center', va='center', weight='bold')
ax.text(2.35, 1.25, 'SD', fontsize=20, ha='center', va='center', weight='bold')
ax.text(1.65, 0.75, 'SH', fontsize=20, ha='center', va='center', weight='bold')
ax.text(2.35, 0.75, 'PD', fontsize=20, ha='center', va='center', weight='bold')

# Diagonal boundary labels
ax.text(2.55, 1.4, r'$S = -T + \alpha$', fontsize=18, rotation=-45, ha='center')
ax.text(1.48, 1.5, r'$S = T$', fontsize=18, rotation=45, ha='center')
ax.text(1.45, 0.25, r'$S = -T + \beta$', fontsize=18, rotation=-45, ha='center')
ax.text(2.6, 0.25, r'$S = T - \alpha + \beta$', fontsize=18, rotation=45, ha='center')

# Axis labels and ticks
ax.set_xlabel(r'$T$', fontsize=20)
ax.set_ylabel(r'$S$', fontsize=20)
ax.set_xticks([1, 2, 3])
ax.set_yticks([0, 1, 2])
ax.set_xlim(0.8, 3.2)
ax.set_ylim(-0.2, 2.2)
ax.set_aspect('equal')

# Clean aesthetics
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

plt.tight_layout()
plt.savefig("GameDiamond", dpi=80)
plt.show()
