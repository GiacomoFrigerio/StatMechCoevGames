import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Outer rectangle: Symmetric (matrix) games
outer = patches.Rectangle((0, 0), 8, 5, edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(outer)

# Middle rectangle: Nash equilibria
middle = patches.Rectangle((1, 1), 6, 3, edgecolor='black', facecolor='lightgrey', linewidth=2)
ax.add_patch(middle)

# Inner rectangle: Strict Nash equilibria
inner = patches.Rectangle((2, 2), 4, 1, edgecolor='black', facecolor='grey', linewidth=2)
ax.add_patch(inner)

# Add labels
#ax.text(8.5, 2.6, 'Symmetric (matrix) games', va='top', ha='left', fontsize=12)
ax.text(4, 3.5, 'Evolutionary stable strategies', va='center', ha='center', fontsize=12)
ax.text(4, 2.5, 'Strict Nash Equilibria', va='center', ha='center', fontsize=12)
ax.text(4, 0.4, 'Nash equilibria', va='center', ha='center', fontsize=12)

# Remove axes
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 5.5)
ax.axis('off')

plt.tight_layout()
plt.savefig("NashESS", dpi=200)
plt.show()
