import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Setup
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
titles = [r"$\text{Choose one individual for reproduction (R) and one for death (D)}$",
          r"$\text{The offspring of the first individual replaces the second}$"]
colors = ['red'] * 6 + ['blue'] * 4  # 6 red, 4 blue

# Shuffle to randomize placement
np.random.seed(42)
np.random.shuffle(colors)

positions = [(i % 5, 1 - i // 5) for i in range(10)]
chosen_repro = 2
chosen_death = 9

for ax, title in zip(axes, titles):
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)

# Left panel
for i, (x, y) in enumerate(positions):
    color = colors[i]
    edgecolor = 'black'
    if i == chosen_repro:
        label = 'R'
    elif i == chosen_death:
        label = 'D'
    else:
        label = ''
    axes[0].add_patch(plt.Circle((x, y), 0.4, color=color, ec=edgecolor))
    if label:
        axes[0].text(x, y, label, color='white', ha='center', va='center', fontsize=10, weight='bold')

# Right panel
colors_new = colors.copy()
colors_new[chosen_death] = colors[chosen_repro]  # Replace with offspring (color of reproducing individual)

for i, (x, y) in enumerate(positions):
    color = colors_new[i]
    axes[1].add_patch(plt.Circle((x, y), 0.4, color=color, ec='black'))

# Add arrow from R to D
x_start, y_start = positions[chosen_repro]
x_end, y_end = positions[chosen_death]
axes[1].annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                 arrowprops=dict(arrowstyle="->", color='orange', lw=2))

plt.tight_layout()
plt.savefig("MoranProc1", dpi = 1000)
plt.show()
