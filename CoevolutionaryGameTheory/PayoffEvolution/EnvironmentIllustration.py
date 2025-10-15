import matplotlib.pyplot as plt
import numpy as np

# Parameters
time_steps = 200
time = np.arange(time_steps)

# Simulate environment fluctuation
resource_level = 0.5 + 0.4 * np.sin(2 * np.pi * time / 100)

# Environment affects the game:
# As resource increases:
# - Temptation to defect T decreases
# - Sucker's payoff S increases
T_high, T_low = 1.5, 1.0
S_low, S_high = -0.5, 0.5

T = T_high - 0.5 * resource_level
S = S_low + 1.0 * resource_level

# Create a color block plot for intuitive direct interpretation
fig, ax = plt.subplots(figsize=(10, 2))

# Fill background with color gradient indicating resource level
for i in range(time_steps - 1):
    color_intensity = resource_level[i]
    ax.axvspan(i, i + 1, color=(1 - color_intensity, 1, 1 - color_intensity), alpha=0.4)

# Add text labels
ax.text(50, 0.5, 'High Resources:\nLower T, Higher S', fontsize=12, ha='center', va='center')
ax.text(150, 0.5, 'Low Resources:\nHigher T, Lower S', fontsize=12, ha='center', va='center')

# Styling
ax.set_xlim(0, time_steps)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel('Time steps')
ax.set_title('Illustration: Environment Dynamically Modulates Game Payoffs')

plt.tight_layout()
plt.savefig("EnvironmentFeedback", dpi=60)
plt.show()
