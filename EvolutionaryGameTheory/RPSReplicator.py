import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm

# --- Payoff matrices for different RPS regimes ---
payoff_matrices = {
    'Neutral oscillations': np.array([[0, -1,  1],
                                      [1,  0, -1],
                                      [-1, 1,  0]]),
    'Damped oscillations':  np.array([[0, -1,  1],
                                      [1,  0, -0.5],
                                      [-0.5, 1, 0]]),
    'Heteroclinic cycle':   np.array([[0, -1,  1],
                                      [1,  0, -1.5],
                                      [-1.5, 1, 0]])
}

def replicator_dynamics(x, A):
    f = A @ x
    phi = np.dot(x, f)
    return x * (f - phi)

def simulate_simplex_dynamics(A, x0, steps=5000, dt=0.01):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(steps):
        x += dt * replicator_dynamics(x, A)
        x = np.clip(x, 0, 1)
        x /= x.sum()
        traj.append(x.copy())
    return np.array(traj)

# Barycentric → Cartesian
def barycentric_to_cartesian(p):
    s = (p[:, 0] + p[:, 1] + p[:, 2])
    x = 0.5 * (2 * p[:, 1] + p[:, 2]) / s
    y = (np.sqrt(3) / 2) * p[:, 2] / s
    return x, y

# --- Figure (large & clean, border inits) ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
titles = list(payoff_matrices.keys())
colors = cm.plasma(np.linspace(0, 1, 6))

for ax, title in zip(axes, titles):
    A = payoff_matrices[title]
    ax.set_title(title, fontsize=16, fontweight='bold')

    # simplex triangle
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    ax.add_patch(Polygon(tri, closed=True, fill=None, edgecolor='black', linewidth=1.8))

    # border initial conditions (your logic preserved)
    rhos = np.linspace(0.2, 0.8, 6) if title == "Neutral oscillations" else np.linspace(0.1, 0.9, 6)

    for i, rho in enumerate(rhos):
        x0 = np.array([rho, 1 - rho, 0.01], dtype=float)
        x0 /= x0.sum()
        traj = simulate_simplex_dynamics(A, x0)
        X, Y = barycentric_to_cartesian(traj)
        ax.plot(X, Y, color=colors[i], linewidth=2.6, alpha=0.95)

    # vertex labels
    ax.text(-0.05, -0.05, 'Rock',     ha='right',  va='top',    fontsize=14)
    ax.text(1.05,  -0.05, 'Paper',    ha='left',   va='top',    fontsize=14)
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Scissors',  ha='center', va='bottom', fontsize=14)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')

plt.suptitle("Replicator Dynamics in the Rock–Paper–Scissors Game\nBorder Initial Condition",
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Save high-quality raster & vector
plt.savefig("RockPaperScissors2_large.png", dpi=400, bbox_inches='tight')
plt.savefig("RockPaperScissors2_large.pdf", bbox_inches='tight')
plt.show()
