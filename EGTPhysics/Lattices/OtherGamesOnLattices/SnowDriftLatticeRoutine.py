import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm

# --- Parameters ---
L = 50  # lattice size
generations = 200  # total generations
updates_per_gen = L * L  # asynchronous updates
R, P = 1.0, 0.0
C, D = 1, 0
kernel = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]])  # von Neumann neighborhood (N=4)

# --- Functions ---
def initialize_grid():
    return np.random.choice([C, D], size=(L, L))

def compute_payoff(grid, b, c):
    coop = (grid == C).astype(int)
    nC = convolve2d(coop, kernel, mode='same', boundary='wrap')
    nD = 4 - nC
    coop_payoff = nC * (b - c / 2) + nD * (b - c)
    defe_payoff = nC * b
    return coop * coop_payoff + (1 - coop) * defe_payoff

def async_update(grid, b, c):
    for _ in range(updates_per_gen):
        x, y = np.random.randint(0, L, 2)
        neighbors = [((x - 1) % L, y), ((x + 1) % L, y),
                     (x, (y - 1) % L), (x, (y + 1) % L)]
        nx, ny = neighbors[np.random.randint(0, 4)]

        payoff = compute_payoff(grid, b, c)
        p_focal = payoff[x, y]
        p_neighbor = payoff[nx, ny]

        if p_neighbor > p_focal:
            grid[x, y] = grid[nx, ny]
    return grid

def run_simulation(r):
    b = 1.0
    c = 2 * r / (1 + r)
    grid = initialize_grid()
    for _ in range(generations):
        grid = async_update(grid, b, c)
    return np.mean(grid == C)

# --- Simulate for various r values ---
r_values = np.linspace(0.01, 0.99, 20)
coop_fractions = [run_simulation(r) for r in tqdm(r_values)]

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.plot(r_values, coop_fractions, 'ks-', label='Lattice Snowdrift (N=4)')
plt.plot(r_values, 1 - r_values, 'k--', label='Well-mixed: $1 - r$')
plt.xlabel("Cost-to-benefit ratio $r = c / (2b - c)$")
plt.ylabel("Fraction of cooperators at equilibrium")
plt.title("Snowdrift Game on a Square Lattice (Asynchronous Update)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("SnowdriftCostBenefitRatio", dpi=150)
plt.show()
