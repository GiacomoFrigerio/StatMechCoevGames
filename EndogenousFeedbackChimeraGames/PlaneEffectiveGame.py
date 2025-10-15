import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import root_scalar

# --- Parameters ---
Dg_arrival = 0.3
Dr_arrival = 0.3
lambda_val = 1
max_roots = 2

# --- Equation to solve ---
def equation(rho, Dg, Dr):
    try:
        return (
            rho**(lambda_val + 1) * (Dg - Dr + Dr_arrival - Dg_arrival)
            + rho**lambda_val * (Dr - Dr_arrival)
            + rho * (Dr - Dg)
            - Dr
        )
    except:
        return np.nan

# --- Root finder ---
def find_all_roots(Dg, Dr, num_points=100):
    rho_vals = np.linspace(0, 1, num_points)
    f_vals = [equation(rho, Dg, Dr) for rho in rho_vals]
    roots = []
    for i in range(len(rho_vals) - 1):
        if np.isnan(f_vals[i]) or np.isnan(f_vals[i+1]):
            continue
        if f_vals[i] * f_vals[i+1] < 0:
            try:
                sol = root_scalar(lambda rho: equation(rho, Dg, Dr),
                                  bracket=[rho_vals[i], rho_vals[i+1]],
                                  method='brentq')
                if sol.converged:
                    root = sol.root
                    if not any(np.isclose(root, r, atol=1e-5) for r in roots):
                        roots.append(root)
            except:
                continue
    return roots

# --- Grid setup ---
Dg_vals = np.linspace(-1, 1, 300)
Dr_vals = np.linspace(-1, 1, 300)
D_G, D_R = np.meshgrid(Dg_vals, Dr_vals)

# --- Storage for roots and transformed points ---
RHO_branches = [np.full_like(D_G, np.nan) for _ in range(max_roots)]
Dg_star_branches = [np.full_like(D_G, np.nan) for _ in range(max_roots)]
Dr_star_branches = [np.full_like(D_G, np.nan) for _ in range(max_roots)]

# --- Compute roots and effective games ---
for i in range(D_G.shape[0]):
    for j in range(D_G.shape[1]):
        Dg = D_G[i, j]
        Dr = D_R[i, j]
        roots = find_all_roots(Dg, Dr)
        for k in range(min(len(roots), max_roots)):
            rho = roots[k]
            rho_lambda = rho ** lambda_val
            RHO_branches[k][i, j] = rho
            Dg_star_branches[k][i, j] = (1 - rho_lambda) * Dg + rho_lambda * Dg_arrival
            Dr_star_branches[k][i, j] = (1 - rho_lambda) * Dr + rho_lambda * Dr_arrival

# --- Plot 2D transformed points and quivers ---
fig, ax = plt.subplots(figsize=(6.5, 6.5))

# Region masks
HG_mask_sh = (D_G < 0) & (D_R < 0) & (D_G < D_R)
HG_mask_cg = (D_G < 0) & (D_R < 0) & (D_G > D_R)
PD_mask = (D_G > 0) & (D_R > 0)
CG_mask = (D_G > 0) & (D_R < 0)
SH_mask = (D_G < 0) & (D_R > 0)

region_colors = {
    'HG_sh': '#80FF00',
    'HG_cg': '#009900',
    'PD': '#FF0000',
    'CG': '#FF9933',
    'SH': '#0000FF'
}

region_masks = [
    (HG_mask_sh, 'HG_sh'),
    (HG_mask_cg, 'HG_cg'),
    (PD_mask, 'PD'),
    (CG_mask, 'CG'),
    (SH_mask, 'SH'),
]

# Plot each root projection by region
for k in range(max_roots):
    for mask, region in region_masks:
        ax.scatter(
            Dg_star_branches[k][mask],
            Dr_star_branches[k][mask],
            color=region_colors[region],
            alpha=0.4,
            s=12
        )

# Starting points
starting_points = [
    (-0.8, -0.3),  # HG
#    (0.8, 0.5),    # PD
    (-0.3, -0.8),  # HG 2
    (0.75, -0.7),  # CG
    (-0.8, 0.2),   # SH
]

# Quiver from each starting point for each root
for Dg0, Dr0 in starting_points:
    if Dg0 < 0 and Dr0 < 0 and Dr0 > Dg0:
        color = region_colors['HG_sh']
    elif Dg0 < 0 and Dr0 < 0 and Dr0 < Dg0:
        color = region_colors['HG_cg']
    elif Dg0 > 0 and Dr0 > 0:
        color = region_colors['PD']
    elif Dg0 > 0 and Dr0 < 0:
        color = region_colors['CG']
    elif Dg0 < 0 and Dr0 > 0:
        color = region_colors['SH']
    else:
        color = 'gray'

    roots = find_all_roots(Dg0, Dr0)
    for rho in roots:
        rho_lambda = rho ** lambda_val
        Dg_star = (1 - rho_lambda) * Dg0 + rho_lambda * Dg_arrival
        Dr_star = (1 - rho_lambda) * Dr0 + rho_lambda * Dr_arrival
        dx = Dg_star - Dg0
        dy = Dr_star - Dr0
        ax.quiver(Dg0, Dr0, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', width=0.007)
        ax.plot(Dg0, Dr0, 'o', color=color, markersize=20, markeredgecolor='black')

# hard coded valori su SH and PD

# ax.plot(0.8, 0.5, 'o', color=region_colors['PD'], markersize=20, markeredgecolor='black')
ax.plot(-0.7, 0.2, 'o', color=region_colors['SH'], markersize=20, markeredgecolor='black')

# Axis and region formatting
ax.set_xlabel(r"$D_g$", fontsize=30)
ax.set_ylabel(r"$D_r$", fontsize=30)
ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
ax.plot([-1, 0], [-1, 0], color='red', linestyle='--', linewidth=0.5)

ax.text(0.5, 0.7, 'PD', fontsize=30, ha='center', va='center')
ax.text(-0.5, 0.7, 'SH', fontsize=30, ha='center', va='center')
ax.text(-0.5, -0.7, 'HG', fontsize=30, ha='center', va='center')
ax.text(0.5, -0.7, 'SD', fontsize=30, ha='center', va='center')

# Arrival point
ax.plot(Dg_arrival - 0.02, Dr_arrival - 0.02, marker='s', color='red', markersize=20)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xticks([-1, 0, 1])
ax.set_yticks([0, 1])
ax.tick_params(axis='both', labelsize=26)
plt.tight_layout()
plt.savefig("SquareIncentives_MultiRoot.svg", format="svg")
plt.show()
