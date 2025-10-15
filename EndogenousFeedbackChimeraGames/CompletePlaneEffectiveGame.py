import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# --- Parameters ---
Dg_arrival = 1
Dr_arrival = 1
lambda_val = 1
max_roots = 1

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
        if np.isnan(f_vals[i]) or np.isnan(f_vals[i + 1]):
            continue
        if f_vals[i] * f_vals[i + 1] < 0:
            try:
                sol = root_scalar(lambda rho: equation(rho, Dg, Dr),
                                  bracket=[rho_vals[i], rho_vals[i + 1]],
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
fig, ax = plt.subplots(figsize=(7, 7))

# --- Region masks ---
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

# --- Plot transformed points for each root ---
for k in range(max_roots):
    for mask, region in region_masks:
        ax.scatter(
            Dg_star_branches[k][mask],
            Dr_star_branches[k][mask],
            color=region_colors[region],
            alpha=0.4,
            s=12
        )

# --- Starting points ---
base_points = [
    (-0.8, -0.3),  # HG
    (-0.3, -0.8),  # HG 2
    (0.7, -0.7),  # CG
    (-0.5, 0.75),   # SH
]

initial_cooperator_fractions = [0.1,0.42,0.74]

#starting_points = [ (x * (1-rho) + rho, y * (1-rho) + rho)
#    for (x, y) in base_points
#    for rho in initial_cooperator_fractions ]

all_roots = []

# --- Quiver arrows from each root ---
for Dg0, Dr0 in base_points:
    # Determine color from region
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
    all_roots.extend(all_roots)
    ax.plot(Dg0, Dr0, 'D', color=color, markersize=20, markeredgecolor='black')

    starting_points = [
    (Dg0 * (1-rho) + rho, Dr0 * (1-rho) + rho)
    for rho in initial_cooperator_fractions ]

    for rho in roots:
        rho_lambda = rho ** lambda_val
        Dg_star = (1 - rho_lambda) * Dg0 + rho_lambda * Dg_arrival
        Dr_star = (1 - rho_lambda) * Dr0 + rho_lambda * Dr_arrival
        dx = Dg_star - Dg0
        dy = Dr_star - Dr0
        ax.plot([Dg0, Dg_arrival], [Dr0, Dr_arrival],
                linestyle='--', color='black', linewidth=1.2, alpha=0.8)
        ax.plot(Dg_arrival - 0.02, Dr_arrival - 0.02, marker='s', color='red', markersize=20)
        ax.plot(Dg_star, Dr_star, marker='*', color='red', markersize=12)

        for x, y in starting_points:
            if x < 0 and y < 0 and y > x:
                color = region_colors['HG_sh']
            elif x < 0 and y < 0 and y < x:
                color = region_colors['HG_cg']
            elif x > 0 and y > 0:
                color = region_colors['PD']
            elif x > 0 and y < 0:
                color = region_colors['CG']
            elif x < 0 and y > 0:
                color = region_colors['SH']
            else:
                color = 'gray'
            dx = Dg_star - x
            dy = Dr_star - y
            ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', width=0.008)
            ax.plot(x, y, 'o', color=color, markersize=12, markeredgecolor='black')



# hard coded valori su SH and PD

# ax.plot(0.8, 0.5, 'o', color=region_colors['PD'], markersize=20, markeredgecolor='black')
# ax.plot(-0.7, 0.2, 'o', color=region_colors['SH'], markersize=20, markeredgecolor='black')

Dg0, Dr0 = (-0.5, 0.75)
ax.plot([Dg0, Dg_arrival], [Dr0, Dr_arrival],
                linestyle='--', color='black', linewidth=1.2, alpha=0.8)
starting_points = [
    (Dg0 * (1-rho) + rho, Dr0 * (1-rho) + rho)
    for rho in initial_cooperator_fractions ]

for x, y in starting_points:
  # Determine color from region
    if x < 0 and y < 0 and y > x:
        color = region_colors['HG_sh']
    elif x < 0 and y < 0 and y < x:
        color = region_colors['HG_cg']
    elif x > 0 and y > 0:
        color = region_colors['PD']
    elif x > 0 and y < 0:
        color = region_colors['CG']
    elif x < 0 and y > 0:
        color = region_colors['SH']
    else:
        color = 'gray'

    dx = Dg0 - x
    dy = Dr0 - y
    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', width=0.008)
    ax.plot(x, y, 'o', color=color, markersize=12, markeredgecolor='black')

ax.text(-0.8, -0.45, '(a)', fontsize=20, ha='center', va='center')
ax.text(-0.3, -0.93, '(b)', fontsize=20, ha='center', va='center')
ax.text(0.72, -0.84, '(c)', fontsize=20, ha='center', va='center')
ax.text(-0.64, 0.75, '(d)', fontsize=20, ha='center', va='center')

ax.set_xlabel(r"$D_g$", fontsize=30)
ax.set_ylabel(r"$D_r$", fontsize=30)
ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
ax.plot([-1, 0], [-1, 0], color='red', linestyle='--', linewidth=0.5)
ax.text(0.5, 0.5, 'PD', fontsize=30, ha='center', va='center')
ax.text(-0.5, 0.5, 'SH', fontsize=30, ha='center', va='center')
ax.text(-0.5, -0.5, 'HG', fontsize=30, ha='center', va='center')
ax.text(0.5, -0.5, 'SD', fontsize=30, ha='center', va='center')
ax.plot(Dg_arrival - 0.02, Dr_arrival - 0.02, marker='s', color='red', markersize=20)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xticks([-1, 0, 1])
ax.set_yticks([0, 1])
ax.tick_params(axis='both', labelsize=26)
plt.tight_layout()
plt.savefig("SquareIncentives_MultiRoot.svg", format="svg")
plt.show()
