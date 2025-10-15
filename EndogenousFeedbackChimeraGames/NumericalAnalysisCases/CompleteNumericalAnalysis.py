import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- Check static admissibility ---
def static_admissible(Dg_eq, Dr_eq, rho):
    """
    Returns True if rho is admissible in static scenario for given Dg_eq, Dr_eq
    """
    if 0 <= Dg_eq <= 1 and 0 <= Dr_eq <= 1:
        return np.isclose(rho, 0)
    elif -1 <= Dg_eq < 0 and 0 <= Dr_eq <= 1:
        return np.isclose(rho, 0) or np.isclose(rho, 1)
    elif 0 <= Dg_eq <= 1 and -1 <= Dr_eq < 0:
        return 0 < rho < 1
    elif -1 <= Dg_eq < 0 and -1 <= Dr_eq < 0:
        return np.isclose(rho, 1)
    else:
        return False  # out of defined static scenario


# --- Equation and derivative ---
def equation(rho, Dg, Dr):
    return (rho**(lambda_val + 1) * (Dg - Dr + Dr_arrival - Dg_arrival)
            + rho**lambda_val * (Dr - Dr_arrival)
            + rho * (Dr - Dg)
            - Dr)

def derivative(rho, Dg, Dr):
    return ((lambda_val + 1) * rho**lambda_val * (Dg - Dr + Dr_arrival - Dg_arrival)
            + lambda_val * rho**(lambda_val - 1) * (Dr - Dr_arrival)
            + (Dr - Dg))

# --- Root finder ---
def find_roots(Dg, Dr, rho_grid=np.linspace(0, 1, 1000)):
    f_vals = equation(rho_grid, Dg, Dr)
    roots = []
    for k in range(len(rho_grid)-1):
        if f_vals[k]*f_vals[k+1] < 0:
            r = rho_grid[k] - f_vals[k]*(rho_grid[k+1]-rho_grid[k])/(f_vals[k+1]-f_vals[k])
            roots.append(r)
    roots = sorted([r for r in roots if 0 <= r <= 1])
    if len(roots) == 0:
        return [np.nan, np.nan]
    elif len(roots) == 1:
        return [roots[0], np.nan]
    else:
        return roots[:2]


# --- Constants ---
lambda_val = 1
# Arrival values
Dg_arrival = -1
Dr_arrival = 1

# --- Grid setup ---
Dg_vals = np.linspace(-1, 1, 200)
Dr_vals = np.linspace(-1, 1, 200)
D_G, D_R = np.meshgrid(Dg_vals, Dr_vals)

# --- Storage ---
RHO_LESSER = np.full_like(D_G, np.nan)
RHO_LARGER = np.full_like(D_G, np.nan)
STABLE_LESSER = np.full_like(D_G, np.nan)
STABLE_LARGER = np.full_like(D_G, np.nan)
NUM_SOLUTIONS = np.zeros_like(D_G, dtype=int)

# --- Solve on grid ---
for i in range(D_G.shape[0]):
    for j in range(D_G.shape[1]):
        Dg, Dr = D_G[i,j], D_R[i,j]
        r1, r2 = find_roots(Dg, Dr)
        RHO_LESSER[i,j] = r1
        RHO_LARGER[i,j] = r2
        NUM_SOLUTIONS[i,j] = np.sum(~np.isnan([r1, r2]))
        if not np.isnan(r1):
            STABLE_LESSER[i,j] = 1 if derivative(r1,Dg,Dr)<0 else 0
        if not np.isnan(r2):
            STABLE_LARGER[i,j] = 1 if derivative(r2,Dg,Dr)<0 else 0

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14,12))

# Roots heatmaps
im0 = axes[0,0].imshow(RHO_LESSER, extent=[-1,1,-1,1], origin='lower', aspect='auto', cmap='viridis')
axes[0,0].set_title('Smaller root ρ(Dg,Dr)')
axes[0,0].set_xlabel('D_g'); axes[0,0].set_ylabel('D_r')
fig.colorbar(im0, ax=axes[0,0])

im1 = axes[0,1].imshow(RHO_LARGER, extent=[-1,1,-1,1], origin='lower', aspect='auto', cmap='viridis')
axes[0,1].set_title('Larger root ρ(Dg,Dr)')
axes[0,1].set_xlabel('D_g'); axes[0,1].set_ylabel('D_r')
fig.colorbar(im1, ax=axes[0,1])

# Stability using contourf with discrete levels
levels = [-0.5,0.5,1.5]  # 0 and 1
cmap_stab = plt.cm.Reds

cf0 = axes[1,0].contourf(D_G,D_R,STABLE_LESSER, levels=levels, cmap=cmap_stab)
axes[1,0].set_title('Stability of smaller root')
axes[1,0].set_xlabel('D_g'); axes[1,0].set_ylabel('D_r')
fig.colorbar(cf0, ax=axes[1,0], ticks=[0,1], label='Stable=1')

cf1 = axes[1,1].contourf(D_G,D_R,STABLE_LARGER, levels=levels, cmap=cmap_stab)
axes[1,1].set_title('Stability of larger root')
axes[1,1].set_xlabel('D_g'); axes[1,1].set_ylabel('D_r')
fig.colorbar(cf1, ax=axes[1,1], ticks=[0,1], label='Stable=1')

plt.suptitle(f"Dg_arrival={Dg_arrival}, Dr_arrival={Dr_arrival}", fontsize=16)
plt.tight_layout()

filename = f'solutions_stability_Dg({Dg_arrival})_Dr({Dr_arrival}).png'
fig.savefig(filename, dpi=300, bbox_inches='tight')
files.download(filename)
plt.show()

# --- Plot number of solutions using contourf ---
levels_ns = [-0.5, 0.5, 1.5, 2.5]
colors_ns = ['red', 'yellow', 'green']

plt.figure(figsize=(7,6))
cf_ns = plt.contourf(D_G, D_R, NUM_SOLUTIONS, levels=levels_ns, colors=colors_ns)
cbar = plt.colorbar(cf_ns, ticks=[0,1,2])
cbar.set_label('Number of solutions')
plt.xlabel('D_g')
plt.ylabel('D_r')
plt.title(f'Number of roots per grid point\nDg_arrival={Dg_arrival}, Dr_arrival={Dr_arrival}')

filename = f'numberofsolutions_Dg({Dg_arrival})_Dr({Dr_arrival}).png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
files.download(filename)
plt.show()

# --- Phase-line plots with function and arrows ---
rep_points = []
for n_sol in [0,1,2]:
    indices = np.argwhere(NUM_SOLUTIONS == n_sol)
    if len(indices) > 0:
        idx = indices[np.random.randint(len(indices))]
        Dg, Dr = D_G[idx[0], idx[1]], D_R[idx[0], idx[1]]
        rep_points.append({'Dg': Dg, 'Dr': Dr, 'n_sol': n_sol})

fig, axes = plt.subplots(1, len(rep_points), figsize=(4*len(rep_points),5))
if len(rep_points) == 1: axes = [axes]

rho_plot = np.linspace(0,1,500)
for ax, pt in zip(axes, rep_points):
    Dg, Dr, n_sol = pt['Dg'], pt['Dr'], pt['n_sol']
    roots = find_roots(Dg, Dr)
    f_vals = equation(rho_plot, Dg, Dr)
    ax.plot(rho_plot, f_vals, color='black', lw=1.5)
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.hlines(0, 0, 1, color='lightgray', alpha=0.5)
    ax.set_xlim(0,1)
    ax.set_ylim(-2*np.max(np.abs(f_vals)), 2*np.max(np.abs(f_vals)))
    ax.set_yticks([])
    ax.set_xlabel('ρ')
    ax.set_title(f"{n_sol} solution(s)\nDg={Dg:.2f}, Dr={Dr:.2f}\n"
                  f"Dg_arrival={Dg_arrival}, Dr_arrival={Dr_arrival}")

    if len(roots) == 0:
        mid_val = equation(0.5,Dg,Dr)
        if mid_val > 0:
            ax.annotate('', xy=(1,0), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        else:
            ax.annotate('', xy=(0,0), xytext=(1,0),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    else:
        for r in roots:
            ax.plot(r,0,'ko',markersize=8)
            deriv = derivative(r,Dg,Dr)
            arrow_length = 0.1
            if deriv < 0:  # stable
                ax.annotate('', xy=(r,0), xytext=(r-arrow_length,0),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax.annotate('', xy=(r,0), xytext=(r+arrow_length,0),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2))
            else:  # unstable
                ax.annotate('', xy=(r-arrow_length,0), xytext=(r,0),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))
                ax.annotate('', xy=(r+arrow_length,0), xytext=(r,0),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()

filename = f'stability_deltapayoff_Dg({Dg_arrival})_Dr({Dr_arrival}).png'
fig.savefig(filename, dpi=300, bbox_inches='tight')
files.download(filename)
plt.show()


# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot for smaller root ---
VALID_EQ_LESS = np.full_like(RHO_LESSER, False, dtype=bool)

for i in range(D_G.shape[0]):
    for j in range(D_G.shape[1]):
        rho = RHO_LESSER[i, j]
        if not np.isnan(rho):
            Dg_eq = D_G[i, j]*(1 - rho) + Dg_arrival*rho
            Dr_eq = D_R[i, j]*(1 - rho) + Dr_arrival*rho
            if static_admissible(Dg_eq, Dr_eq, rho):
                VALID_EQ_LESS[i, j] = True

# axes[0].scatter(D_G[~VALID_EQ_LESS], D_R[~VALID_EQ_LESS],
#                 s=6, c='lightgray', alpha=0.3, label='Not admissible')
# axes[0].scatter(D_G[VALID_EQ_LESS], D_R[VALID_EQ_LESS],
#                 s=6, c='green', label='Admissible')
# axes[0].set_xlabel('D_g')
# axes[0].set_ylabel('D_r')
# axes[0].set_title(f'Smaller root\nDg_arr={Dg_arrival}, Dr_arr={Dr_arrival}')
# axes[0].legend(frameon=False)
# axes[0].set_xlim(-1, 1)
# axes[0].set_ylim(-1, 1)

# --- Plot for larger root ---
VALID_EQ_LARGE = np.full_like(RHO_LARGER, False, dtype=bool)

for i in range(D_G.shape[0]):
    for j in range(D_G.shape[1]):
        rho = RHO_LARGER[i, j]
        if not np.isnan(rho):
            Dg_eq = D_G[i, j]*(1 - rho) + Dg_arrival*rho
            Dr_eq = D_R[i, j]*(1 - rho) + Dr_arrival*rho
            if static_admissible(Dg_eq, Dr_eq, rho):
                VALID_EQ_LARGE[i, j] = True

# axes[1].scatter(D_G[~VALID_EQ_LARGE], D_R[~VALID_EQ_LARGE],
#                 s=6, c='lightgray', alpha=0.3, label='Not admissible')
# axes[1].scatter(D_G[VALID_EQ_LARGE], D_R[VALID_EQ_LARGE],
#                 s=6, c='green', label='Admissible')
# axes[1].set_xlabel('D_g')
# axes[1].set_ylabel('D_r')
# axes[1].set_title(f'Larger root\nDg_arr={Dg_arrival}, Dr_arr={Dr_arrival}')
# axes[1].legend(frameon=False)
# axes[1].set_xlim(-1, 1)
# axes[1].set_ylim(-1, 1)

# filename = f'EGT_admissible_Dg({Dg_arrival})_Dr({Dr_arrival}).png'
# fig.savefig(filename, dpi=300, bbox_inches='tight')
# files.download(filename)
# plt.show()

# --- Masked: only non-admissible but stable solutions ---
# Smaller root: non-admissible and stable
NON_ADMISSIBLE_STABLE_LESS = (~VALID_EQ_LESS) & (~np.isnan(RHO_LESSER)) & (STABLE_LESSER == 1)

# Larger root: non-admissible and stable
NON_ADMISSIBLE_STABLE_LARG = (~VALID_EQ_LARGE) & (~np.isnan(RHO_LARGER)) & (STABLE_LARGER == 1)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(D_G[NON_ADMISSIBLE_STABLE_LESS], D_R[NON_ADMISSIBLE_STABLE_LESS],
                s=6, c='red', label='Stable non-admissible')
axes[0].set_xlabel('D_g')
axes[0].set_ylabel('D_r')
axes[0].set_title(f'Smaller root stable non-admissible\nDg_arr={Dg_arrival}, Dr_arr={Dr_arrival}')
axes[0].set_xlim(-1, 1)
axes[0].set_ylim(-1, 1)
axes[0].legend(frameon=False)

axes[1].scatter(D_G[NON_ADMISSIBLE_STABLE_LARG], D_R[NON_ADMISSIBLE_STABLE_LARG],
                s=6, c='red', label='Stable non-admissible')
axes[1].set_xlabel('D_g')
axes[1].set_ylabel('D_r')
axes[1].set_title(f'Larger root stable non-admissible\nDg_arr={Dg_arrival}, Dr_arr={Dr_arrival}')
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(-1, 1)
axes[1].legend(frameon=False)

filename = f'stable_non_admissible_Dg({Dg_arrival})_Dr({Dr_arrival}).png'
fig.savefig(filename, dpi=300, bbox_inches='tight')
files.download(filename)
plt.show()
