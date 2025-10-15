import matplotlib.pyplot as plt
import numpy as np

def plot_game_regions():
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlim(0, 2)
    ax.set_ylim(-1, 1)

    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$S$')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Fill regions
    # HG region
    hg_x = [0, 1, 1, 0]
    hg_y = [0, 0, 1, 1]
    ax.fill(hg_x, hg_y, color='green', alpha=0.2)

    # SH region
    sh_x = [0, 1, 1, 0]
    sh_y = [-1, -1, 0, 0]
    ax.fill(sh_x, sh_y, color='blue', alpha=0.2)

    # PD region
    pd_x = [1, 2, 2, 1]
    pd_y = [-1, -1, 0, 0]
    ax.fill(pd_x, pd_y, color='red', alpha=0.2)

    # SD region (strictly above S=0, below S=1-T)
    T_vals = np.linspace(1, 2, 100)
    S_lower = np.maximum(0, 1 - T_vals)  # Ensure S >= 0
    S_upper = np.ones_like(S_lower)
    ax.fill_between(T_vals, S_lower, S_upper, color='orange', alpha=0.2)

    # Game boundaries
    ax.axvline(x=1, color='green', linestyle='--')
    ax.axhline(y=0, color='green', linestyle='--')
    ax.plot([1, 2], [0, 0], color='red', linewidth=3)
    ax.plot([1, 2], [1, 0], color='blue', linestyle=':')

    # Labels
    ax.text(0.5, 0.5, '(HG)', fontsize=12, ha='center', va='center')
    ax.text(1.5, 0.5, 'SD', fontsize=12, ha='center', va='center')
    ax.text(0.5, -0.5, 'SH', fontsize=12, ha='center', va='center')
    ax.text(1.5, -0.5, 'PD', fontsize=12, ha='center', va='center')

    ax.grid(False)
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig("ts_game_regions.png", dpi=150)
    plt.show()

plot_game_regions()
