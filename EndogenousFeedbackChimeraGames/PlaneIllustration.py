import matplotlib.pyplot as plt
import numpy as np

def plot_dilemma_strength_plane():
    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    # Limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$D_g$ (gamble-intending dilemma)', fontsize=16, labelpad=10)
    ax.set_ylabel(r'$D_r$ (risk-averting dilemma)', fontsize=16, labelpad=10)

    # Fill quadrants (HG, SD, SH, PD)
    ax.fill_betweenx([0, 1], 0, 1, color='red', alpha=0.25)
    ax.fill_betweenx([-1, 0], 0, 1, color='orange', alpha=0.25)
    ax.fill_betweenx([0, 1], -1, 0, color='blue', alpha=0.25)
    ax.fill_betweenx([-1, 0], -1, 0, color='green', alpha=0.25)

    # Axes
    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=1.2)

    # Region labels (bigger and bold)
    ax.text(0.5, 0.5, 'PD', fontsize=22, ha='center', va='center', color='darkred', fontweight='bold')
    ax.text(0.5, -0.5, 'SD', fontsize=22, ha='center', va='center', color='darkorange', fontweight='bold')
    ax.text(-0.5, 0.5, 'SH', fontsize=22, ha='center', va='center', color='navy', fontweight='bold')
    ax.text(-0.5, -0.5, 'HG', fontsize=22, ha='center', va='center', color='darkgreen', fontweight='bold')

    # Ticks: only -1, 0, 1 with larger font
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)

    # Style
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("dg_dr_plane.png", dpi=300)
    plt.show()

plot_dilemma_strength_plane()
