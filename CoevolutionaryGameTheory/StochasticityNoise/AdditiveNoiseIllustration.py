import numpy as np
import matplotlib.pyplot as plt

# === Distributions from Eqs. (2)–(4) ===
def xi_uniform(chi, a=1.0):
    return a * (-2.0*chi + 1.0)

def xi_exponential(chi, a=1.0):
    chi = np.clip(chi, 1e-12, 1.0)
    return a * (-np.log(chi) - 1.0)

def xi_powerlaw(chi, a=1.0, n=2):
    chi = np.clip(chi, 1e-12, 1.0)
    return a * (chi**(-1.0/n) - n/(n-1.0))

# === Generate one row of ξ for each distribution ===
def cross_section(L=300, a=1.0, dist="uniform", seed=0):
    rng = np.random.default_rng(seed)
    chi = rng.random(L)
    if dist == "uniform":
        xi = xi_uniform(chi, a)
    elif dist == "exponential":
        xi = xi_exponential(chi, a)
    elif dist == "powerlaw":
        xi = xi_powerlaw(chi, a, n=2)
    else:
        raise ValueError("dist must be 'uniform', 'exponential', or 'powerlaw'")
    return xi

# === Plot Figure 2 cross-sections ===
def plot_figure2_crosssections(L=300, a=1.0, seed=42, ylimits=(0,5)):
    dists = ["uniform", "exponential", "powerlaw"]
    labels = {"uniform":"uniform (u)", "exponential":"exponential (e)", "powerlaw":"power-law (p)"}
    xs = np.arange(L)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, dist in zip(axes, dists):
        xi = cross_section(L=L, a=a, dist=dist, seed=seed)
        ax.plot(xs, xi, lw=1)
        ax.set_ylim(ylimits)
        ax.set_ylabel(r"$\xi_i$")
        ax.set_title(f"{labels[dist]}, a={a}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("L (cross-section index)")
    plt.tight_layout()
    plt.savefig("CrossSections", dpi=100)
    plt.show()

# Run with fixed y-range
plot_figure2_crosssections(L=300, a=1.0, seed=42, ylimits=(0,5))
