import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
lambda_val = 1

# Loop over arrival values
for Dg_arrival in [-1, 1]:
    for Dr_arrival in [-1, 1]:
        print(f"Dg_arrival={Dg_arrival}, Dr_arrival={Dr_arrival}")

        # --- Functions ---
        def equation(rho, Dg, Dr):
            return (rho**(lambda_val + 1) * (Dg - Dr + Dr_arrival - Dg_arrival)
                    + rho**lambda_val * (Dr - Dr_arrival)
                    + rho * (Dr - Dg)
                    - Dr)

        def derivative(rho, Dg, Dr):
            return ((lambda_val + 1) * rho**lambda_val * (Dg - Dr + Dr_arrival - Dg_arrival)
                    + lambda_val * rho**(lambda_val - 1) * (Dr - Dr_arrival)
                    + (Dr - Dg))

        def find_roots(Dg, Dr, rho_grid=np.linspace(0,1,1000)):
            f_vals = equation(rho_grid, Dg, Dr)
            roots = []
            for k in range(len(rho_grid)-1):
                if f_vals[k]*f_vals[k+1] < 0:
                    # linear interpolation
                    r = rho_grid[k] - f_vals[k]*(rho_grid[k+1]-rho_grid[k])/(f_vals[k+1]-f_vals[k])
                    roots.append(r)
            return sorted([r for r in roots if 0 <= r <= 1])

        # --- Grid ---
        Dg_vals_grid = np.linspace(-1,1,200)
        Dr_vals_grid = np.linspace(-1,1,200)
        D_G, D_R = np.meshgrid(Dg_vals_grid, Dr_vals_grid)
        NUM_SOLUTIONS = np.zeros_like(D_G, dtype=int)

        # --- Count solutions ---
        for i in range(D_G.shape[0]):
            for j in range(D_G.shape[1]):
                NUM_SOLUTIONS[i,j] = len(find_roots(D_G[i,j], D_R[i,j]))

        # --- Select representative points ---
        rep_points = []
        for n_sol in [0,1,2]:
            indices = np.argwhere(NUM_SOLUTIONS == n_sol)
            if len(indices) > 0:
                idx = indices[np.random.randint(len(indices))]
                Dg, Dr = D_G[idx[0], idx[1]], D_R[idx[0], idx[1]]
                rep_points.append({'Dg': Dg, 'Dr': Dr, 'n_sol': n_sol})

        # --- Plot phase-lines with function ---
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
        plt.show()
