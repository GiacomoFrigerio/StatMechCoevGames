import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
lambda_val = 1

for Dg_arrival in [-1,1]:
    for Dr_arrival in [-1,1]:
        print(f"Dg_arrival={Dg_arrival}, Dr_arrival={Dr_arrival}")

        # --- Equation and derivative ---
        def equation(rho, Dg, Dr):
            return (
                rho**(lambda_val + 1) * (Dg - Dr + Dr_arrival - Dg_arrival)
                + rho**lambda_val * (Dr - Dr_arrival)
                + rho * (Dr - Dg)
                - Dr
            )

        def derivative(rho, Dg, Dr):
            term1 = (lambda_val + 1) * rho**lambda_val * (Dg - Dr + Dr_arrival - Dg_arrival)
            term2 = lambda_val * rho**(lambda_val - 1) * (Dr - Dr_arrival)
            term3 = (Dr - Dg)
            return term1 + term2 + term3

        # --- Root finder ---
        def find_roots(Dg, Dr, rho_grid=np.linspace(0,1,1000)):
            f_vals = equation(rho_grid, Dg, Dr)
            roots = []
            for k in range(len(rho_grid)-1):
                if f_vals[k]*f_vals[k+1] < 0:
                    r = rho_grid[k] - f_vals[k]*(rho_grid[k+1]-rho_grid[k])/(f_vals[k+1]-f_vals[k])
                    roots.append(r)
            roots = sorted([r for r in roots if 0 <= r <= 1])
            if len(roots)==0:
                return [np.nan, np.nan]
            elif len(roots)==1:
                return [roots[0], np.nan]
            else:
                return roots[:2]

        # --- Grid setup ---
        Dg_vals = np.linspace(-1,1,200)
        Dr_vals = np.linspace(-1,1,200)
        D_G, D_R = np.meshgrid(Dg_vals, Dr_vals)

        RHO_LESSER = np.full_like(D_G, np.nan)
        RHO_LARGER = np.full_like(D_G, np.nan)
        STABLE_LESSER = np.full_like(D_G, np.nan)
        STABLE_LARGER = np.full_like(D_G, np.nan)

        # --- Solve on grid ---
        for i in range(D_G.shape[0]):
            for j in range(D_G.shape[1]):
                Dg = D_G[i,j]
                Dr = D_R[i,j]
                r1,r2 = find_roots(Dg,Dr)
                RHO_LESSER[i,j] = r1
                RHO_LARGER[i,j] = r2
                if not np.isnan(r1):
                    STABLE_LESSER[i,j] = 1 if derivative(r1,Dg,Dr)<0 else 0
                if not np.isnan(r2):
                    STABLE_LARGER[i,j] = 1 if derivative(r2,Dg,Dr)<0 else 0

        # --- Plot ---
        fig, axes = plt.subplots(2, 2, figsize=(14,12))

        # Roots heatmaps
        im0 = axes[0,0].imshow(RHO_LESSER, extent=[-1,1,-1,1], origin='lower', aspect='auto', cmap='viridis')
        axes[0,0].set_title('Lesser root ρ(Dg,Dr)')
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

        plt.tight_layout()
        plt.show()
