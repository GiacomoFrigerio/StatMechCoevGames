import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
epsilon = 0.1
theta = 2
R, S, T, P = 3, 0, 5, 1
delta_TR = T - R  # = 2
delta_PS = P - S  # = 1

# System
def system(t, y):
    x, n = y
    dxdt = (1/epsilon) * x * (1 - x) * (delta_PS + (delta_TR - delta_PS) * x) * (1 - 2 * n)
    dndt = n * (1 - n) * (-1 + (1 + theta) * x)
    return [dxdt, dndt]

# Time and evaluation
t_span = [0, 50]
t_eval = np.linspace(*t_span, 10000)

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# LEFT PANEL: Time Series (preserve)
initial_condition = [0.8, 0.1]
sol = solve_ivp(system, t_span, initial_condition, t_eval=t_eval, rtol=1e-8)

ax1.plot(sol.t, sol.y[0], label='Fraction of Cooperators $x$', color='blue')
ax1.plot(sol.t, sol.y[1], label='Environment $n$', color='green')
ax1.set_xlabel('Time')
ax1.set_ylabel('State')
ax1.set_ylim(-0.05, 1.05)
ax1.set_title('Time Series of $x$ and $n$')
ax1.legend()
ax1.grid(True)

# RIGHT PANEL: Phase Portrait with Streamlines
x_vals = np.linspace(0, 1, 25)
n_vals = np.linspace(0, 1, 25)
X, N = np.meshgrid(x_vals, n_vals)

U = (1/epsilon) * X * (1 - X) * (delta_PS + (delta_TR - delta_PS) * X) * (1 - 2 * N)
V = N * (1 - N) * (-1 + (1 + theta) * X)

speed = np.sqrt(U**2 + V**2)
U_norm = U / (speed + 1e-8)
V_norm = V / (speed + 1e-8)

ax2.streamplot(X, N, U_norm, V_norm, color='gray', density=1.2, arrowsize=1)

# Overlay trajectories from selected initial conditions for clear orbits
initial_conditions = [
    [0.8, 0.1],
    [0.7, 0.3],
    [0.6, 0.5],
    [0.5, 0.7],
    [0.4, 0.9],
    [0.3, 0.2],
    [0.2, 0.4]
]

# for ic in initial_conditions:
#     sol_traj = solve_ivp(system, t_span, ic, t_eval=t_eval, rtol=1e-8)
#     ax2.plot(sol_traj.y[0], sol_traj.y[1], lw=1.5)

# Mark interior fixed point
x_star = 1 / (1 + theta)  # = 1/3
n_star = 0.5
ax2.plot(x_star, n_star, 'r*', markersize=12, label='Interior FP')

ax2.set_xlabel('Fraction of Cooperators $x$')
ax2.set_ylabel('Environment $n$')
ax2.set_title('Phase Plane Dynamics with Streamlines')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("WeitzTragedy", dpi=120)
plt.show()
