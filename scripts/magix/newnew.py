import numpy as np
from scipy.integrate import odeint
import torch
import matplotlib.pyplot as plt
from .dynamic import nnMTModule
from .inference import FMAGI

# ----------------------
# True FitzHugh–Nagumo dynamics
# ----------------------
def FN_true(y, t, a=0.2, b=0.2, c=3.0):
    V, R = y
    dVdt = c * (V - V**3 / 3.0 + R)
    dRdt = -1.0 / c * (V - a + b * R)
    return np.array([dVdt, dRdt])

# ----------------------
# Generate noisy observations
# ----------------------
t_max = 40
n_points = 1281
noise_level = 0.1

t_grid = np.linspace(0, t_max, n_points)
x_clean = odeint(FN_true, [-1.0, 1.0], t_grid)
y_obs = np.column_stack((t_grid, x_clean))  # each row [t, V, R]

torch.set_default_dtype(torch.double)
# Build and run MAGI-X inference
# ys: list of tensors [n x 2] for each component: [time, value]
ys = []
for i in range(1, y_obs.shape[1]):
    data = torch.tensor(y_obs[:, [0, i]], dtype=torch.double)
    ys.append(data)

dim = 2
hidden_layers = [512]
# Initialize neural ODE model
dynamic_model = nnMTModule(dim, hidden_layers)

print("Before MAGI-X:", next(dynamic_model.parameters())[0,0].item())
# Run MAGI-X via FMAGI
magi = FMAGI(ys, dynamic_model, grid_size=201, interpolation_orders=3)
# Infer latent states and train the ODE network
t_grid_inf, x_inferred = magi.map(max_epoch=2500, learning_rate=1e-3,
                                  decay_learning_rate=True,
                                  hyperparams_update=False,
                                  dynamic_standardization=True,
                                  verbose=True, returnX=True)
# The network weights in dynamic_model are now updated by MAGI-X
print("After MAGI-X: ", next(dynamic_model.parameters())[0,0].item())

# Initial condition from inferred trajectory
y0 = x_inferred[0]
# Forecast using the trained MAGI-X model
t_pred, y_pred = magi.predict(tp=np.linspace(0, t_max, n_points),
                                t0=t_grid_inf, x0=x_inferred)

# ----------------------
# Vector field and nullcline utilities
# ----------------------
def compute_true_vector_field(grid_size=20, x_range=(-2.5,2.5), y_range=(-2.0,2.0)):
    x = np.linspace(*x_range, grid_size)
    y = np.linspace(*y_range, grid_size)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_size):
        for j in range(grid_size):
            dx = FN_true([X[i,j], Y[i,j]], 0)
            U[i,j], V[i,j] = dx
    return X, Y, U, V

def compute_estimated_vector_field(model, grid_size=20,
                                    x_range=(-2.5,2.5), y_range=(-2.0,2.0)):
    x = np.linspace(*x_range, grid_size)
    y = np.linspace(*y_range, grid_size)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_size):
        for j in range(grid_size):
            inp = torch.tensor([X[i,j], Y[i,j]], dtype=torch.double)
            with torch.no_grad():
                dx = model(inp).numpy().ravel()
            U[i,j], V[i,j] = dx
    return X, Y, U, V

def compute_estimated_nullclines(model, resolution=200,
                                  x_range=(-2.5,2.5), y_range=(-2.0,2.0)):
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(resolution):
        for j in range(resolution):
            inp = torch.tensor([X[i,j], Y[i,j]], dtype=torch.double)
            with torch.no_grad():
                dx = model(inp).numpy().ravel()
            U[i,j], V[i,j] = dx
    return {'X': X, 'Y': Y, 'U': U, 'V': V}

# Analytical nullclines for FitzHugh–Nagumo
def true_nullclines(x_null):
    y_null1 = x_null**3 / 3 - x_null
    y_null2 = (0.2 - x_null) / 0.2
    return y_null1, y_null2

# ----------------------
# Compute fields and nullclines
# ----------------------
t_true = compute_true_vector_field()
t_est  = compute_estimated_vector_field(dynamic_model)
nulls_est = compute_estimated_nullclines(dynamic_model)
x_null = np.linspace(-2.5, 2.5, 400)
y_null1, y_null2 = true_nullclines(x_null)

# ----------------------
# Plot vector fields and nullclines
# ----------------------
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# True field
axes[0].quiver(*t_true, cmap='viridis', scale=25)
axes[0].plot(x_null, y_null1, 'r--', label='V-nullcline')
axes[0].plot(x_null, y_null2, 'g--', label='R-nullcline')
axes[0].set_title('True Vector Field')
axes[0].set_xlabel('V')
axes[0].set_ylabel('R')
axes[0].set_xlim([-2.5,2.5])
axes[0].set_ylim([-2.0,2.0])
axes[0].legend()

# Estimated field
axes[1].quiver(*t_est, cmap='plasma', scale=25)
# plot true nullclines for reference
axes[1].plot(x_null, y_null1, 'r--', alpha=0.5)
axes[1].plot(x_null, y_null2, 'g--', alpha=0.5)
# plot MAGI-X estimated nullclines
axes[1].contour(nulls_est['X'], nulls_est['Y'], nulls_est['U'], levels=[0], colors=['r'], linewidths=1.2)
axes[1].contour(nulls_est['X'], nulls_est['Y'], nulls_est['V'], levels=[0], colors=['g'], linewidths=1.2)
axes[1].set_title('MAGI-X Estimated Field & Nullclines')
axes[1].set_xlabel('V')
axes[1].set_ylabel('R')
axes[1].set_xlim([-2.5,2.5])
axes[1].set_ylim([-2.0,2.0])

plt.tight_layout()
plt.savefig('vector_fields.png', dpi=300)
plt.show()
print("Done.")
