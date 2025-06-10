import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== STEP 1: Load Custom Geometry ==========
geometry_file = "geometry.txt"
data = np.loadtxt(geometry_file, delimiter="\t", skiprows=1, usecols=(1, 2, 3))

# Extract X, Y, Z coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
L = 0.2  # Characteristic length
points_nondim = np.column_stack((X, Y, Z)) / L

# ========== STEP 2: Define DeepXDE Geometry ==========
epsilon = 1e-3 / L  
boundary_mask = np.isclose(points_nondim[:, 2], 0, atol=epsilon)
boundary_points = points_nondim[boundary_mask]

if len(boundary_points) == 0:
    raise ValueError("No boundary points found near z = 0. Check epsilon value!")

geom = dde.geometry.PointCloud(points_nondim, boundary_points=boundary_points)

# ========== STEP 3: Load Temperature Data ==========
temp_file = "surfacetemp2.txt"
temp_data = np.loadtxt(temp_file, delimiter="\t", skiprows=1, usecols=(1, 2, 3, 4), encoding="latin1")

X_temp, Y_temp, Z_temp, Temperature = temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], temp_data[:, 3]
temp_points_nondim = np.column_stack((X_temp, Y_temp, Z_temp)) / L

T0 = 40  # Reference temperature (°C)
temperature_nondim = Temperature / T0  # Convert to nondimensional

# ========== STEP 4: Define Characteristic Scales ==========
rhoc = 7850.0 * 434.0  # Density × Specific heat capacity (J/(m³·K))
kap = 60500.0  # Thermal conductivity (W/m·K)
alpha = kap / rhoc  # Thermal diffusivity (m²/s)
tau = L**2 / alpha  # Characteristic time scale (s)

power_rate_dim = 800  # W/m³
power_rate = power_rate_dim * (L**2) / (kap * T0)  # Non-dimensionalized heat source

depth = torch.nn.Parameter(torch.tensor(0.027 / L, dtype=torch.float32, requires_grad=True))
radius = torch.nn.Parameter(torch.tensor(0.00458 / L, dtype=torch.float32, requires_grad=True))

x_coor = torch.tensor(0.089 / L)
y_coor = torch.tensor(0.092 / L)

t_bottom = 24.4 / T0  # Bottom temperature (Non-dimensional)

# ========== STEP 5: Define Non-Dimensionalized PDE ==========
def pde(X, T):
    """Non-dimensionalized heat equation with a localized heat source"""
    
    # Compute second derivatives (Laplacian)
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=0, j=1)
    dT_zz = dde.grad.hessian(T, X, i=0, j=2)
    dT_t = dde.grad.jacobian(T, X, i=0, j=3)  # Time derivative

    # Extract spatial and time coordinates
    x, y, z, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # Non-dimensional heat source
    x0, y0, z0 = x_coor, y_coor, depth  # Heat source center
    R = radius  # Non-dimensional heat source radius

    r = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    source = (power_rate / (4 / 3 * np.pi * R**3)) * torch.relu(1 - r / R)  # Non-dimensional source

    # Non-dimensionalized heat equation
    return dT_t - (dT_xx + dT_yy + dT_zz) - source

# ========== STEP 6: Train Model ==========
data = dde.data.TimePDE(
    geom, pde, [], num_domain=1000, num_boundary=500, num_initial=500, num_test=1000
)

net = dde.nn.FNN([4] + [256] * 5 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

optimizer = torch.optim.Adam([
    {"params": model.net.parameters(), "lr": 0.0001},
    {"params": [radius, depth], "lr": 0.0002}  
])

loss_weights = [10.0, 1.0, 1.0]
model.compile(optimizer, loss_weights=loss_weights, external_trainable_variables=[radius, depth])

# Track variables
radius_history = []
depth_history = []
def track_variable_updates():
    radius_history.append(radius.item() * L)
    depth_history.append(depth.item() * L)
    print(f"Step {model.train_state.step}: Radius = {radius.item() * L}, Depth = {depth.item() * L}")

# Train with Adam
losshistory, train_state = model.train(iterations=5000, display_every=500, callbacks=[track_variable_updates])

# ========== STEP 7: L-BFGS Fine-Tuning ==========
for _ in range(5):  
    model.compile("L-BFGS", external_trainable_variables=[radius, depth])
    losshistory_lbfgs, train_state_lbfgs = model.train()
    
    if abs(radius.item() * L - 0.00458) < 1e-5 and abs(depth.item() * L - 0.027) < 1e-5:
        break  

# Save the model
model.save("realbreast_LBFGS.ckpt")

# ========== STEP 8: Save & Plot Results ==========
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using: {'GPU' if torch.cuda.is_available() else 'CPU'}")

print(f"Final optimized nondimensional radius: {radius.item() * L}")
print(f"Final optimized nondimensional depth: {depth.item() * L}")

# Plot Training Loss
iterations = range(len(losshistory.loss_train) + len(losshistory_lbfgs.loss_train))
full_loss = losshistory.loss_train + losshistory_lbfgs.loss_train

plt.figure(figsize=(8,5))
plt.plot(iterations, full_loss, label="Training Loss", color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve with L-BFGS Fine-Tuning")
plt.legend()
plt.grid()
plt.show()

# Plot Evolution of Trainable Variables
plt.figure(figsize=(8,5))
plt.plot(range(len(radius_history)), radius_history, label="Optimized Radius", color="red")
plt.plot(range(len(depth_history)), depth_history, label="Optimized Depth", color="blue")
plt.xlabel("Iterations")
plt.ylabel("Value (meters)")
plt.title("Evolution of Trainable Variables")
plt.legend()
plt.grid()
plt.show()
