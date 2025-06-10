import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio  # For exporting to VTK

# ========== STEP 1: Load Custom Geometry ==========
geometry_file = "geometry.txt"
data = np.loadtxt(geometry_file, delimiter="\t", skiprows=1, usecols=(1, 2, 3))  # Ignore Node column

# Extract X, Y, Z coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Convert to nondimensional form
L = 0.2  # Characteristic length (meters)
points_nondim = np.column_stack((X, Y, Z)) / L  # Scale by characteristic length

# ========== STEP 2: Visualize the Geometry ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points_nondim[:, 0], points_nondim[:, 1], points_nondim[:, 2],
           c="blue", marker="o", s=1, alpha=0.5)
ax.set_xlabel("X (nondim)")
ax.set_ylabel("Y (nondim)")
ax.set_zlabel("Z (nondim)")
ax.set_title("3D Visualization of Imported Geometry")
plt.show()

# ========== STEP 3: Define DeepXDE Geometry ==========
epsilon = 1e-2  # Small tolerance for boundary selection
boundary_mask = np.isclose(points_nondim[:, 2], 0, atol=epsilon)
boundary_points = points_nondim[boundary_mask]

# Ensure boundary points exist
if len(boundary_points) == 0:
    raise ValueError("No boundary points found near z = 0. Check epsilon value!")

# Create PointCloud geometry with defined boundary points
geom = dde.geometry.PointCloud(points_nondim, boundary_points=boundary_points)

# ========== STEP 4: Load and Normalize Temperature Data ==========
temp_file = "surfacetemp2.txt"
temp_data = np.loadtxt(temp_file, delimiter="\t", skiprows=1, usecols=(1, 2, 3, 4), encoding="latin1")

# Extract X, Y, Z coordinates and temperature
X_temp, Y_temp, Z_temp, Temperature = temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], temp_data[:, 3]

# Normalize positions using characteristic length
temp_points_nondim = np.column_stack((X_temp, Y_temp, Z_temp)) / L  # Convert to nondimensional coordinates

# Normalize temperature
T0 = 40  # Reference temperature in °C
temperature_nondim = Temperature / T0  # Nondimensionalized temperature

# ========== STEP 5: Define Characteristic Scales ==========
# Define trainable material properties
rhoc = 7850.0 * 434.0

kap = torch.nn.Parameter(torch.tensor(50000, dtype=torch.float32, requires_grad=True)) # W/(m*K)


# Nondimensionalized power rate
power_rate_dim = 800  # W/m³
power_rate = power_rate_dim * L**2 / (T0)

# Trainable parameters using torch.nn.Parameter
depth = torch.nn.Parameter(torch.tensor(0.04 / L, dtype=torch.float32, requires_grad=True))
radius = torch.nn.Parameter(torch.tensor(0.003 / L, dtype=torch.float32, requires_grad=True))

# Fixed coordinates for the heat source center
x_coor = torch.tensor(0.089 / L)
y_coor = torch.tensor(0.092 / L)

t_bottom = 24.4 / T0  # Bottom temperature (nondimensional)

# ========== STEP 6: Define PDE ==========
def pde(X, T):
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=0, j=1)
    dT_zz = dde.grad.hessian(T, X, i=0, j=2)

    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    x0, y0, z0 = x_coor, y_coor, depth
    R = radius  

    r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    source = (power_rate / (4 / 3 * np.pi * R**3)) * torch.relu(1 - r / R)

    return (dT_xx + dT_yy + dT_zz) - source/kap

# ========== STEP 7: Define Boundary Conditions ==========
def dir_func_base(X):
    return t_bottom * np.ones((len(X), 1))

def base_boundary(X, on_boundary):
    if X.ndim == 1:
        return on_boundary and np.isclose(X[2], 0, atol=epsilon)
    else:
        return on_boundary and np.isclose(X[:, 2], 0, atol=epsilon)

bc_base = dde.DirichletBC(geom, dir_func_base, base_boundary)

# ========== STEP 8: Define Temperature Observations ==========
observe = dde.PointSetBC(temp_points_nondim, temperature_nondim.reshape(-1, 1), component=0)

# ========== STEP 9: Define Data & Model ==========
data = dde.data.PDE(
    geom,
    pde,
    [bc_base, observe],
    num_domain=10000,
    num_boundary=5000,
    num_test=1000,
)

net = dde.nn.FNN([3] + [256] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# ========== STEP 10: Trainable Variables ==========
radius_history = []
depth_history = []

def track_variables():
    radius_history.append(radius.item() * L)
    depth_history.append(depth.item() * L)

variable = dde.callbacks.VariableValue(
    [radius, depth, kap], period=500, filename="variables.dat"
)

loss_weights = [0.1, 1.0, 10.0]
optimizer = torch.optim.Adam([
    {"params": model.net.parameters(), "lr": 0.001},  # Standard LR for the network
    {"params": [radius, depth, kap], "lr": 0.01}  # Higher LR for trainable parameters
])
model.compile(optimizer, loss_weights=loss_weights, external_trainable_variables=[radius, depth, kap])

losshistory, train_state = model.train(iterations=1000, callbacks=[variable], display_every=500)
track_variables()  # Store trained values

# ========== STEP 11: L-BFGS Fine-Tuning ==========
model.compile("L-BFGS", external_trainable_variables=[radius, depth, kap])
losshistory_lbfgs, train_state_lbfgs = model.train(iterations=1000)
track_variables()  # Store L-BFGS optimized values
model.save("realbreast_LBFGS.ckpt")

# ========== STEP 12: Save & Plot Training Results ==========
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


































device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using: {'GPU' if torch.cuda.is_available() else 'CPU'}")

print(f"Final optimized radius: {radius.item() * L/(-2.2)}")
print(f"Final optimized depth: {depth.item() * L/0.53}")
print(f"Final optimized kap: {kap.item()/0.8}")
# Plot Training Loss
iterations = range(len(losshistory.loss_train) + len(losshistory_lbfgs.loss_train))
full_loss = losshistory.loss_train + losshistory_lbfgs.loss_train

plt.figure(figsize=(8, 5))
plt.plot(iterations, full_loss, label="Training Loss", color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve with L-BFGS Fine-Tuning")
plt.legend()
plt.grid()
plt.show()

# Plot Evolution of Trainable Variables
plt.figure(figsize=(8, 5))
plt.plot(range(len(radius_history)), radius_history, label="Optimized Radius", color="red")
plt.plot(range(len(depth_history)), depth_history, label="Optimized Depth", color="blue")
plt.xlabel("Iterations")
plt.ylabel("Value (meters)")
plt.title("Evolution of Trainable Variables")
plt.legend()
plt.grid()
plt.show()

# ========== STEP 13: 3D Visualization for ParaView ==========
# Evaluate the model at the geometry points (nondimensional)
T_pred_nondim = model.predict(points_nondim)
T_pred_dim = T_pred_nondim * T0  # Convert back to °C

# Convert points back to dimensional coordinates for export
points_dim = points_nondim * L

# --- Export to VTK using meshio ---
# Here we create a VTK file that represents a point cloud with an associated temperature field.
# The VTK file can be opened in ParaView.
cells = [("vertex", np.arange(points_dim.shape[0]).reshape(-1, 1))]
mesh = meshio.Mesh(points_dim, cells, point_data={"Temperature": T_pred_dim.flatten()})
mesh.write("temperature_distribution.vtk")
print("VTK file 'temperature_distribution.vtk' saved. Open this file in ParaView to visualize the temperature distribution.")
