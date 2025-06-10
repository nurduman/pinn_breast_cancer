import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== STEP 1: Load Custom Geometry ==========
geometry_file = "geometry.txt"
data = np.loadtxt(geometry_file, delimiter="\t", skiprows=1, usecols=(1, 2, 3))  # Ignore Node column

# Extract X, Y, Z coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Convert to nondimensional form
L = 0.05  # Characteristic length
points_nondim = np.column_stack((X, Y, Z)) / L  # Scale by characteristic length

# ========== STEP 2: Visualize the Geometry ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points_nondim[:, 0], points_nondim[:, 1], points_nondim[:, 2], c="blue", marker="o", s=1, alpha=0.5)
ax.set_xlabel("X (nondim)")
ax.set_ylabel("Y (nondim)")
ax.set_zlabel("Z (nondim)")
ax.set_title("3D Visualization of Imported Geometry")
plt.show()

# ========== STEP 3: Define DeepXDE Geometry ==========
# Identify boundary points where z ≈ 0 (for base boundary condition)
epsilon = 1e-3 / L  # Small tolerance for boundary selection
boundary_mask = np.isclose(points_nondim[:, 2], 0, atol=epsilon)
boundary_points = points_nondim[boundary_mask]

# Ensure boundary points exist
if len(boundary_points) == 0:
    raise ValueError("No boundary points found near z = 0. Check epsilon value!")

# Create PointCloud geometry with defined boundary points
geom = dde.geometry.PointCloud(points_nondim, boundary_points=boundary_points)

# ========== STEP 4: Load and Normalize Temperature Data ==========
temp_file = "surfacetemp.txt"
temp_data = np.loadtxt(temp_file, delimiter="\t", skiprows=1, usecols=(1, 2, 3, 4), encoding="latin1")  # Ignoring Node column

# Extract X, Y, Z coordinates and temperature
X_temp, Y_temp, Z_temp, Temperature = temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], temp_data[:, 3]

# Normalize positions using characteristic length (L = 0.05 m)
temp_points_nondim = np.column_stack((X_temp, Y_temp, Z_temp)) / L  # Convert to nondimensional coordinates

# Normalize temperature
T0 = 40  # Reference temperature in °C
temperature_nondim = Temperature / T0  # Nondimensionalized temperature

# ========== STEP 5: Define Characteristic Scales ==========
rhoc = 7850.0 * 434.0  # Density * specific heat capacity (J/(m^3*K))
kap = 60500.0  # Thermal conductivity (W/(m*K))

tau = (L**2 * rhoc) / kap  # Characteristic time scale (s)

# Nondimensionalized constants
power_rate_dim = 3369800
power_rate = power_rate_dim / (T0)

depth = torch.tensor(0.02 / L, requires_grad=True)
radius = torch.tensor(0.005 / L, requires_grad=True)

t_bottom = 24.4 / T0  # Bottom temperature

# ========== STEP 6: Define PDE ==========
def pde(X, T):
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=0, j=1)
    dT_zz = dde.grad.hessian(T, X, i=0, j=2)

    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    
    x0, y0, z0 = 0.0, 0.0, depth
    R = radius  

    r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

    source = (power_rate / (4 / 3 * np.pi * R**3)) * torch.relu(1 - r / R)

    return (dT_xx + dT_yy + dT_zz) - source

# ========== STEP 7: Define Boundary Conditions ==========
def dir_func_base(X):
    """Defines the fixed temperature at the base (z ≈ 0)."""
    return t_bottom * np.ones((len(X), 1))

def base_boundary(X, on_boundary):
    """Selects points where z ≈ 0 for Dirichlet BC, handling both 1D and 2D cases."""
    if X.ndim == 1:  # If X is 1D, check the third element directly
        return on_boundary and np.isclose(X[2], 0, atol=epsilon)
    else:  # If X is 2D, check the third column
        return on_boundary and np.isclose(X[:, 2], 0, atol=epsilon)

bc_base = dde.DirichletBC(geom, dir_func_base, base_boundary)

# ========== STEP 8: Define Temperature Observations ==========
observe = dde.PointSetBC(temp_points_nondim, temperature_nondim.reshape(-1, 1), component=0)

# ========== STEP 9: Define Data & Model ==========
data = dde.data.PDE(
    geom,
    pde,
    [bc_base, observe],
    num_domain=1000,
    num_boundary=500,
    num_test=1000,
)

net = dde.nn.FNN([3] + [256] * 5 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: abs(y))
model = dde.Model(data, net)

# ========== STEP 10: Trainable Variables ==========
external_trainable_variables = [radius, depth]
variable = dde.callbacks.VariableValue(
    external_trainable_variables, period=600, filename="variables.dat"
)

loss_weights = [5.0, 10.0, 20.0]
model.compile(
    "adam", lr=0.0001, loss_weights=loss_weights, external_trainable_variables=external_trainable_variables
)
losshistory, train_state = model.train(iterations=10000, callbacks=[variable])

# ========== STEP 11: Save & Plot Results ==========
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Print optimized parameters
print(f"Optimized nondimensional radius: {radius.item() * 0.05}")
print(f"Optimized nondimensional depth: {depth.item() * 0.05}")
