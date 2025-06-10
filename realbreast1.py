import deepxde as dde
import pandas as pd
import numpy as np
import torch

# ============================
# 1. Define characteristic scales and known parameters
# ============================
L = 0.05       # Characteristic length (m)
L1 = 0.06665   # Conversion factor for physical units (m)
T0 = 36.6      # Characteristic temperature (°C)
rhoc = 7850.0 * 434.0  # Density * specific heat capacity (J/(m^3*K))
kap = 60500.0  # Thermal conductivity (W/(m*K))
tau = (L**2 * rhoc) / kap  # Characteristic time (s) -- not used in steady state

# Known (true) parameters for the simulation (nondimensional)
radius = torch.tensor(0.0144 / L, requires_grad=True)  # nondimensional radius
depth  = torch.tensor(0.027 / L, requires_grad=True)    # nondimensional depth

power_rate_dim = 3369761
power_rate = power_rate_dim / (T0 * rhoc)

# For the boundary condition on nodes at the "bottom" surface,
# we want to prescribe 24.4°C. Convert to nondimensional:
t_surface = 24.4 / T0  # nondimensional temperature (~0.66667)

epsilon = 1e-3 / L  # tolerance for identifying boundary points

# ============================
# 2. Load geometry coordinates from file and create the point cloud
# ============================
# Read the file using pandas.
# The file is tab-separated and uses commas as decimal separators.
df = pd.read_csv("geom.txt", sep="\t", decimal=",")
print("Head of the loaded data:")
print(df.head())

# Extract the coordinate columns (adjust the column names if necessary)
coords = df[["X Location (m)", "Y Location (m)", "Z Location (m)"]].to_numpy()

# Nondimensionalize the coordinates (if the simulation expects nondimensional values)
coords_nondim = coords / L

# Determine the minimum nondimensional z-value in the point cloud.
z_min = np.min(coords_nondim[:, 2])
print("Minimum nondimensional z:", z_min)

# Create a point cloud geometry from the nondimensional coordinates.
geom = dde.geometry.PointCloud(coords_nondim)

# ============================
# 3. Define the boundary condition for the "bottom" surface
# ============================
# Here we impose that nodes whose nondimensional z coordinate equals z_min (within tolerance)
# are held at t_surface (i.e. 24.4°C in physical units).
def boundary_fn(x, on_boundary):
    # Check if x[2] is close to z_min within the tolerance epsilon.
    return on_boundary and np.isclose(x[2], z_min, atol=epsilon)

def dirichlet_bc(x):
    return t_surface * np.ones((len(x), 1))

bc_bottom = dde.DirichletBC(geom, dirichlet_bc, boundary_fn)

# ============================
# 4. Additional point boundary conditions for specific nodes
# ============================
# Define the extra boundary points in physical coordinates along with their prescribed temperatures.
extra_points_physical = np.array([
    [-0.089, 0.092, -0.745],
    [-0.089, 0.1,   -0.745],
    [-0.089, 0.084, -0.7]
])
# Convert these points to nondimensional coordinates.
extra_points_nondim = extra_points_physical / L

# Define the corresponding temperatures (in physical °C) and convert to nondimensional:
extra_temps_physical = np.array([27.066, 26.931, 27.431]).reshape(-1, 1)
extra_temps_nondim = extra_temps_physical / T0

# Create a point-set boundary condition for these extra points.
bc_extra = dde.PointSetBC(extra_points_nondim, extra_temps_nondim, component=0)

# Combine all boundary conditions into a list.
bcs = [bc_bottom, bc_extra]

# ============================
# 5. Define the steady-state PDE
# ============================
def pde(X, T):
    # Compute second derivatives with respect to x, y, and z
    dT_xx = dde.grad.hessian(T, X, i=0, j=0)
    dT_yy = dde.grad.hessian(T, X, i=0, j=1)
    dT_zz = dde.grad.hessian(T, X, i=0, j=2)
    
    # Unpack spatial coordinates (nondimensional)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    
    # Define the heat source center (nondimensional). Here, the source is centered at (0, depth, 0).
    x0, y0, z0 = 0.0, depth, 0.0
    R = radius  # nondimensional source radius
    
    # Compute the nondimensional distance from the source center
    r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    
    # Define the heat source term (active within the source region)
    source = (power_rate / ((4 / 3) * np.pi * R**3)) * torch.relu(1 - r / R)
    
    # Steady-state heat equation: Laplacian(T) + source = 0
    return dT_xx + dT_yy + dT_zz + source

# ============================
# 6. Set up the PDE data and model using the point cloud geometry and combined boundary conditions
# ============================
data = dde.data.PDE(geom, pde, bcs, num_test=1000)

# Define the neural network: 3 inputs (x, y, z) and 1 output (T)
net = dde.nn.FNN([3] + [256] * 5 + [1], "tanh", "Glorot uniform")
external_trainable_variables = [radius, depth]

model = dde.Model(data, net)
model.compile(
    optimizer="adam",
    lr=0.001,
    external_trainable_variables=external_trainable_variables,
)

losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

print(f"Optimized physical radius (m): {radius.item() * L1}")
print(f"Optimized physical depth (m): {depth.item() * L1}")
