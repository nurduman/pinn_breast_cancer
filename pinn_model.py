import deepxde as dde
import numpy as np
import torch
import os
import tempfile

# Define a function to run the PINN model
def run_pinn_model(
    conductivity: float,
    radius: float,
    depth: float,
    geometry_file_path: str,
    surface_temp_file_path: str
) -> dict:
    # Store trainable variable histories
    radius_history = []
    depth_history = []
    kap_history = []

    # ========== STEP 1: Load Custom Geometry ==========
    data = np.loadtxt(geometry_file_path, delimiter="\t", skiprows=1, usecols=(1, 2, 3))  # Ignore Node column
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
    L = 0.2  # Characteristic length (meters)
    points_nondim = np.column_stack((X, Y, Z)) / L  # Scale by characteristic length

    # ========== STEP 3: Define DeepXDE Geometry ==========
    epsilon = 1e-2  # Small tolerance for boundary selection
    boundary_mask = np.isclose(points_nondim[:, 2], 0, atol=epsilon)
    boundary_points = points_nondim[boundary_mask]

    if len(boundary_points) == 0:
        raise ValueError("No boundary points found near z = 0. Check epsilon value!")

    geom = dde.geometry.PointCloud(points_nondim, boundary_points=boundary_points)

    # ========== STEP 4: Load and Normalize Temperature Data ==========
    temp_data = np.loadtxt(surface_temp_file_path, delimiter="\t", skiprows=1, usecols=(1, 2, 3, 4), encoding="latin1")
    X_temp, Y_temp, Z_temp, Temperature = temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], temp_data[:, 3]
    temp_points_nondim = np.column_stack((X_temp, Y_temp, Z_temp)) / L
    T0 = 40  # Reference temperature in °C
    temperature_nondim = Temperature / T0

    # ========== STEP 5: Define Characteristic Scales ==========
    rhoc = 7850.0 * 434.0
    kap = torch.nn.Parameter(torch.tensor(conductivity, dtype=torch.float32, requires_grad=True))  # Use provided conductivity directly
    power_rate_dim = 800  # W/m³
    power_rate = power_rate_dim * L**2 / (T0)

    # Initialize trainable parameters with provided values
    depth_param = torch.nn.Parameter(torch.tensor(depth / L, dtype=torch.float32, requires_grad=True))
    radius_param = torch.nn.Parameter(torch.tensor(radius / L, dtype=torch.float32, requires_grad=True))

    x_coor = torch.tensor(0.089 / L)
    y_coor = torch.tensor(0.092 / L)
    t_bottom = 24.4 / T0

    # ========== STEP 6: Define PDE ==========
    def pde(X, T):
        dT_xx = dde.grad.hessian(T, X, i=0, j=0)
        dT_yy = dde.grad.hessian(T, X, i=0, j=1)
        dT_zz = dde.grad.hessian(T, X, i=0, j=2)
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        x0, y0, z0 = x_coor, y_coor, depth_param
        R = radius_param
        r = torch.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        source = (power_rate / (4 / 3 * np.pi * R**3)) * torch.relu(1 - r / R)
        return (dT_xx + dT_yy + dT_zz) - source / kap

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
        num_domain=10000,  # Match original
        num_boundary=5000,  # Match original
        num_test=1000,     # Match original
    )

    net = dde.nn.FNN([3] + [256] * 3 + [1], "tanh", "Glorot uniform")  # Match original
    model = dde.Model(data, net)

    # ========== STEP 10: Trainable Variables ==========
    def track_variables():
        current_radius = radius_param.item() * L
        current_depth = depth_param.item() * L
        current_kap = kap.item()
        radius_history.append(current_radius)
        depth_history.append(current_depth)
        kap_history.append(current_kap)
        print(f"Training Progress - Radius: {current_radius:.6f}, Depth: {current_depth:.6f}, Conductivity: {current_kap:.6f}")

    variable = dde.callbacks.VariableValue(
        [radius_param, depth_param, kap], period=500, filename="variables.dat"  # Match original
    )

    loss_weights = [0.1, 1.0, 10.0]  # Match original
    optimizer = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": 0.001},  # Match original
        {"params": [radius_param, depth_param, kap], "lr": 0.01}  # Match original
    ])
    model.compile(optimizer, loss_weights=loss_weights, external_trainable_variables=[radius_param, depth_param, kap])

    losshistory, train_state = model.train(iterations=1000, callbacks=[variable], display_every=500)  # Match original
    track_variables()

    # ========== STEP 11: L-BFGS Fine-Tuning ==========
    model.compile("L-BFGS", external_trainable_variables=[radius_param, depth_param, kap])
    losshistory_lbfgs, train_state_lbfgs = model.train(iterations=1000)  # Match original
    track_variables()

    # Get the final optimized values with higher precision
    final_radius = float(radius_param.item() * L)
    final_depth = float(depth_param.item() * L)
    final_kap = float(kap.item())

    # Print final values for debugging
    print(f"Final Values - Radius: {final_radius:.6f}, Depth: {final_depth:.6f}, Conductivity: {final_kap:.6f}")

    # Return with 6 decimal places for higher accuracy
    return {
        "depth": round(final_depth, 6),
        "radius": round(final_radius, 6),
        "conductivity": round(final_kap, 6)  # Return kap directly, no scaling
    }

# Note: The visualization and VTK export steps are omitted as they are not needed for the API

if __name__ == "__main__":
    # For standalone testing (optional)
    output = run_pinn_model(50000, 0.003, 0.04, "geometry.txt", "surfacetemp2.txt")
    print(output)