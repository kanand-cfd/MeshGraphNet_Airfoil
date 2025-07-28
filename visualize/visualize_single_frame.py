import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# Load static mesh #
mesh_path = "data/Airfoil"
grid = np.load(f"{mesh_path}/test_grid.npy")
cells = np.load(f"{mesh_path}/test_cells.npy")

# Load dynamic data
data = np.load(f"{mesh_path}/test_data.npy")

print(f"Grid shape: {grid.shape}, Cells shape: {cells.shape}, Data shape: {data.shape}")

# Choose one frame to visualize
airfoil_id = 50
time_step = 15

# === Slice ===
node_pos = grid[airfoil_id, :, time_step, :]          # (5233, 2)
triangles = cells[airfoil_id, :, time_step, :]        # (10216, 3)
velocity = data[airfoil_id, :, time_step, :2]         # (5233, 2)
pressure = data[airfoil_id, :, time_step, 2]          # (5233,)

# Compute velocity magnitude
vel_mag = np.linalg.norm(velocity, axis=1)

# Trianglulation for plotting
tri = Triangulation(node_pos[:,0], node_pos[:, 1], triangles)

# Plot 1: Mesh and Velocity vectors
plt.figure(figsize=(8, 6))
contour = plt.tricontourf(tri, vel_mag, levels=50, cmap='plasma')
plt.triplot(tri, color='k', linewidth=0.2)  # overlay mesh
plt.colorbar(contour, label='Velocity Magnitude')
plt.title("Velocity Magnitude Field with Mesh")
plt.axis('equal')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 0.5)
plt.tight_layout()
plt.show()



# Plot 2: Pressure Field
plt.figure(figsize=(8, 6))
p_contour = plt.tricontourf(tri, pressure, levels=50, cmap='viridis')
plt.triplot(tri, color='k', linewidth=0.2)  # overlay mesh
plt.colorbar(p_contour, label='Pressure')
plt.title("Pressure Field with Mesh")
plt.axis('equal')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 0.5)
plt.tight_layout()
plt.show()