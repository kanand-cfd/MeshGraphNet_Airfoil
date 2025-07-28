import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation

# Load data
data_dir = "data/Airfoil"
airfoil_id = 0

grid = np.load(f"{data_dir}/test_grid.npy")
cells = np.load(f"{data_dir}/test_cells.npy")
data = np.load(f"{data_dir}/test_data.npy")

n_timesteps = grid.shape[2]

node_pos = grid[airfoil_id, :, 0, :]   # fixed node positions
triangles = cells[airfoil_id, :, 0, :]  # triangle connectivity
tri = Triangulation(node_pos[:,0], node_pos[:,1], triangles)

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Initialize plots
v_plot = ax1.tricontourf(tri, np.zeros_like(node_pos[:,0]), levels=50, cmap='plasma')
p_plot = ax2.tricontourf(tri, np.zeros_like(node_pos[:,0]), levels=50, cmap='viridis')

# Mesh Overlay
#ax1.triplot(tri, color='k', linewidth=0.2)
#ax2.triplot(tri, color='k', linewidth=0.2)

# Axis setup
for ax, title in zip((ax1, ax2), ("Velocity Magnitude", "Pressure Field")):
    ax.triplot(tri, color='k', linewidth=0.2)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-1., 1.])
    ax.set_aspect('equal')
    ax.set_title(title)

n_levels=50

# === Initialize dummy contourf to attach colorbars ===
vel_mag_0 = np.linalg.norm(data[airfoil_id, :, 0, :2], axis=1)
pressure_0 = data[airfoil_id, :, 0, 2]

v_contour = ax1.tricontourf(tri, vel_mag_0, levels=n_levels, cmap='plasma')
p_contour = ax2.tricontourf(tri, pressure_0, levels=n_levels, cmap='viridis')

cbar1 = fig.colorbar(v_contour, ax=ax1, label="Velocity Magnitude")
cbar2 = fig.colorbar(p_contour, ax=ax2, label="Pressure")


def update(frame):
    global cbar1, cbar2

    velocity = data[airfoil_id, :, frame, :2]
    pressure = data[airfoil_id, :, frame, 2]
    vel_mag = np.linalg.norm(velocity, axis=1)

    # Remove old contours
    for coll in ax1.collections[n_levels:]:
        coll.remove()
    for coll in ax2.collections[n_levels:]:
        coll.remove()

    # Create new contours
    v_contour = ax1.tricontourf(tri, vel_mag, levels=n_levels, cmap='plasma')
    p_contour = ax2.tricontourf(tri, pressure, levels=n_levels, cmap='viridis')


    ax1.set_title(f"Velocity Magnitude – Frame {frame}")
    ax2.set_title(f"Pressure Field – Frame {frame}")

    ax1.triplot(tri, color='k', linewidth=0.1)
    ax2.triplot(tri, color='k', linewidth=0.1)

    return []

# Create the animation

anim = FuncAnimation(fig, update, frames=n_timesteps, interval=10)

plt.tight_layout()
#plt.show()

anim.save("airfoil_animation.mp4", fps=5, dpi=400)