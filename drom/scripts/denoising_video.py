import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
import argparse

import drom.demos

def get_actions_from_hdf5(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    ep = "demo_0"
    actions = np.array(f[f"data/{ep}/actions_abs"][()])
    t = np.linspace(0, 1, actions.shape[0])
    return np.stack([t, actions[:, 0]], axis=1)

def add_noise(traj, noise_level):
    """Add Gaussian noise to a trajectory, keeping start and goal fixed."""
    noisy_traj = traj.copy()
    noisy_traj[1:-1] += noise_level * np.random.randn(*traj[1:-1].shape)
    return noisy_traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to your demonstration directory containing the demo.hdf5 file.",
    )
    args = parser.parse_args()

    hdf5_path = os.path.join(drom.demos.demo_root, args.directory, "dmp/image_200.hdf5")
    traj_gt = get_actions_from_hdf5(hdf5_path=hdf5_path)

    # Diffusion noise levels from high → low
    # noise_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.15, 0.1, 0.05, 0.0]
    noise_levels = np.arange(20, 0.0, step=-0.5)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect('equal')
    ax.set_title("Diffusion Denoising Process")
    ax.set_xlabel("Time")
    ax.set_ylabel("Trajectory Value")

    # Initialize plot elements
    line, = ax.plot([], [], color='blue', lw=2)
    start_point = ax.scatter([], [], color='green', s=50, marker='o', label='Start')
    goal_point = ax.scatter([], [], color='red', s=50, marker='o', label='Goal')
    ax.legend(loc='upper right')

    # Set axis limits
    margin = 0.1
    ax.set_xlim(traj_gt[:,0].min() - margin, traj_gt[:,0].max() + margin)
    ax.set_ylim(traj_gt[:,1].min() - margin, traj_gt[:,1].max() + margin)

    # Animation function
    def update(frame):
        traj_noisy = add_noise(traj_gt, noise_levels[frame])
        line.set_data(traj_noisy[:,0], traj_noisy[:,1])
        start_point.set_offsets([traj_gt[0,0], traj_gt[0,1]])
        goal_point.set_offsets([traj_gt[-1,0], traj_gt[-1,1]])
        ax.set_title(f"Diffusion Step {frame+1} / {len(noise_levels)}")
        return line, start_point, goal_point

    anim = FuncAnimation(fig, update, frames=len(noise_levels), interval=100, blit=True)

    plt.show()
