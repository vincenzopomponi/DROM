import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

import drom.demos

def get_actions_from_hdf5(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    ep = "demo_0"
    actions = np.array(f[f"data/{ep}/actions_abs"][()])
    t = t = np.linspace(0, 1, actions.shape[0])
    return np.stack([t, actions[:, 0]], axis=1)

def generate_clean_trajectory(T=100):
    """Ground-truth smooth trajectory (e.g., robot end-effector path)."""
    t = np.linspace(0, 1, T)
    x = t
    # y = np.sin(2 * np.pi * t)
    y = np.sin((2 / 3) * np.pi * t)
    return np.stack([x, y], axis=1)

def add_noise(traj, noise_level):
    """Add Gaussian noise to a trajectory, keeping start and goal fixed."""
    noisy_traj = traj.copy()
    # Apply noise to intermediate points only
    noisy_traj[1:-1] += noise_level * np.random.randn(*traj[1:-1].shape)
    return noisy_traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to your demonstration directory that contains the demo.hdf5 file, e.g.: "
        "'path_to_demos_dir/hdf5/YOUR_DEMONSTRATION'",
    ),
    args = parser.parse_args()

    hdf5_path = os.path.join(drom.demos.demo_root, args.directory)
    hdf5_path = os.path.join(hdf5_path, "dmp/image_200.hdf5")

    # Ground truth trajectory
    traj_gt = generate_clean_trajectory()
    # traj_gt = get_actions_from_hdf5(hdf5_path=hdf5_path)

    # Diffusion steps (noise levels from high → low)
    # noise_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.15, 0.1, 0.05, 0.0]
    noise_levels = [0.4, 0.2, 0.15, 0.1, 0.05, 0.00]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))  # 2 rows × 6 cols
    axes = axes.flatten()  # flatten to 1D array for easy indexing

    for i, nl in enumerate(noise_levels):
        traj_noisy = add_noise(traj_gt, nl)

        # Plot: last step as line, others as scatter
        # if i == len(noise_levels) - 1:
        #     axes[i].plot(traj_noisy[:, 0], traj_noisy[:, 1], color='blue')
        # else:
        axes[i].scatter(traj_noisy[:, 0], traj_noisy[:, 1], color='blue', s=10)

        # Plot start (green) and goal (red)
        axes[i].scatter(traj_gt[0, 0], traj_gt[0, 1], color='green', s=30, marker='o')  # start
        axes[i].scatter(traj_gt[-1, 0], traj_gt[-1, 1], color='red', s=30, marker='o')   # goal

        axes[i].set_title(f"Step {i+1}")
        axes[i].set_aspect("equal")
        axes[i].axis("off")

    fig.suptitle(f"Diffusion Denoising Process", fontsize=14)
    plt.tight_layout()
    plt.show()
