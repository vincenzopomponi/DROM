import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import drom.demos

def plot_drom_trajectories(
    trajectories, hard_conds,
    axs,
    demo=None, cartesian=True):

    # --- Titles based on mode ---
    n_channels = trajectories.shape[-1]
    if cartesian:
        if n_channels == 7:
            titles = ["PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ", "Gripper"]
        elif n_channels == 8:
            titles = ["PosX", "PosY", "PosZ", "QuatX", "QuatY", "QuatZ", "QuatW", "Gripper"]
        else:
            raise ValueError(f"Unexpected number of channels: {n_channels}")
    else:
        titles = [f"J{i+1}" for i in range(n_channels)]

    # --- Convert tensors to numpy ---
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()
    if isinstance(demo, torch.Tensor):
        demo = demo.detach().cpu().numpy()
    start_cond = hard_conds[0]
    goal_cond = hard_conds[1]
    if isinstance(start_cond,torch.Tensor):
        start_cond = start_cond.detach().cpu().numpy()
    if isinstance(goal_cond,torch.Tensor):
        goal_cond = goal_cond.detach().cpu().numpy()

    # --- Plot trajectories ---
    if trajectories.ndim == 2:
        trajectories = np.expand_dims(trajectories, axis=0)
    
    for traj_index in range(len(trajectories)):
        for ch in range(n_channels):
            row, col = divmod(ch, 2)
            axs[row, col].plot(trajectories[traj_index, :, ch])

    # --- Hard conditions ---
    for i in range(n_channels):
        row, col = divmod(i, 2)
        ax = axs[row, col]
        ax.scatter(0, start_cond[i], c='k')
        ax.scatter(trajectories.shape[1]-1, goal_cond[i], c='k')
        ax.set_title(titles[i])

    # --- Demo overlay ---
    if demo is not None:
        for ch in range(min(demo.shape[1], n_channels)):
            row, col = divmod(ch, 2)
            axs[row, col].plot(demo[:, ch], "r--")

    # # --- Axis limits ---
    # pos_lim = (-1.0, 1.0)
    # rot_lim = (-1.0, 1.0)
    # grip_lim = (-1.1, 0.0)

    # if cartesian:
    #     # Positions
    #     for ch in range(3):
    #         row, col = divmod(ch, 2)
    #         axs[row, col].set_ylim(pos_lim)
    #     # Rotations or quaternion
    #     for ch in range(3, 7):
    #         row, col = divmod(ch, 2)
    #         axs[row, col].set_ylim(rot_lim)
    #     # Gripper
    #     # row, col = divmod(7 if n_channels == 8 else 6, 2)
    #     # axs[row, col].set_ylim(grip_lim)
    # else:
    #     # Joint limits (optional, could be left automatic)
    #     for ch in range(n_channels):
    #         row, col = divmod(ch, 2)
    #         axs[row, col].set_ylim([-np.pi, np.pi])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the full content of an HDF5 demonstration file.")
    parser.add_argument("--directory", type=str,
                        help="Path to your demonstration directory that contains the demo.hdf5 file, e.g.: 'path_to_demos_dir/hdf5/YOUR_DEMONSTRATION'")
    parser.add_argument("--dmp", action="store_true")
    parser.add_argument("--mimicgen", action="store_true")
    args = parser.parse_args()

    # Build full directory path
    if args.mimicgen:
        dir_path = os.path.join(drom.demos.demo_root, "mimicgen", args.directory)
    else:
        dir_path = os.path.join(drom.demos.demo_root, args.directory)

    if args.dmp:
        dir_path = os.path.join(dir_path, "dmp")
    hdf5_files = [f for f in os.listdir(dir_path) if f.endswith(".hdf5")]

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in: {dir_path}")

    print("\nAvailable .hdf5 files:")
    for i, file in enumerate(hdf5_files, start=1):
        print(f"{i}: {file}")

    while True:
        ans = input(f"Select file [1-{len(hdf5_files)}] (default=1): ").strip()
        if ans == "":
            selection = 1
            break
        elif ans.isdigit() and 1 <= int(ans) <= len(hdf5_files):
            selection = int(ans)
            break
        else:
            print("Invalid input. Please enter a valid number or press Enter for default.")

    filename = hdf5_files[selection - 1]
    path = os.path.join(dir_path, filename)

    print(f"\n✅ Selected file: {filename}")
    print(f"📂 Full path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found at: {path}")

    print(f"\n🔍 Inspecting HDF5 file: {path}\n")

    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        
        # --- Create figure ---
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        fig.suptitle(f"Task: {args.directory}")
        
        for demo in tqdm(demos, desc="Plotting trajectories...", unit="demo"):
            states = f["data/{}/states".format(demo)][()]
            actions = f["data/{}/actions_abs".format(demo)][()]

            plot_drom_trajectories(trajectories=actions, hard_conds=[actions[0], actions[-1]], axs=axs)
        
        plt.show()