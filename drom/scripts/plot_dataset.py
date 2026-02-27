import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import drom.demos

def plot_episode_actions(hdf5_file):
    """
    Plots 7-DoF actions for each episode in the given HDF5 file using subplots.
    
    Parameters:
        hdf5_file (h5py.File): The opened HDF5 file containing demonstration data.
    """
    demos = list(hdf5_file["data"].keys())
    fig, axs = plt.subplots(7, 1, figsize=(12, 10), sharex=True)

    for i in tqdm(range(len(demos)), desc="Validating model...", unit="test"):
        ep = "demo_" + str(i)
        actions = np.array(hdf5_file[f"data/{ep}/actions_abs"][()])
        num_steps = actions.shape[0]
        
        fig.suptitle(f"Episode {ep} - Action Trajectories", fontsize=16)

        dof_labels = [
            "Position X", "Position Y", "Position Z",
            "Rotation X", "Rotation Y", "Rotation Z",
            "Gripper"
        ]

        for i in range(7):
            axs[i].plot(range(num_steps), actions[:, i])
            axs[i].set_ylabel(dof_labels[i])
            axs[i].grid(True)

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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
    

    f = h5py.File(hdf5_path, "r")

    plot_episode_actions(f)
