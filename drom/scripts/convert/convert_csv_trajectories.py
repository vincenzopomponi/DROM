import os
import argparse
import pandas as pd
import h5py
import datetime
import torch
from tqdm import tqdm
from pathlib import Path
import drom.demos


def from_csv_to_hdf5(N, directory):
    print("\nConverting CSV trajectories to HDF5 format...")

    # Build directories safely
    csv_dir = Path(drom.demos.demos_root) / directory
    hdf5_dir = Path(drom.demos.demos_root) / "hdf5"

    # --- Error checking ---
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    tasks = [t for t in os.listdir(csv_dir) if (csv_dir / t).is_dir()]
    if not tasks:
        raise RuntimeError(f"No task folders found inside {csv_dir}")

    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each task with a progress bar ---
    for t in tqdm(tasks, desc="Processing tasks", unit="task"):
        task_dir = csv_dir / t
        
        hdf5_task_dir = hdf5_dir / f"{t}/dmp"
        hdf5_task_dir.mkdir(parents=True, exist_ok=True)

        hdf5_path = hdf5_task_dir / "demo.hdf5"

        # Open HDF5 file
        with h5py.File(hdf5_path, "w") as h5f:
            grp = h5f.create_group("data")

            for i in tqdm(range(N), desc=f"Task: {t}", leave=False, unit="traj"):
                path_traj = task_dir / f"traj_{i}.csv"

                # Check file existence
                if not path_traj.exists():
                    raise FileNotFoundError(f"Missing trajectory file: {path_traj}")

                # Load CSV → numpy
                traj = pd.read_csv(path_traj).to_numpy()

                # --- Create group: data/demo_i ---
                demo_group = grp.create_group("demo_{}".format(i))

                # --- Create dataset: data/demo_i/actions ---
                demo_group.create_dataset("actions", data=traj)
        
            # write dataset attributes (metadata)
            now = datetime.datetime.now()
            grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
            grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
            # grp.attrs["repository_version"] = dmg.__version__
            # grp.attrs["repository_version"] = robosuite.__version__
            # grp.attrs["env"] = env_name
            # grp.attrs["env_info"] = env_info
        h5f.close()


def from_csv_to_tensor(N, directory, device):
    print("\nConverting CSV trajectories to PyTorch tensors...")

    # Build directories safely
    csv_dir = Path(drom.demos.demos_root) / directory
    tensor_dir = Path(drom.demos.demos_root) / "tensors"

    # --- Error checking ---
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    tasks = [t for t in os.listdir(csv_dir) if (csv_dir / t).is_dir()]
    if not tasks:
        raise RuntimeError(f"No task folders found inside {csv_dir}")

    tensor_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each task with a progress bar ---
    for t in tqdm(tasks, desc="Processing tasks", unit="task"):
        task_dir = csv_dir / t
        trajectories = []

        for i in tqdm(range(N), desc=f"Task: {t}", leave=False, unit="traj"):
            path_traj = task_dir / f"traj_{i}.csv"

            # Check file existence
            if not path_traj.exists():
                raise FileNotFoundError(f"Missing trajectory file: {path_traj}")

            traj = pd.read_csv(path_traj).to_numpy()
            traj = torch.tensor(traj, device=device).unsqueeze(0)
            trajectories.append(traj)

        trajectories = torch.cat(trajectories)
        torch.save(trajectories, tensor_dir / f"{t}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV trajectories to tensor or HDF5 files")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing CSV demos")
    parser.add_argument("--N", type=int, default=2000, help="Number of trajectories per task")
    parser.add_argument("--mode", type=str, choices=["tensor", "hdf5"], required=True, help="Output format")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device for tensor ops (only for tensor mode)")

    args = parser.parse_args()

    if args.mode == "tensor":
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        if args.device == "cuda" and device.type == "cpu":
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
        from_csv_to_tensor(args.N, args.directory, device)
    else:
        from_csv_to_hdf5(args.N, args.directory)
