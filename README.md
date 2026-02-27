# DROM: Multi-task Robotic Manipulation via Diffusion Models

# Abstract:
Day by day, robots are asked to solve various tasks in diverse domains.
The advent of recent Generative AI techniques is tackling this request with success.
Despite this, it demands a tremendous amount of data and training resources.
In this work, we propose DROM (Diffusion for RObotic Manipulation) to solve different tasks from the domestic and industrial domains, reducing the need for enormous demonstrative datasets.
We leverage Dynamic Movement Primitives' capabilities of generalizing the shape of a trajectory to create a dataset from a single expert demonstration for each task and then train a single Diffusion Model to solve all the tasks.
Results show that the proposed method is capable of generating trajectories for solving all the tasks with a very high success rate (76\% for joint space trajectories and 92\% for Cartesian space trajectories), generalizing on a vast workspace (0.6 m and 1.1 m on the X and Y axes respectively), using a single demonstration for each of the considered tasks.

# Installation

```bash
source install.sh
```

# Train DROM
```bash
python drom/scripts/train_drom.py --config drom/configs/low_dim/drom.json
```

# Test DROM
```bash
python drom/scripts/test_drom.py --directory \<directory\> --validations 30 --seed 30 --render
```

# Print HDF5 file contents
```bash
python drom/scripts/print_hdf5_dataset.py --directory <dir-path>
```

# Merge datasets
This script merges all datasets in the specified folder into a single dataset. It creates a new folder named ‘dmp’ and saves the merged dataset file inside it.
```bash
python drom/scripts/merge_dataset.py --directory <dir-name>
```

# Delete a demo(s) within a HDF5 file
```bash
python drom/scripts/delete_demos.py --file <path-file> --demo demo_32 --renumber
```

# Plot the dataset
```bash
python drom/scripts/plot_dataset.py --directory <dir-name>
```