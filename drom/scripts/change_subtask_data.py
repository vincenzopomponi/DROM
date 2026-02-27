import os
import h5py
import json
from tqdm import tqdm
import argparse
import drom.demos


def sorted_demo_keys(data_group):
    return sorted(
        data_group.keys(),
        key=lambda x: int(x.split("_")[1])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing dataset folders"
    )
    args = parser.parse_args()

    dir_path = os.path.join(drom.demos.demo_root, args.directory)

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
            print("Invalid input.")

    filename = hdf5_files[selection - 1]
    input_path = os.path.join(dir_path, filename)

    demo_counter = 0

    with h5py.File(input_path, "r+") as f_in:

        data_group = f_in["data"]

        for demo_key in tqdm(
            sorted_demo_keys(data_group),
            desc="Processing demos",
            unit="demo"
        ):

            demo = data_group[demo_key]

            if "subtask_data" not in demo.attrs:
                print(f"⚠ subtask_data missing in {demo_key}")
                continue

            demo.attrs["num_samples"] = 338

            # Load JSON (handle possible bytes)
            raw_attr = demo.attrs["subtask_data"]
            if isinstance(raw_attr, bytes):
                raw_attr = raw_attr.decode("utf-8")

            subtask_data = json.loads(raw_attr)

            # Modify only ts_split
            subtask_data["ts_split"] = [0, 64, 128, 207, 274, 338]

            # Write back
            demo.attrs["subtask_data"] = json.dumps(subtask_data)

            demo_counter += 1

    print(f"\n✅ Modified {demo_counter} demos in-place.")
