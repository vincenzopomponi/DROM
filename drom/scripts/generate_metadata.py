import os
import json
import h5py
import argparse

import drom.demos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo-length metadata for an HDF5 demonstration file.")
    parser.add_argument("--directory", type=str,
                        help="Path to your demonstration directory containing the .hdf5 file.")
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

    # Choose file
    while True:
        ans = input(f"Select file [1-{len(hdf5_files)}] (default=1): ").strip()
        if ans == "":
            selection = 1
            break
        elif ans.isdigit() and 1 <= int(ans) <= len(hdf5_files):
            selection = int(ans)
            break
        else:
            print("Invalid input. Try again.")

    filename = hdf5_files[selection - 1]
    path = os.path.join(dir_path, filename)

    print(f"\n🔍 Selected file: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    print("\nExtracting demonstration lengths...\n")

    # Dictionaries to store metadata
    demo_to_length = {}
    length_to_demos = {}
    demo_to_subtask = {}

    # --------------------------
    #   READ ORIGINAL HDF5 FILE
    # --------------------------
    with h5py.File(path, "r") as f:

        if "data" not in f:
            raise RuntimeError("No 'data' group found in this file.")

        for demo_key in f["data"]:
            demo = f["data"][demo_key]

            if "actions" not in demo:
                print(f"⚠ Demo {demo_key} has no 'actions' dataset. Skipping.")
                continue

            # Length extraction
            length = demo["actions"].shape[0]
            demo_to_length[demo_key] = length

            if length not in length_to_demos:
                length_to_demos[length] = []
            length_to_demos[length].append(demo_key)

            # Subtask extraction
            if "subtask_data" in demo.attrs:
                raw = demo.attrs["subtask_data"]

                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                try:
                    sub_dict = json.loads(raw)
                except Exception:
                    print(f"⚠ Could not decode subtask_data for {demo_key}. Saving as raw text.")
                    sub_dict = {"raw": str(raw)}
            else:
                sub_dict = {}

            demo_to_subtask[demo_key] = sub_dict

    # Print summary
    print("=== Demonstration Lengths ===")
    for k, v in demo_to_length.items():
        print(f" {k}: {v}")

    print("\n=== Unique Lengths ===")
    for L, demos in length_to_demos.items():
        print(f" Length {L}: {demos}")

    print("\n=== Subtask data ===")
    for k, v in demo_to_subtask.items():
        print(f" {k}: {v}")

    # ------------------------------
    #   SAVE METADATA TO NEW HDF5
    # ------------------------------

    metadata_path = os.path.join(dir_path, "metadata.hdf5")
    print(f"\n💾 Saving metadata to: {metadata_path}")

    with h5py.File(metadata_path, "w") as mf:
        demo_keys = list(demo_to_length.keys())
        for demo in demo_keys:
            demo_group = mf.create_group("{}".format(demo))

            # Save demo keys as UTF-8 strings
            dt = h5py.string_dtype(encoding='utf-8')

            # Save demo subtask data as JSON-encoded strings
            demo_subtasks_json = json.dumps(demo_to_subtask[demo])
            demo_group.create_dataset("subtask_data", data=demo_subtasks_json, dtype=dt)

    print("\n✅ Metadata saved successfully!")
