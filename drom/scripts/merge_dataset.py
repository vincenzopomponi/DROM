import os
import sys
import shutil
import h5py
import argparse
import numpy as np
import drom.demos


def copy_attrs(src, dst):
    """Copy attributes from one HDF5 object to another."""
    for key, val in src.attrs.items():
        dst.attrs[key] = val


def copy_non_demo_data(src_file, dst_file):
    """Copy all groups and datasets except 'data'."""
    for key in src_file.keys():
        if key != "data" and key not in dst_file:
            src_file.copy(key, dst_file)
            copy_attrs(src_file[key], dst_file[key])


def sorted_demo_keys(data_group):
    """Sort demo keys numerically: demo_1, demo_2, ..."""
    keys = sorted(
        data_group.keys(),
        key=lambda x: int(x.split("_")[1])
    )
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing multiple dataset folders"
    )
    args = parser.parse_args()

    dir_path = os.path.join(drom.demos.demo_root, args.directory)

    dataset_dirs = sorted([
        os.path.join(dir_path, d)
        for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d))
    ])

    if not dataset_dirs:
        raise RuntimeError(f"No subfolders found in {dir_path}")
    
    dmp_dirs = [p for p in dataset_dirs if p.endswith(os.sep + "dmp") or p.endswith("dmp")]

    if dmp_dirs:
        print("\nMerged dataset already present:")
        for p in dmp_dirs:
            print(f"  - {p}")

        ans = input("Delete it? [y: yes, n: no] ").strip().lower()

        if ans in ["y", "yes"]:
            for p in dmp_dirs:
                print(f"🗑️  Deleting: {p}")
                shutil.rmtree(p)
            dataset_dirs = [p for p in dataset_dirs if p not in dmp_dirs]
        else:
            print("❌ Aborted by user.")
            sys.exit(0)

    print(f"\nI found the following datasets:\n{dataset_dirs}")
    ans = input("Continue? [y: yes, n: no] ").strip().lower()
    if ans not in ["y", "yes"]:
        print("Aborted.")
        exit(0)

    out_dir = os.path.join(dir_path, "dmp")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "image_200.hdf5")

    demo_counter = 0
    copied_metadata = False

    hdf5_name = input("Name of the hdf5 file: ")

    with h5py.File(output_path, "w") as f_out:
        data_out = f_out.create_group("data")

        for dataset_dir in dataset_dirs:
            demo_path = os.path.join(dataset_dir, f"{hdf5_name}.hdf5")

            if not os.path.exists(demo_path):
                print(f"⚠ Skipping (no {hdf5_name}.hdf5): {dataset_dir}")
                continue

            print(f"📂 Merging: {demo_path}")

            with h5py.File(demo_path, "r") as f_in:
                try:
                    _ = list(f_in["data"].keys())
                except Exception as e:
                    print(f"❌ Corrupted HDF5 file: {demo_path}")
                    print(e)
                    continue

                if "data" not in f_in:
                    print(f"⚠ No 'data' group in {demo_path}, skipping.")
                    continue

                if not copied_metadata:
                    copy_non_demo_data(f_in, f_out)
                    copy_attrs(f_in["data"], data_out)
                    copied_metadata = True

                for demo_key in sorted_demo_keys(f_in["data"]):

                    new_demo_key = f"demo_{demo_counter}"

                    # Copy demo
                    f_in.copy(f"data/{demo_key}", data_out, name=new_demo_key)
                    demo_out = data_out[new_demo_key]

                    copy_attrs(f_in["data"][demo_key], demo_out)

                    # -------------------------------------------------
                    # ✅ ADD states dataset: (len(actions_abs), 34)
                    # -------------------------------------------------
                    if "actions_abs" not in demo_out:
                        print(f"'actions_abs' dataset missing in {new_demo_key}")
                        continue

                    T = demo_out["actions_abs"].shape[0]

                    if "states" not in demo_out:
                        states = np.zeros((T, 34), dtype=np.float32)
                        demo_out.create_dataset(
                            "states",
                            data=states,
                            compression="gzip"
                        )

                    demo_counter += 1

    print(f"\n✅ Merged {demo_counter} demos")
    print(f"💾 Saved merged dataset to: {output_path}")
