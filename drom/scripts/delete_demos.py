import os
import h5py
import argparse
import shutil
import tempfile

def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def sorted_demo_keys(data_group):
    return sorted(
        data_group.keys(),
        key=lambda x: int(x.split("_")[1])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete a single demo from an HDF5 file."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to demo.hdf5"
    )
    parser.add_argument(
        "--demo",
        type=str,
        required=True,
        help="Demo to delete (e.g. demo_17)"
    )
    parser.add_argument(
        "--renumber",
        action="store_false",
        help="Renumber demos after deletion (demo_1, demo_2, ...)"
    )
    args = parser.parse_args()

    hdf5_path = args.file
    demo_to_delete = args.demo

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(hdf5_path)

    # Temporary file
    tmp_path = hdf5_path + ".tmp"

    with h5py.File(hdf5_path, "r") as f_in, h5py.File(tmp_path, "w") as f_out:

        # ------------------------------------------------
        # Copy everything except 'data'
        # ------------------------------------------------
        for key in f_in.keys():
            if key != "data":
                f_in.copy(key, f_out)
                copy_attrs(f_in[key], f_out[key])

        if "data" not in f_in:
            raise RuntimeError("No 'data' group found.")

        data_out = f_out.create_group("data")
        copy_attrs(f_in["data"], data_out)

        # ------------------------------------------------
        # Copy demos except the one to delete
        # ------------------------------------------------
        new_idx = 1
        for demo_key in sorted_demo_keys(f_in["data"]):

            if demo_key == demo_to_delete:
                print(f"🗑️  Deleting {demo_key}")
                continue

            new_demo_key = (
                f"demo_{new_idx}" if args.renumber else demo_key
            )

            f_in.copy(
                f"data/{demo_key}",
                data_out,
                name=new_demo_key
            )

            copy_attrs(
                f_in["data"][demo_key],
                data_out[new_demo_key]
            )

            new_idx += 1

    # ------------------------------------------------
    # Replace original file atomically
    # ------------------------------------------------
    shutil.move(tmp_path, hdf5_path)

    print(f"\n✅ Successfully deleted {demo_to_delete}")
    print(f"📁 Updated file: {hdf5_path}")
