import h5py  # pyright: ignore
import shutil
from pathlib import Path
from typing import Union


def rename_h5_dataset(
    h5_file_path: Union[str, Path], old_key: str, new_key: str
) -> None:
    """
    Rename a dataset in an HDF5 file by copying it to a new key and deleting the old one.

    Parameters
    ----------
    h5_file_path : str or Path
        Path to the HDF5 file
    old_key : str
        Key/name of the existing dataset to rename
    new_key : str
        New key/name for the dataset

    Raises
    ------
    FileNotFoundError
        If the HDF5 file doesn't exist
    KeyError
        If the old_key doesn't exist in the file
    ValueError
        If the new_key already exists in the file
    """
    h5_file_path = Path(h5_file_path)

    # Check if file exists
    if not h5_file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

    with h5py.File(h5_file_path, "r+") as f:
        # Check if old key exists
        if old_key not in f:
            raise KeyError(f"Dataset '{old_key}' not found in {h5_file_path}")

        # Check if new key already exists
        if new_key in f:
            raise ValueError(f"Dataset '{new_key}' already exists in {h5_file_path}")

        # Get the original dataset
        old_dataset = f[old_key]

        # Copy the dataset with all its attributes
        # This preserves data type, shape, compression, etc.
        f.copy(old_dataset, new_key)

        # Alternatively, if you want more control over the copying process:
        # Create new dataset with same properties
        # new_dataset = f.create_dataset(
        #     new_key,
        #     data=old_dataset[:],
        #     dtype=old_dataset.dtype,
        #     compression=old_dataset.compression,
        #     compression_opts=old_dataset.compression_opts,
        #     shuffle=old_dataset.shuffle,
        #     fletcher32=old_dataset.fletcher32,
        #     chunks=old_dataset.chunks
        # )
        #
        # # Copy attributes
        # for attr_name, attr_value in old_dataset.attrs.items():
        #     new_dataset.attrs[attr_name] = attr_value

        # Delete the old dataset
        del f[old_key]

        print(
            f"Successfully renamed dataset '{old_key}' to '{new_key}' in {h5_file_path}"
        )


def rename_h5_dataset_safe(
    h5_file_path: Union[str, Path], old_key: str, new_key: str, backup: bool = True
) -> None:
    """
    Safely rename a dataset in an HDF5 file with optional backup.

    This version creates a backup of the original file before making changes.

    Parameters
    ----------
    h5_file_path : str or Path
        Path to the HDF5 file
    old_key : str
        Key/name of the existing dataset to rename
    new_key : str
        New key/name for the dataset
    backup : bool, default=True
        Whether to create a backup file before making changes

    Raises
    ------
    FileNotFoundError
        If the HDF5 file doesn't exist
    KeyError
        If the old_key doesn't exist in the file
    ValueError
        If the new_key already exists in the file
    """
    h5_file_path = Path(h5_file_path)

    # Check if file exists
    if not h5_file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

    # Create backup if requested
    backup_path = None
    if backup:
        backup_path = h5_file_path.with_suffix(h5_file_path.suffix + ".backup")
        _ = shutil.copy2(h5_file_path, backup_path)
        print(f"Created backup: {backup_path}")

    try:
        rename_h5_dataset(h5_file_path, old_key, new_key)

        # Remove backup if operation was successful and backup was created
        if backup and backup_path is not None and backup_path.exists():
            backup_path.unlink()
            print(f"Removed backup file: {backup_path}")

    except Exception as e:
        if backup and backup_path is not None and backup_path.exists():
            print(f"Error occurred. Backup file preserved at: {backup_path}")
        raise e


if __name__ == "__main__":
    # Example usage and command-line interface
    import argparse

    # Command-line interface
    parser = argparse.ArgumentParser(description="Rename a dataset in an HDF5 file")
    _ = parser.add_argument("h5_file", nargs="?", help="Path to the HDF5 file")
    _ = parser.add_argument(
        "old_key", nargs="?", help="Current name/key of the dataset"
    )
    _ = parser.add_argument("new_key", nargs="?", help="New name/key for the dataset")
    _ = parser.add_argument(
        "--safe", action="store_true", help="Use safe mode with backup"
    )

    args = parser.parse_args()

    if args.h5_file and args.old_key and args.new_key:
        try:
            if args.safe:
                rename_h5_dataset_safe(args.h5_file, args.old_key, args.new_key)
            else:
                rename_h5_dataset(args.h5_file, args.old_key, args.new_key)
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python rename_h5_dataset.py file.h5 old_name new_name")
        print("  python rename_h5_dataset.py file.h5 old_name new_name --safe")
