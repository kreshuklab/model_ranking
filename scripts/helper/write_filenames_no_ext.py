import os
import sys
from typing import Optional


def write_filenames_no_ext(dir_path: str, output_txt: Optional[str] = None):
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        return
    filenames = [
        os.path.splitext(f)[0]
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
    filenames.sort()
    if output_txt is None:
        output_txt = os.path.join(dir_path, "filenames.txt")
    with open(output_txt, "w") as f:
        for name in filenames:
            _ = f.write(name + "\n")
    print(f"Wrote {len(filenames)} filenames to {output_txt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python write_filenames_no_ext.py <directory_path> [output_txt]")
        sys.exit(1)
    dir_path = sys.argv[1]
    output_txt = sys.argv[2] if len(sys.argv) > 2 else None
    write_filenames_no_ext(dir_path, output_txt)
