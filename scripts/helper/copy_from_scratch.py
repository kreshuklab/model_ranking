from tqdm import tqdm
from pathlib import Path
import shutil

scratch_base = Path("/scratch/talks")
g_base = Path("/g/kreshuk/talks/")
dir_path = Path("consistency_results/patch_segmentation/mitochondria")
combined_path = scratch_base / dir_path

paths = list(combined_path.rglob("**/P_full"))
for path in tqdm(paths):
    # print(path)
    g_path = Path(str(path).replace(str(scratch_base), str(g_base)))
    # print(g_path)
    _ = shutil.copytree(path, g_path, dirs_exist_ok=True)
