from pathlib import Path
import glob
import shutil


yolo_path  = Path("G:\\TK\\PDS\\ww\\pds_RBB\\data")
labels_dir = yolo_path / "labels//train"
images_dir = yolo_path / "images//train"

labels_path = glob.glob(str(labels_dir) + "//**", recursive=False)
images_path = glob.glob(str(images_dir) + "//**", recursive=False)

ok_path = yolo_path / "ok_ng//ok_ng_ok"
ng_path = yolo_path / "ok_ng//ok_ng_ng"
ok_path.mkdir(parents=True,exist_ok=True)
ng_path.mkdir(parents=True, exist_ok=True)
for label_path in labels_path:
    with open(label_path, "r") as f:
        label = f.read().strip().splitlines()
    img_path = label_path.replace("labels","images").split(".")[0] + ".bmp"
    if len(label):
        shutil.copy(img_path, ng_path)
    else:
        shutil.copy(img_path, ok_path)
    print(str(img_path))
