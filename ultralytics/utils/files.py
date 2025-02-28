import glob
import os
from pathlib import Path


def get_latest_run(search_dir="."):
    """返回在runs文件夹的最近的last.pt，"""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """创建数字后缀+1不存在的保存路径
    Args:
        exist_ok(bool): 如果文件已存在， 不新建文件， 默认false"""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)   #创建目录
    return path

def file_size(path):
    if isinstance(path, (str, Path)):
        mb = 1 << 20   #MB
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0