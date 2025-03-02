import shutil
import subprocess
import requests
from urllib import parse, request
import torch.hub

from ultralytics.utils import LOGGER,url2file,clean_url,is_online,emojis
from ultralytics.utils import checks
import re
from pathlib import Path
from multiprocessing.pool import ThreadPool
from itertools import repeat
import contextlib

GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = (
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + ["mobile_sam.pt"]
)
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]


def get_google_drive_file_info(link):
    """
    检索可共享的 Google 云端硬盘文件链接的直接下载链接和文件名。
    :param link(str): google文件的共享链接
    :return:
        (str):google文件的直接下载地址
        (str):google文件的原文件名
    Example:
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
    """
    file_id = link.split("/d/")[1].split("/view")[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None
    with requests.Session() as session:
        respose = session.get(drive_url, stream=True)
        if "quota exceeded" in str(respose.content.lower()):
            LOGGER.error(f"❌ goole 文件下载超出配额，请稍后重试或收到到{link}进行下载")

        for k, v in respose.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confim={v}"   #v is token
        cd = respose.headers.get("content-disposition")
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename

def check_disk_space(url="https://ultralytics.com/assets/coco128.zip", sf=1.5, hard=True):
    """
    检查是否有充足的磁盘空间下载和存储文件
    :param url（str, optional）: 下载文件地址
    :param sf(float, optional): 保证安全的尺度，需求空白空间的为下载文件的倍数，默认2.0
    :param hard(bool, optional): 没有充足空间下载文件时是否报错，默认True
    :return: 
        (bool) True/False
    """
    try:
        r = requests.head(url) #response  请求网址响应
        assert r.status_code < 400, f"URL error: {url}: {r.status_code} {r.reason}"   #检查响应，404为未找到 200为正常
    except Exception:
        return True  #请求出问题，返回Ture

    #文件大小
    gib = 1<<30   #1GB 1073741824byte
    data = int (r.headers.get("Content-Length",0)) / gib  #文件大小GB
    total, used, free = (x/gib for x in shutil.disk_usage(Path.cwd()))    #获取当前磁盘路径的总存储空间，已使用存储空间和未使用存储空间 （GB）

    if data *sf < free:
        return True  #空间充足


    #空间不足
    text = {
        f"WARNING ⚠️ 磁盘剩余存储空间不足{free:.1f}GB < {data*sf:.3f}GB， 请确保磁盘剩余存储空间大于{free:.1f}再重试"
    }
    if hard:
        raise  MemoryError(text)
    LOGGER.warning(text)
    return False

def unzip_file(file, path=None, exclude=(".DS_Store","__MACOSX"), exist_ok=False, progress=True):
    """
    解压缩*.zip文件至指定文件夹中
    :param file: 压缩文件路径
    :param path: 保存路径， 默认None
    :param exclude: 要排序的文件名字符串元组，默认（".DS_Store","__MACOSX"）
    :param exist_ok: 是否覆盖已有文件，默认False
    :param progress: 是否显示进度条，默认True
    :return: 
        （Path）: 已解压缩文件的路径
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"'{file}'不存在或者不是一个压缩文件")
    if path is None:
        path = Path(file).parent  #默认路径为父级目录

    #解压缩
    #路径
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}   #最上级路径
        if len(top_level_dirs) > 1 or (len(files) > 1 and not files[0].endswith(("/"))):
            #压缩文件在最上级有多个文件
            path = extract_path = Path(path)/Path(file).stem  # i.e.  ../datasets/coco8    #stem文件名不带后最
        else:
            #压缩文件只有一个最上级目录
            extract_path = path   #i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  #i.e.  ../datasets/coco8

        #检查目标文件夹是否已经存在且包含文件
        if path.exists() and any(path.iterdir()) and not exist_ok:
            LOGGER.warning(f"WARNING ⚠️ 目标文件夹{path}不为空，跳过解压缩{file}")
            return path

        for f in files:
            if ".." in Path(f).parts:
                LOGGER.warning(f"WARNING ⚠️ 不安全的路径：{f}...跳过")
                continue
            zipObj.extract(f, extract_path)   #解压缩
    return path




def safe_download(
        url,
        file=None,
        dir=None,
        unzip=True,
        delete=False,
        curl=False,
        retry=3,
        min_byte=1e0,
        exist_ok=False,
        progress=True
):
    """
    从一个URL中下载文件，可选参数有重新下载次数，解压缩，删除文件
    :param url（str）: 将要下载文件的网址
    :param file（str, optional）: 下载文件保存的文件名，如果未提供，文件将保存为跟URL一样的文件名
    :param dir（str, optinal）: 下载文件的保存路径，如果未提供，文件将保存在当前路径
    :param unzip（bool, optional）: 是否对下载文件进行解压， 默认True
    :param delete（bool, optional）: 是否在解压之后删除原文件，默认False
    :param curl（bool， optional）: 是否使用curl命令行工具下载， 默认False
    :param retry（int, optioan）: 下载失败时重新下载的次数， 默认3
    :param min_byte（float, optional）: 一个下载文件最小的字节数，默认1E0
    :param exist_ok（bool, optional）: 解压过程中是否覆盖已存在的内容，默认False
    :param progress（bool, optional）: 下载过程是否显示进度条，默认True
    :return:
    """
    gdrive = url.startswith("https://drive.google.com/")  #检测是否Google链接
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = Path(dir or ".")/(file or url2file(url))    #URL -> filename
    if "://" not in str(url) and Path(url).is_file():  #确保URL的存在 （"://"的检查被python<3.10需要）
        f = Path(url)   #filename
    elif not f.is_file():  #URL和文件不存在
        desc = f"下载{url if gdrive else clean_url(url)} 到 {f}"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)   #创建目录
        check_disk_space(url)
        for i in range(retry + 1):
            try:
                if curl or i > 0:   #curl download
                    s = "sS" * (not progress)
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode   #执行外部命令进行下载
                    assert r == 0, f"Curl return value.{r}"
                else:    #urllib download
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)   #下载文件
                    else:
                        with request.urlopen(url) as response:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                if f.exists():
                    if f.stat().st_size > min_byte:
                        break
                    f.unlink()

            except Exception as e:
                if i ==0 and not is_online():
                    raise ConnectionError(emojis(f"❌ {url}下载失败，网络未连接")) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f"❌ {url}下载失败，超出重试次数")) from e
                LOGGER.warning(f"WARNING⚠️ 下载失败，重试{i+1}/{retry}  {url}...")
    #解压缩
    if unzip and f.exists() and f.suffix in ("","zip",".tar",".gz"):
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve() #解压缩到父级目录，如果dir未提供
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  #解压缩
        elif f.suffix in (".tar", ".gz"):
            LOGGER.info(f"解压缩{f}到{unzip_dir}...")
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)  #运行外部命令
        if delete:
            f.unlink()  #删除zip文件
        return unzip_dir

def download(url, dir=Path.cwd(), unzip=True, delete=False,curl=False, threads=1, retry=3, exist_ok=False):
    dir =Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=threads<=1
                ),
                zip(url, repeat(dir))
            )
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)

def get_github_assets(repo="ultralytics/assets", version="latest", retry=False):
    """
    从GIthub存储库中获取指定版本的tag和assets。如果版本不存在，则获取自信版本的assets

    Args:
        repo(str, optional): 格式为‘owner/repo’的Github的存储库，默认为‘ultralytics/assets’
        version（str, optional）: 从中获取assets的发布版本，默认”lastest
        retry(bool, optional): 请求失败是否重试， 默认False
    Returns:
        (tuple): 一个包含版本tag和asset names列表的元组
    Example:
        ```python
        tag, assets = get_github_assets(repo='ultralytics/assets', version='latest')
    """
    if version != "latest":
        version = f"tags/{version}"  #i.e. tags/v6.2
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)  #github api
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:  #请求失败并且不是403速度限制导致，重试
        r = requests.get(url)
    if r.status_code != 200:
        LOGGER.warning(f"WARNING⚠️GitHub assets检测失败{url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]   #tag, assets i.e. ['yolov8n.pt', 'yolov8s.pt', ...]


def attempt_download_asset(file, repo="ultralytics/assets", release="v8.1.0", **kwargs):
    """
    如果本地未发现file，则尝试从GitHub rekease assets下载file
    :param file（str | Path）: 文件名或文件路径
    :param repo（str, optional）: GitHub存储库格式 'owner/repo'， 默认 'yltralytic/assets'
    :param release（str, optional）: 指定下载文件的发布版本，默认v8.1.0
    :param kwargs:
    :return(str): 下载完成文件的路径
    Example:
        ```python
        file_path = attempt_download_asset('yolov5s.pt', repo='ultralytics/assets', release='latest')
    """
    from ultralytics.utils import SETTINGS

    #YOLOv3/5u 更新
    file = str(file)
    file = checks.check_yolo5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        #URL
        name = Path(parse.unquote(str(file))).name   #解码  ‘%2F’ ->  '/'
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):   #满足下载条件下载
            url = str(file).replace(":/", "://")    #Pathlib 将:// -> :/
            file = url2file(name)   #要下载的文件名
            if Path(file).is_file():
                LOGGER.info(f"在本地{file}发现{clean_url(url)} ")    #文件已经存在
            else:
                safe_download(url=url, file=file,min_byte=1e5, **kwargs)
        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_byte=1e5, **kwargs)
        else:
            tag, assets = get_github_assets(repo,release)
            if not assets:
                tag, assets = get_github_assets(repo)  #最新版本
            if name in assets:
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_byte=1e5, **kwargs)
        return str(file)

def is_url(url, check=True):
    """
    判断所给的字符串是否一个URL，并检查这个URL在互联网上是否存在（可选）
    """
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  #is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200
        return True
    return False