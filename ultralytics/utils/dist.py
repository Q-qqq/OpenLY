import os
import shutil
import socket
import sys
import tempfile
from ultralytics.utils import USER_CONFIG_DIR

def find_free_network_port() -> int:
    """寻找一个本地主机ip的空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind("127.0.0.1", 0)
        return s.getsockname()[1] #port

def generate_ddp_file(trainer):
    """产生一个ddp文件，并返回他的文件名"""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix="_temp_",
            suffix=f"{id(trainer)}.py",
            mode="w+",
            encoding="utf-8",
            dir=USER_CONFIG_DIR / "DDP",
            delete=False,
    ) as file:
        file.write(content)
    return file.name

def generate_ddp_command(world_size, trainer):
    import __main__
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)   #清空保存的目录
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file

def ddp_cleanup(trainer, file):
    if f"{id(trainer)}.py" in file:   #temp 文件后缀在file里
        os.remove(file)
