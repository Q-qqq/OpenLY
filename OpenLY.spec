# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks  import collect_submodules,collect_data_files


def find_py_paths(root_dir):
    return [os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir,  d))]

datas = collect_data_files('ultralytics')
a = Analysis(
    ['OpenLY.py'],
    pathex=[],
    binaries=[],
    datas=[("APP/resources", "APP/resources"), ("SETTINGS.yaml", "."), ("yolov8n.pt", ".")] + datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
a.pathex.extend(find_py_paths('APP'))
a.pathex.extend(find_py_paths('ultralytics'))
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OpenLY',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OpenLY',
)
