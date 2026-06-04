# -*- mode: python ; coding: utf-8 -*-
"""
Sythm.spec — build standalone ONEFILE, console masquee.

    .venv\\Scripts\\pyinstaller.exe Sythm.spec --noconfirm

Strategie CUDA (cf. investigation) : le visualiseur n'a besoin AU RUNTIME que de
  - nvrtc64_*.dll + nvrtc-builtins64_*.dll   (compilation des kernels CUDA)
  - cudart64_*.dll                           (assurance ; souvent fourni par le pilote)
  - les headers CUDA  (include/)             (NVRTC compile des kernels qui les #include)
On N'EMBARQUE PAS cuFFT (FFT deplacee en CPU), ni cuBLAS/cuBLASLt (matmuls audio
remplaces par des reductions), ni nvJitLink (compilation PTX standard). Le pilote
NVIDIA de la machine cible fournit nvcuda.dll. -> ~130 Mo de CUDA au lieu de ~1.5 Go.
"""
import os
import glob
from PyInstaller.utils.hooks import collect_all

# --- Localisation du CUDA Toolkit (pour piocher DLL + headers a la compilation)
CUDA = os.environ.get("CUDA_PATH")
if not CUDA or not os.path.isdir(CUDA):
    _cands = sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"))
    CUDA = _cands[-1] if _cands else None
assert CUDA and os.path.isdir(CUDA), "CUDA_PATH introuvable : definis-le vers le toolkit."
CUDA_BIN = os.path.join(CUDA, "bin", "x64")

# --- DLL CUDA minimales -> _MEIPASS/bin/x64/ --------------------------------
binaries = []
for _pat in ("nvrtc64_*.dll", "nvrtc-builtins64_*.dll", "cudart64_*.dll"):
    for _f in glob.glob(os.path.join(CUDA_BIN, _pat)):
        binaries.append((_f, "bin/x64"))
assert any("nvrtc64" in b[0] for b in binaries), f"nvrtc introuvable dans {CUDA_BIN}"

# --- Donnees : shaders GLSL + headers CUDA ----------------------------------
datas = [
    ("shaders", "shaders"),
    (os.path.join(CUDA, "include"), "include"),   # headers pour NVRTC
]

# --- Paquets a collecter integralement (code + .pyd + data) -----------------
# NB : 'graphlib' (stdlib) est importe depuis du Cython compile (cupy/_core/
# _scalar.pyx) -> invisible a l'analyse statique de PyInstaller, a declarer ici.
hiddenimports = ["cuda", "cuda.pathfinder", "cffi", "_cffi_backend", "graphlib"]
for _pkg in ("cupy", "cupy_backends", "cupyx", "cuda",
             "soundcard", "glfw", "imageio_ffmpeg", "moderngl", "glcontext"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# --- Exclusions (allege le bundle ; rien de tout ca n'est utilise) ----------
excludes = ["tkinter", "matplotlib", "scipy", "PIL", "pandas", "pytest",
            "IPython", "notebook", "sphinx", "setuptools", "pip", "wheel"]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rt_hook.py"],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# --- Retire les grosses DLL CUDA INUTILISEES que PyInstaller capture via les
#     dependances PE des .pyd cupy_backends (cufft/cublas/cublasLt/... trouvees
#     sur le PATH du toolkit a la compilation). Elles ne sont JAMAIS chargees au
#     runtime (FFT deplacee en CPU, matmuls audio -> reductions) : verifie par
#     instrumentation du process. -> ~900 Mo economises.
_DROP = ("cublas64_", "cublaslt64_", "cufft64_", "cufftw64_", "curand64_",
         "cusolver64_", "cusolvermg64_", "cusparse64_", "cutensor",
         "nvjitlink_", "nvjpeg", "cudnn", "nppc64_", "nppi", "npps")
_before = len(a.binaries)
a.binaries = [b for b in a.binaries
              if not os.path.basename(b[0]).lower().startswith(_DROP)]
print(f"[Sythm.spec] DLL CUDA inutiles retirees de a.binaries : "
      f"{_before - len(a.binaries)}")

# --- Retire les copies de nvrtc/cudart/builtins placees A LA RACINE de _MEIPASS
#     par l'analyse de dependances de PyInstaller (deps PE des .pyd cupy). On ne
#     garde que celles de bin/x64/ : sinon cuda-pathfinder trouve la copie racine
#     et CuPy derive un cuda_path FAUX (= parent de _MEIPASS), d'ou un crash
#     `os.add_dll_directory(...\\Temp\\bin)` au tout debut de `import cupy`.
_RUNTIME = ("nvrtc64_", "nvrtc-builtins64_", "cudart64_")
def _is_root_runtime(dest):
    d = dest.replace("/", "\\")
    return ("\\" not in d) and os.path.basename(d).lower().startswith(_RUNTIME)
_before = len(a.binaries)
a.binaries = [b for b in a.binaries if not _is_root_runtime(b[0])]
print(f"[Sythm.spec] copies racine nvrtc/cudart retirees : "
      f"{_before - len(a.binaries)}  (gardees uniquement dans bin/x64/)")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Sythm",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX casse souvent les DLL CUDA -> desactive
    runtime_tmpdir=None,
    console=False,             # <<< CONSOLE MASQUEE
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="logo.ico",          # icone de l'executable (depuis GitHub/logo.png)
)
