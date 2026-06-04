# -*- coding: utf-8 -*-
"""
rt_hook.py — runtime hook PyInstaller (s'execute AVANT main.py, donc avant
`import cupy`). Deux roles :

  1. CUDA EMBARQUE : on pointe CuPy / cuda-pathfinder vers les bibliotheques et
     headers CUDA empaquetes dans le bundle, de sorte que l'utilisateur final
     n'ait PAS besoin du CUDA Toolkit (un pilote NVIDIA recent suffit). Le
     bundle reproduit la disposition d'un toolkit :
         <_MEIPASS>/bin/x64/{nvrtc*, cudart*}.dll
         <_MEIPASS>/include/      (headers CUDA pour la compilation NVRTC)
     CuPy derive sa racine CUDA de l'emplacement de nvrtc (via pathfinder) ;
     definir CUDA_PATH=<_MEIPASS> rend cette derivation correcte.

  2. CONSOLE MASQUEE (--windowed) : sans console, sys.stdout/stderr valent None ;
     un print()/flush() leve alors une exception. On les redirige vers un fichier
     log a cote de l'executable pour que le programme survive et reste debogable.
"""
import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

# --- 1. CUDA embarque -------------------------------------------------------
# On force CUDA_PATH vers le bundle (version connue et testee), meme si la
# machine possede deja un toolkit : reproductibilite garantie.
os.environ["CUDA_PATH"] = _base
os.environ.setdefault("CUDA_HOME", _base)
for _sub in (os.path.join(_base, "bin", "x64"), os.path.join(_base, "bin")):
    if os.path.isdir(_sub):
        try:
            os.add_dll_directory(_sub)
        except (OSError, AttributeError):
            pass

# --- 2. stdout/stderr valides en mode fenetre (console masquee) -------------
# En --windowed, PyInstaller remplace stdout/stderr par un ecriveur muet (ou
# None) : un print()/flush() peut lever, et surtout toute trace d'erreur est
# perdue. On redirige donc INCONDITIONNELLEMENT vers un fichier log a cote de
# l'executable, pour que le programme survive ET reste debogable.
try:
    _logp = os.path.join(os.path.dirname(sys.executable), "sythm.log")
    _sink = open(_logp, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _sink
    sys.stderr = _sink
    print(f"[rt_hook] demarrage. _MEIPASS={_base}")
    print(f"[rt_hook] CUDA_PATH={os.environ.get('CUDA_PATH')}")
    print(f"[rt_hook] CUDA embarque : bin/x64="
          f"{os.path.isdir(os.path.join(_base, 'bin', 'x64'))} "
          f"include={os.path.isdir(os.path.join(_base, 'include'))}")
except OSError:
    pass
