# -*- coding: utf-8 -*-
"""
gl_interop.py
=============
INTEROP ZÉRO-COPIE CUDA <-> OpenGL — extraite de particles.py (préoccupation à part).

Le simulateur (`particles.ParticleSystem`) ne s'occupe QUE de la simulation et du
LANCEMENT des kernels ; il ignore désormais ctypes, les pointeurs device, le
map/unmap et le repli. Tout cela vit ici :

  _CudaGLDriver    : fin wrapper ctypes sur l'API DRIVER CUDA (cuGraphics*), sans
                     PyCUDA — enregistre/mappe des buffers OpenGL pour CUDA.
  GLInteropBuffers : gère une PAIRE de buffers GL (positions, couleurs) en
                     ZÉRO-COPIE, avec REPLI transparent (upload CuPy->GL) si l'interop
                     échoue. On lui passe la fonction de lancement des kernels via
                     `run(launch_fn)` : elle mappe (ou prend les buffers de repli),
                     appelle `launch_fn(gl_pos, gl_col)` qui ÉCRIT dedans, puis
                     démappe+synchronise (ou uploade). Repli automatique sur erreur.
"""
from __future__ import annotations

import sys
import ctypes

try:
    import cupy as cp                       # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None                                # type: ignore
    _HAS_CUPY = False

_CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02


# ===========================================================================
#  Wrapper ctypes bas niveau sur l'API DRIVER CUDA (cuGraphics*).
# ===========================================================================
class _CudaGLDriver:
    def __init__(self):
        lib = self._load_driver()
        if lib is None:
            raise RuntimeError("driver CUDA introuvable (nvcuda.dll / libcuda.so)")
        self._lib = lib
        self._c_init = lib.cuInit
        self._c_init.restype = ctypes.c_int
        self._c_init.argtypes = [ctypes.c_uint]
        self._check(self._c_init(0), "cuInit")
        self._c_register = lib.cuGraphicsGLRegisterBuffer
        self._c_register.restype = ctypes.c_int
        self._c_register.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint]
        self._c_map = lib.cuGraphicsMapResources
        self._c_map.restype = ctypes.c_int
        self._c_map.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        getptr = getattr(lib, "cuGraphicsResourceGetMappedPointer_v2", None)
        if getptr is None:
            getptr = lib.cuGraphicsResourceGetMappedPointer
        self._c_getptr = getptr
        self._c_getptr.restype = ctypes.c_int
        self._c_getptr.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
        self._c_unmap = lib.cuGraphicsUnmapResources
        self._c_unmap.restype = ctypes.c_int
        self._c_unmap.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self._c_unregister = lib.cuGraphicsUnregisterResource
        self._c_unregister.restype = ctypes.c_int
        self._c_unregister.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _load_driver():
        for name in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        return None

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} -> CUresult={rc}")

    def register(self, glo):
        res = ctypes.c_void_p()
        self._check(self._c_register(ctypes.byref(res), ctypes.c_uint(int(glo)),
                    ctypes.c_uint(_CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)),
                    "cuGraphicsGLRegisterBuffer")
        return res

    def map(self, res_array, stream_ptr):
        self._check(self._c_map(ctypes.c_uint(len(res_array)), res_array,
                    ctypes.c_void_p(stream_ptr)), "cuGraphicsMapResources")

    def get_pointer(self, res_array, index):
        dptr = ctypes.c_uint64(); size = ctypes.c_size_t()
        self._check(self._c_getptr(ctypes.byref(dptr), ctypes.byref(size),
                    ctypes.c_void_p(res_array[index])), "cuGraphicsResourceGetMappedPointer")
        return int(dptr.value), int(size.value)

    def unmap(self, res_array, stream_ptr):
        self._check(self._c_unmap(ctypes.c_uint(len(res_array)), res_array,
                    ctypes.c_void_p(stream_ptr)), "cuGraphicsUnmapResources")

    def unregister(self, res):
        self._check(self._c_unregister(res), "cuGraphicsUnregisterResource")


# ===========================================================================
#  Interop d'une PAIRE de buffers GL (positions, couleurs) — zéro-copie + repli.
# ===========================================================================
class GLInteropBuffers:
    """Interop ZÉRO-COPIE pour les deux buffers GL (pos, col) du Renderer, avec REPLI
    transparent (upload CuPy->GL). Le simulateur passe sa fonction de lancement de
    kernels à `run(launch_fn)` ; cette classe lui fournit les deux tableaux CuPy à
    remplir (mappés en zéro-copie, ou les buffers de repli) et s'occupe ensuite du
    démappage+synchro (ou de l'upload). N'expose ni ctypes ni pointeur device."""

    def __init__(self, gl_pos_buffer, gl_col_buffer, n_total, stream):
        self._gl_pos_buffer = gl_pos_buffer
        self._gl_col_buffer = gl_col_buffer
        self.n_total = int(n_total)
        self._stream = stream
        self.zero_copy = False
        self._gl = None
        self._res_pos = None
        self._res_col = None
        self._res_array = None
        self._fallback_pos = None
        self._fallback_col = None
        self._setup()

    @property
    def path_label(self) -> str:
        """Étiquette de chemin pour le log de démarrage."""
        return "ZERO-COPIE (driver CUDA/GL)" if self.zero_copy else "REPLI (upload CuPy->GL)"

    # ------------------------------------------------------------- setup
    def _setup(self):
        try:
            pos_glo = int(self._gl_pos_buffer.glo)
            col_glo = int(self._gl_col_buffer.glo)
        except Exception as exc:
            print(f"[particles] .glo introuvable ({exc}) -> repli.", file=sys.stderr)
            self._init_fallback(); return
        try:
            cp.cuda.Device(0).use()
            self._gl = _CudaGLDriver()
            self._res_pos = self._gl.register(pos_glo)
            self._res_col = self._gl.register(col_glo)
            self._res_array = (ctypes.c_void_p * 2)(self._res_pos.value, self._res_col.value)
            self.zero_copy = True
        except Exception as exc:
            print(f"[particles] interop zéro-copie indisponible ({exc}) -> repli.", file=sys.stderr)
            if self._gl is not None:
                for res in (self._res_pos, self._res_col):
                    if res is not None:
                        try: self._gl.unregister(res)
                        except Exception: pass
            self._gl = None; self._res_pos = None; self._res_col = None; self._res_array = None
            self._init_fallback()

    def _init_fallback(self):
        # Le REPLI alloue 2 buffers CuPy de plus (n_total*4 floats chacun) en VRAM.
        # Sur une carte serrée où le zéro-copie a échoué, ça peut manquer de mémoire
        # -> on échoue PROPREMENT (message actionnable) plutôt qu'avec un OOM brut.
        self.zero_copy = False
        try:
            self._fallback_pos = cp.empty(self.n_total * 4, dtype=cp.float32)
            self._fallback_col = cp.empty(self.n_total * 4, dtype=cp.float32)
        except Exception as exc:
            raise RuntimeError(
                f"[particles] VRAM insuffisante pour le repli upload à "
                f"{self.n_total:,} particules. Baissez N_PARTICLES / EMIT_RATE / "
                f"EMITTED_LIFETIME dans la fenêtre de config.") from exc

    def _switch_to_fallback(self):
        if self._gl is not None:
            for res in (self._res_pos, self._res_col):
                if res is not None:
                    try: self._gl.unregister(res)
                    except Exception: pass
        self._gl = None; self._res_pos = None; self._res_col = None; self._res_array = None
        self._init_fallback()

    def _wrap_device_ptr(self, ptr, nbytes):
        mem = cp.cuda.UnownedMemory(int(ptr), int(nbytes), owner=self)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray((self.n_total * 4,), dtype=cp.float32, memptr=memptr)

    # ------------------------------------------------------------- run / commit
    def run(self, launch_fn):
        """Mappe (zéro-copie) ou prend les buffers de repli, appelle
        launch_fn(gl_pos, gl_col) qui ÉCRIT dedans, puis démappe+synchronise (ou
        uploade+synchronise). Sur erreur d'interop : bascule en repli et RELANCE la
        frame en repli (transparent pour l'appelant)."""
        if not self.zero_copy:
            self._run_fallback(launch_fn)
            return
        arr = self._res_array
        stream_ptr = self._stream.ptr
        mapped = False
        try:
            self._gl.map(arr, stream_ptr); mapped = True
            pos_ptr, pos_size = self._gl.get_pointer(arr, 0)
            col_ptr, col_size = self._gl.get_pointer(arr, 1)
            gl_pos = self._wrap_device_ptr(pos_ptr, pos_size)
            gl_col = self._wrap_device_ptr(col_ptr, col_size)
            launch_fn(gl_pos, gl_col)
            self._gl.unmap(arr, stream_ptr); mapped = False
            self._stream.synchronize()
        except Exception as exc:
            print(f"[particles] interop KO ({exc}) -> repli.", file=sys.stderr)
            if mapped:
                try: self._gl.unmap(arr, stream_ptr)
                except Exception: pass
            try: self._stream.synchronize()
            except Exception: pass
            self._switch_to_fallback()
            self._run_fallback(launch_fn)

    def _run_fallback(self, launch_fn):
        launch_fn(self._fallback_pos, self._fallback_col)
        self._stream.synchronize()
        self._gl_pos_buffer.write(cp.asnumpy(self._fallback_pos))
        self._gl_col_buffer.write(cp.asnumpy(self._fallback_col))

    def release(self):
        """Désenregistre les ressources d'interop (le stream est synchronisé par
        l'appelant AVANT cet appel)."""
        if self._gl is not None:
            for res in (self._res_pos, self._res_col):
                if res is not None:
                    try: self._gl.unregister(res)
                    except Exception as exc:
                        print(f"[particles] erreur désenreg. interop : {exc}", file=sys.stderr)
        self._gl = None; self._res_pos = None; self._res_col = None; self._res_array = None
        self.zero_copy = False
        self._fallback_pos = None; self._fallback_col = None
