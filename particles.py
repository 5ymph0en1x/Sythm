# -*- coding: utf-8 -*-
"""
particles.py
============

NUÉE (champ caché Lorenz) + ÉMISSION DE TRAÎNÉES ÉPHÉMÈRES + INTEROP CUDA/GL
---------------------------------------------------------------------------
Visualiseur de particules 3D audio-réactif (RTX 4090). Python + CUDA C (CuPy) +
interop ZÉRO-COPIE avec les buffers OpenGL.

DEUX POPULATIONS
================
  1. ORIGINES (n_origin) : remplissent l'espace visible et sont advectées par un
     CHAMP DE VITESSE 3D (flot ABC) dont les coefficients A,B,C SONT l'état d'un
     système de LORENZ CACHÉ (jamais affiché — il sous-tend la dynamique). Flot à
     divergence nulle -> space-filling, chaotique. Chaque origine MÉMORISE son
     vecteur-vitesse (déplacement immédiat).
  2. ÉMISES (n_emit) : chaque frame, chaque origine ÉMET des particules de courte
     DURÉE DE VIE (≈1 s), lancées en BALISTIQUE dans la DIRECTION et à la VITESSE
     du déplacement immédiat de l'origine émettrice. -> des traînées vivantes qui
     révèlent le flux. Tens of millions.

POOL D'ÉMISES = RING BUFFER
===========================
On émet E = n_emit / lifetime * dt particules/frame en avançant une tête
d'écriture circulaire. Comme on réécrit au rythme exact de la mort (lifetime), le
slot recyclé est justement expiré : recyclage sans recherche de morts. La
luminosité d'une émise décroît avec l'âge -> elle « meurt » en s'éteignant.

RENDU « chaque particule nette »
================================
Luminosité par particule UNIFORME (pas de gain par densité) ; bloom/motion-blur
OFF côté Integrator ; émises faibles + fondu par l'âge.

CONTRAT (mis à jour pour l'émission — main.py est le seul appelant)
==================================================================
    ParticleSystem(n_origin, n_emit, gl_pos_buffer, gl_col_buffer,
                   shape='sphere', radius=1.0, lifetime=1.0)
    .update(dt, features) ; .release()
Buffers GL : taille n_origin+n_emit ; POSITION=(x,y,z,brightness), COULEUR=(r,g,b,a).
Interop : API DRIVER CUDA (ctypes), contexte primaire CuPy ; repli upload si KO.
"""

from __future__ import annotations

import sys
import math
import ctypes

import numpy as np

try:
    import cupy as cp                       # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None                                # type: ignore
    _HAS_CUPY = False

_CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02


# ===========================================================================
#  Source CUDA C
# ===========================================================================
_CUDA_SOURCE = r"""
extern "C" {

__device__ __forceinline__ unsigned int hash_u(unsigned int x){
    x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x;
}
__device__ __forceinline__ float hash_f(unsigned int x){
    return (hash_u(x)&0x00FFFFFFu)*(1.0f/16777216.0f);
}
__device__ __forceinline__ float hash3i(int xi,int yi,int zi){
    unsigned int h=(unsigned int)(xi*73856093)^(unsigned int)(yi*19349663)^(unsigned int)(zi*83492791);
    return hash_f(h);
}
__device__ __forceinline__ float fadef(float t){ return t*t*t*(t*(t*6.0f-15.0f)+10.0f); }
__device__ __forceinline__ float lerpf(float a,float b,float t){ return a+t*(b-a); }
__device__ float vnoise3(float x,float y,float z){
    int xi=(int)floorf(x),yi=(int)floorf(y),zi=(int)floorf(z);
    float xf=x-xi,yf=y-yi,zf=z-zi; float u=fadef(xf),v=fadef(yf),w=fadef(zf);
    float c000=hash3i(xi,yi,zi),c100=hash3i(xi+1,yi,zi),c010=hash3i(xi,yi+1,zi),c110=hash3i(xi+1,yi+1,zi);
    float c001=hash3i(xi,yi,zi+1),c101=hash3i(xi+1,yi,zi+1),c011=hash3i(xi,yi+1,zi+1),c111=hash3i(xi+1,yi+1,zi+1);
    float x00=lerpf(c000,c100,u),x10=lerpf(c010,c110,u),x01=lerpf(c001,c101,u),x11=lerpf(c011,c111,u);
    return lerpf(lerpf(x00,x10,v),lerpf(x01,x11,v),w);
}
__device__ void curl_noise(float x,float y,float z,float* out){
    const float e=0.4f, inv2e=1.0f/(2.0f*e);
    float p_y1=vnoise3(x,y+e,z)-vnoise3(x,y-e,z);
    float p_z1=vnoise3(x,y,z+e)-vnoise3(x,y,z-e);
    float ox=31.4f,oy=17.7f,oz=47.1f;
    float q_x1=vnoise3(x+e+ox,y+oy,z+oz)-vnoise3(x-e+ox,y+oy,z+oz);
    float q_z1=vnoise3(x+ox,y+oy,z+e+oz)-vnoise3(x+ox,y+oy,z-e+oz);
    float rx=-59.2f,ry=11.3f,rz=23.8f;
    float r_x1=vnoise3(x+e+rx,y+ry,z+rz)-vnoise3(x-e+rx,y+ry,z+rz);
    float r_y1=vnoise3(x+rx,y+e+ry,z+rz)-vnoise3(x+rx,y-e+ry,z+rz);
    out[0]=(r_y1-q_z1)*inv2e; out[1]=(p_z1-r_x1)*inv2e; out[2]=(q_x1-p_y1)*inv2e;
}
__device__ void hsv2rgb(float h,float s,float v,float* rgb){
    h=h-floorf(h); float i=floorf(h*6.0f); float f=h*6.0f-i;
    float p=v*(1.0f-s),q=v*(1.0f-f*s),t=v*(1.0f-(1.0f-f)*s); int ii=((int)i)%6; float r,g,b;
    switch(ii){case 0:r=v;g=t;b=p;break;case 1:r=q;g=v;b=p;break;case 2:r=p;g=v;b=t;break;
        case 3:r=p;g=q;b=v;break;case 4:r=t;g=p;b=v;break;default:r=v;g=p;b=q;break;}
    rgb[0]=r;rgb[1]=g;rgb[2]=b;
}

// INIT origines : remplissage uniforme de la boîte [-L,L]^3.
__global__ void init_field(float* pos, const int n, const float L, const unsigned int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    unsigned int b=seed+(unsigned int)i*2654435761u;
    pos[i*3+0]=(hash_f(b+0u)*2.0f-1.0f)*L;
    pos[i*3+1]=(hash_f(b+1u)*2.0f-1.0f)*L;
    pos[i*3+2]=(hash_f(b+2u)*2.0f-1.0f)*L;
}

// Couleur RICHE pilotée par les BANDES DE FRÉQUENCES (sort du vert-jaune) :
//   grave = rouge (0), médium = vert (0.33), aigu = bleu (0.66) -> teinte spectrale
//   centrée sur la fréquence dominante, + étalement par particule/vitesse pour la
//   richesse spatiale. Saturation dopée par le beat et les aigus.
__device__ void spectral_color(
    float hue_id, float local, float t,
    float bass, float mid, float high, float beat, float val, float* rgb)
{
    float wsum = bass + mid + high + 0.02f;
    float spec = (mid * 0.33f + high * 0.66f) / wsum;   // 0 rouge, 0.33 vert, 0.66 bleu
    float hue = spec + (hue_id - 0.5f) * 0.30f + (local - 0.5f) * 0.20f + t * 0.01f;
    float sat = fminf(0.97f, 0.75f + 0.20f * beat + 0.12f * high);
    hsv2rgb(hue, sat, val, rgb);
}

// UPDATE origines : advection champ ABC(Lorenz) + MÉMORISE la vitesse + écrit GL.
__global__ void update_origin(
    float* pos, float* vel, float* gl_pos, float* gl_col,
    const int n, const float t, const float dt, const float L,
    const float lx, const float ly, const float lz,
    const float field_strength, const float k, const float turb_base,
    const float amp, const float beat, const float centroid,
    const float bass, const float mid, const float high)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float px=pos[i*3+0],py=pos[i*3+1],pz=pos[i*3+2];
    float kx=px*k,ky=py*k,kz=pz*k;
    float fx=lx*sinf(kz)+lz*cosf(ky);
    float fy=ly*sinf(kx)+lx*cosf(kz);
    float fz=lz*sinf(ky)+ly*cosf(kx);
    // MOUVEMENT calé sur les BASSES : les graves accélèrent le flux + la turbulence.
    float spd=field_strength*(1.0f+0.5f*amp+2.0f*bass);
    float c[3]; float ns=0.6f;
    curl_noise(px*ns+t*0.2f, py*ns, pz*ns-t*0.2f, c);
    float turb=turb_base*(1.0f+4.0f*beat+2.5f*bass);
    // Vitesse de déplacement immédiat (sert AUSSI à lancer les particules émises).
    float tvx=fx*spd + c[0]*turb;
    float tvy=fy*spd + c[1]*turb;
    float tvz=fz*spd + c[2]*turb;
    px+=tvx*dt; py+=tvy*dt; pz+=tvz*dt;
    float W=2.0f*L;
    if(px>L)px-=W; else if(px<-L)px+=W;
    if(py>L)py-=W; else if(py<-L)py+=W;
    if(pz>L)pz-=W; else if(pz<-L)pz+=W;
    pos[i*3+0]=px; pos[i*3+1]=py; pos[i*3+2]=pz;
    vel[i*3+0]=tvx; vel[i*3+1]=tvy; vel[i*3+2]=tvz;
    // couleur RICHE pilotée par les bandes de fréquences (cf. spectral_color).
    float vlen=sqrtf(tvx*tvx+tvy*tvy+tvz*tvz)+1e-4f;
    float local=tvz/vlen*0.5f+0.5f;
    float val=fminf(0.50f+amp*0.25f+beat*0.30f+bass*0.30f, 1.15f);  // basses -> + lumineux
    float rgb[3];
    spectral_color(hash_f((unsigned int)i), local, t, bass, mid, high, beat, val, rgb);
    float brightness=0.55f;
    gl_pos[i*4+0]=px; gl_pos[i*4+1]=py; gl_pos[i*4+2]=pz; gl_pos[i*4+3]=brightness;
    gl_col[i*4+0]=rgb[0]; gl_col[i*4+1]=rgb[1]; gl_col[i*4+2]=rgb[2]; gl_col[i*4+3]=1.0f;
}

// ÉMISSION : E particules réécrites dans le ring depuis des origines.
//   es[slot*7+0..2]=pos, +3..5=vel (déplacement immédiat de l'origine), +6=age(0).
__global__ void emit_particles(
    const float* opos, const float* ovel, float* es,
    const int E, const int n_origin, const int n_emit, const int head)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=E) return;
    int slot=(head+tid)%n_emit;
    int src=(head+tid)%n_origin;             // balaie les origines uniformément
    es[slot*7+0]=opos[src*3+0]; es[slot*7+1]=opos[src*3+1]; es[slot*7+2]=opos[src*3+2];
    es[slot*7+3]=ovel[src*3+0]; es[slot*7+4]=ovel[src*3+1]; es[slot*7+5]=ovel[src*3+2];
    es[slot*7+6]=0.0f;
}

// UPDATE émises : intégration balistique + fondu par l'âge + écrit GL (offset n_origin).
__global__ void update_emitted(
    float* es, float* gl_pos, float* gl_col,
    const int n_emit, const int n_origin, const float dt,
    const float lifetime, const float emit_bright, const float centroid,
    const float t, const float bass, const float mid, const float high, const float beat)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n_emit) return;
    float px=es[i*7+0],py=es[i*7+1],pz=es[i*7+2];
    float vx=es[i*7+3],vy=es[i*7+4],vz=es[i*7+5];
    float age=es[i*7+6]+dt;
    // MOUVEMENT calé sur les BASSES : les graves font surgir les traînées.
    float em_speed=1.0f+1.2f*bass;
    px+=vx*dt*em_speed; py+=vy*dt*em_speed; pz+=vz*dt*em_speed;
    es[i*7+0]=px; es[i*7+1]=py; es[i*7+2]=pz; es[i*7+6]=age;

    int gi=n_origin+i;
    float bright=0.0f, alpha=0.0f; float rgb[3]={0.0f,0.0f,0.0f};
    if (age < lifetime) {
        float u = age / lifetime;                // 0 -> 1
        // Enveloppe DOUCE : court fondu d'apparition + extinction en (1-u)^2
        // (valeur ET pente nulles à la mort) -> AUCUNE coupure franche.
        float fin = fminf(u * 12.5f, 1.0f);
        float fout = 1.0f - u; fout = fout * fout;
        float env = fin * fout;
        float vlen=sqrtf(vx*vx+vy*vy+vz*vz)+1e-4f;
        float local=vz/vlen*0.5f+0.5f;
        spectral_color(hash_f((unsigned int)i), local, t, bass, mid, high, beat, 0.75f, rgb);
        bright=emit_bright*env;                  // apparition + extinction douces
        alpha=1.0f;
    }
    gl_pos[gi*4+0]=px; gl_pos[gi*4+1]=py; gl_pos[gi*4+2]=pz; gl_pos[gi*4+3]=bright;
    gl_col[gi*4+0]=rgb[0]; gl_col[gi*4+1]=rgb[1]; gl_col[gi*4+2]=rgb[2]; gl_col[gi*4+3]=alpha;
}

}  // extern "C"
"""


# ===========================================================================
#  Constantes (réglables à l'œil)
# ===========================================================================
_THREADS_PER_BLOCK = 256
_SHAPE_MODES = {"sphere": 0, "cube": 1, "lorenz": 2, "murmuration": 3}

_FIELD_STRENGTH = 0.9
_WAVE = 1.2
_TURB_BASE = 0.25
_EMIT_BRIGHT = 0.45        # luminosité de base d'une particule émise (fond à 0)

_LZ_SIGMA, _LZ_RHO, _LZ_BETA = 10.0, 28.0, 2.6666667
_LZ_ZC = 25.0
_LZ_NX, _LZ_NY, _LZ_NZ = 1.0/18.0, 1.0/24.0, 1.0/24.0
_GUIDE_RATE = 3.0
_GUIDE_HMAX = 0.01


# ===========================================================================
#  Interop CUDA <-> OpenGL via l'API DRIVER CUDA (ctypes) — sans PyCUDA.
# ===========================================================================
class _GLInterop:
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


class ParticleSystem:
    """Nuée (champ Lorenz caché) + émission de traînées éphémères ; interop GL."""

    def __init__(self, n_origin, n_emit, gl_pos_buffer, gl_col_buffer,
                 shape='sphere', radius=1.0, lifetime=1.0):
        if not _HAS_CUPY:
            raise RuntimeError("[particles] CuPy indisponible : installe cupy-cuda13x.")
        self.n_origin = int(n_origin)
        self.n_emit = int(n_emit)
        self.n_total = self.n_origin + self.n_emit
        self.radius = float(radius)
        self.lifetime = float(lifetime)
        self._gl_pos_buffer = gl_pos_buffer
        self._gl_col_buffer = gl_col_buffer
        self._nbytes = self.n_total * 4 * 4
        self._t = 0.0
        self._centroid = 0.5
        self._emit_head = 0
        self._lx, self._ly, self._lz = 0.9, 0.0, 25.0
        # scalaires de la frame courante (remplis par update, lus par _launch).
        self._cur = {}

        self._stream = cp.cuda.Stream(non_blocking=True)
        self._module = cp.RawModule(code=_CUDA_SOURCE, options=("--use_fast_math",))
        self._k_init = self._module.get_function("init_field")
        self._k_origin = self._module.get_function("update_origin")
        self._k_emit = self._module.get_function("emit_particles")
        self._k_emitted = self._module.get_function("update_emitted")

        f32 = cp.float32
        self.pos_state = cp.empty(self.n_origin * 3, dtype=f32)   # origines : position
        self.vel_state = cp.zeros(self.n_origin * 3, dtype=f32)   # origines : vitesse
        # Pool d'émises : (pos3, vel3, age1). age initial > lifetime -> invisibles.
        self.emit_state = cp.zeros(self.n_emit * 7, dtype=f32)
        self.emit_state.reshape(self.n_emit, 7)[:, 6] = self.lifetime + 1.0

        self._block = (_THREADS_PER_BLOCK, 1, 1)
        self._grid_o = ((self.n_origin + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1, 1)
        self._grid_e = ((self.n_emit + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1, 1)

        seed = np.uint32(0xC0FFEE ^ (self.n_origin * 2654435761 & 0xFFFFFFFF))
        with self._stream:
            self._k_init(self._grid_o, self._block,
                         (self.pos_state, np.int32(self.n_origin),
                          np.float32(self.radius), seed))
        self._stream.synchronize()

        self._interop = False
        self._gl = None
        self._res_pos = None
        self._res_col = None
        self._res_array = None
        self._fallback_pos = None
        self._fallback_col = None
        self._setup_interop()

        path = "ZERO-COPIE (driver CUDA/GL)" if self._interop else "REPLI (upload CuPy->GL)"
        print(f"[particles] {self.n_origin:,} origines + {self.n_emit:,} émises "
              f"= {self.n_total:,} (champ Lorenz caché + traînées) | {path}".replace(",", " "),
              file=sys.stderr)

    # ------------------------------------------------------- interop setup
    def _setup_interop(self):
        try:
            pos_glo = int(self._gl_pos_buffer.glo)
            col_glo = int(self._gl_col_buffer.glo)
        except Exception as exc:
            print(f"[particles] .glo introuvable ({exc}) -> repli.", file=sys.stderr)
            self._init_fallback(); return
        try:
            cp.cuda.Device(0).use()
            self._gl = _GLInterop()
            self._res_pos = self._gl.register(pos_glo)
            self._res_col = self._gl.register(col_glo)
            self._res_array = (ctypes.c_void_p * 2)(self._res_pos.value, self._res_col.value)
            self._interop = True
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
        self._interop = False
        self._fallback_pos = cp.empty(self.n_total * 4, dtype=cp.float32)
        self._fallback_col = cp.empty(self.n_total * 4, dtype=cp.float32)

    def _switch_to_fallback(self):
        if self._gl is not None:
            for res in (self._res_pos, self._res_col):
                if res is not None:
                    try: self._gl.unregister(res)
                    except Exception: pass
        self._gl = None; self._res_pos = None; self._res_col = None; self._res_array = None
        self._init_fallback()

    # --------------------------------------------- Lorenz caché (CPU) -> coeffs
    def _advance_lorenz(self, dt, amp, beat, bass):
        # Les BASSES accélèrent la dérive des coefficients -> le champ se réorganise
        # au rythme des graves (mouvement d'ensemble calé sur les basses).
        rate = _GUIDE_RATE * (1.0 + 1.0*amp + 1.5*beat + 2.0*bass)
        total = min(dt*rate, 8.0*_GUIDE_HMAX)
        nsub = max(1, min(8, int(math.ceil(total/_GUIDE_HMAX))))
        h = total/nsub
        x, y, z = self._lx, self._ly, self._lz
        for _ in range(nsub):
            dx=_LZ_SIGMA*(y-x); dy=x*(_LZ_RHO-z)-y; dz=x*y-_LZ_BETA*z
            x+=dx*h; y+=dy*h; z+=dz*h
        self._lx, self._ly, self._lz = x, y, z
        return (x*_LZ_NX, y*_LZ_NY, (z-_LZ_ZC)*_LZ_NZ)

    # ------------------------------------------------------------- update
    def update(self, dt, features):
        dt = float(dt)
        if dt <= 0.0: dt = 1.0/120.0
        dt = min(dt, 1.0/30.0)
        self._t += dt

        amp = float(getattr(features, "amplitude", 0.0)) if features is not None else 0.0
        beat = float(getattr(features, "beat", 0.0)) if features is not None else 0.0
        bass = float(getattr(features, "bass", 0.0)) if features is not None else 0.0
        mid = float(getattr(features, "mid", 0.0)) if features is not None else 0.0
        high = float(getattr(features, "high", 0.0)) if features is not None else 0.0
        centroid_attr = getattr(features, "centroid", None) if features is not None else None
        target_centroid = float(centroid_attr) if centroid_attr is not None else 0.5
        self._centroid += 0.15*(target_centroid - self._centroid)

        la, lb, lc = self._advance_lorenz(dt, amp, beat, bass)
        k = _WAVE / max(self.radius, 1e-3) * math.pi
        # Taux d'émission = CAPACITÉ du ring (n_emit/lifetime). PAS de boost audio :
        # dépasser cette capacité recyclerait des particules ENCORE VISIBLES ->
        # coupure franche. À capacité fixe, le cycle de recyclage = lifetime, donc
        # chaque particule a le temps de s'éteindre AVANT d'être réécrite.
        emit_per_sec = self.n_emit / max(self.lifetime, 1e-3)
        E = int(emit_per_sec * dt)
        E = max(0, min(E, self.n_emit))

        self._cur = dict(dt=dt, la=la, lb=lb, lc=lc, k=k, amp=amp, beat=beat, E=E,
                         head=self._emit_head, bass=bass, mid=mid, high=high)

        if self._interop:
            self._update_interop()
        else:
            self._update_fallback()

        self._emit_head = (self._emit_head + E) % self.n_emit

    def _launch(self, gl_pos, gl_col):
        """Lance les 3 kernels (origines -> émission -> émises) qui écrivent dans
        les buffers GL (origines en [0, n_origin), émises en [n_origin, n_total))."""
        c = self._cur
        f32, i32 = np.float32, np.int32
        with self._stream:
            self._k_origin(self._grid_o, self._block, (
                self.pos_state, self.vel_state, gl_pos, gl_col,
                i32(self.n_origin), f32(self._t), f32(c["dt"]), f32(self.radius),
                f32(c["la"]), f32(c["lb"]), f32(c["lc"]),
                f32(_FIELD_STRENGTH), f32(c["k"]), f32(_TURB_BASE),
                f32(c["amp"]), f32(c["beat"]), f32(self._centroid),
                f32(c["bass"]), f32(c["mid"]), f32(c["high"])))
            if c["E"] > 0:
                g_emit = ((c["E"] + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1, 1)
                self._k_emit(g_emit, self._block, (
                    self.pos_state, self.vel_state, self.emit_state,
                    i32(c["E"]), i32(self.n_origin), i32(self.n_emit), i32(c["head"])))
            self._k_emitted(self._grid_e, self._block, (
                self.emit_state, gl_pos, gl_col,
                i32(self.n_emit), i32(self.n_origin), f32(c["dt"]),
                f32(self.lifetime), f32(_EMIT_BRIGHT), f32(self._centroid),
                f32(self._t), f32(c["bass"]), f32(c["mid"]), f32(c["high"]), f32(c["beat"])))

    def _update_interop(self):
        arr = self._res_array
        stream_ptr = self._stream.ptr
        mapped = False
        try:
            self._gl.map(arr, stream_ptr); mapped = True
            pos_ptr, pos_size = self._gl.get_pointer(arr, 0)
            col_ptr, col_size = self._gl.get_pointer(arr, 1)
            gl_pos = self._wrap_device_ptr(pos_ptr, pos_size)
            gl_col = self._wrap_device_ptr(col_ptr, col_size)
            self._launch(gl_pos, gl_col)
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
            self._update_fallback()

    def _wrap_device_ptr(self, ptr, nbytes):
        mem = cp.cuda.UnownedMemory(int(ptr), int(nbytes), owner=self)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray((self.n_total * 4,), dtype=cp.float32, memptr=memptr)

    def _update_fallback(self):
        self._launch(self._fallback_pos, self._fallback_col)
        self._stream.synchronize()
        self._gl_pos_buffer.write(cp.asnumpy(self._fallback_pos))
        self._gl_col_buffer.write(cp.asnumpy(self._fallback_col))

    # ------------------------------------------------------------- release
    def release(self):
        try:
            if self._stream is not None: self._stream.synchronize()
        except Exception: pass
        if self._gl is not None:
            for res in (self._res_pos, self._res_col):
                if res is not None:
                    try: self._gl.unregister(res)
                    except Exception as exc:
                        print(f"[particles] erreur désenreg. interop : {exc}", file=sys.stderr)
        self._gl = None; self._res_pos = None; self._res_col = None; self._res_array = None
        self._interop = False
        self.pos_state = None; self.vel_state = None; self.emit_state = None
        self._fallback_pos = None; self._fallback_col = None


if __name__ == "__main__":
    print("=== particles.py (nuée + émission, Lorenz caché) : auto-vérification ===")
    print(f"CuPy disponible      : {'OUI' if _HAS_CUPY else 'NON'}")
    _drv = _GLInterop._load_driver()
    print(f"driver CUDA (interop): {'trouvé' if _drv is not None else 'absent'}")
