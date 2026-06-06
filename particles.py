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

# Valeurs de paramétrage : LUES depuis la fenêtre de config (source de vérité
# unique). Les constantes _* exposées plus bas ne codent plus de valeur en dur.
from config_window import DEFAULTS as _CFG

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

// Échantillonnage interpolé du RELIEF TONAL (tableau 1D, u in [0,1] = rayon).
__device__ __forceinline__ float relief_at(const float* relief, int nrel, float u){
    if(u<0.0f)u=0.0f; if(u>1.0f)u=1.0f;
    float fi=u*(float)(nrel-1);
    int i0=(int)fi; if(i0<0)i0=0; if(i0>nrel-2)i0=nrel-2;
    float fr=fi-(float)i0;
    return relief[i0]*(1.0f-fr)+relief[i0+1]*fr;
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
    float bass, float mid, float high, float beat, float centroid, float harm_hue, float val, float* rgb)
{
    float wsum = bass + mid + high + 0.02f;
    float spec = (mid * 0.33f + high * 0.66f) / wsum;   // 0 rouge, 0.33 vert, 0.66 bleu
    float hue = spec + (hue_id - 0.5f) * 0.30f + (local - 0.5f) * 0.20f + t * 0.01f;
    // TEMPÉRATURE DE TEINTE par le CENTROÏDE spectral (brillance timbrale) : timbre
    // brillant (centroid haut) -> teinte plus froide ; sourd (bas) -> plus chaude.
    hue += (centroid - 0.5f) * 0.12f;
    hue += harm_hue;            // HARMONIE : modalité (chaud/froid) + teinte-maison de tonalité
    float sat = fminf(0.97f, 0.75f + 0.20f * beat + 0.12f * high);
    hsv2rgb(hue, sat, val, rgb);
}

// UPDATE origines : advection champ ABC(Lorenz) + ONDES DE CHOC + PAYSAGE TONAL
//                   + MÉMORISE la vitesse (les traînées héritent du geste) + GL.
__global__ void update_origin(
    float* pos, float* vel, float* gl_pos, float* gl_col,
    const int n, const float t, const float dt, const float L,
    const float lx, const float ly, const float lz,
    const float field_strength, const float k, const float turb_base,
    const float amp, const float beat, const float centroid,
    const float bass, const float mid, const float high,
    const float* wpos, const float* wpar, const int n_waves,
    const float* relief, const int nrel,
    const float tonal_strength, const float tonal_cap, const float tonal_glow,
    const float breath, const float accel_gain, const float accel_inv_scale,
    const float build, const float drop, const float harm_hue)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float px=pos[i*3+0],py=pos[i*3+1],pz=pos[i*3+2];
    float ovx=vel[i*3+0], ovy=vel[i*3+1], ovz=vel[i*3+2];   // vitesse frame n-1 (avant écrasement)
    float kx=px*k,ky=py*k,kz=pz*k;
    float fx=lx*sinf(kz)+lz*cosf(ky);
    float fy=ly*sinf(kx)+lx*cosf(kz);
    float fz=lz*sinf(ky)+ly*cosf(kx);
    // MOUVEMENT calé sur les BASSES : les graves accélèrent le flux + la turbulence.
    float spd=field_strength*(1.0f+0.5f*amp+2.0f*bass)*(1.0f+0.8f*build);  // le build accélère le flux
    float c[3]; float ns=0.6f;
    curl_noise(px*ns+t*0.2f, py*ns, pz*ns-t*0.2f, c);
    float turb=turb_base*(1.0f+4.0f*beat+2.5f*bass);
    // Vitesse de déplacement immédiat (sert AUSSI à lancer les particules émises).
    float tvx=fx*spd + c[0]*turb;
    float tvy=fy*spd + c[1]*turb;
    float tvz=fz*spd + c[2]*turb;

    // ----- ONDES DE CHOC PERCUSSIVES : coques sphériques qui TRAVERSENT -----
    // Là où la coque gaussienne d'une onde passe (r ≈ vitesse·âge), on ajoute une
    // POUSSÉE radiale (kick), un CISAILLEMENT tangentiel (snare) et un ÉCLAT
    // (charley). Le rythme devient une météo qu'on VOIT voyager dans la matière.
    float wave_bright=0.0f;
    for(int w=0; w<n_waves; ++w){
        float st=wpar[w*6+0];               // force (déjà fondue dans le temps)
        if(st<1e-4f) continue;              // onde éteinte -> on saute
        float ex=wpos[w*3+0], ey=wpos[w*3+1], ez=wpos[w*3+2];
        float dx=px-ex, dy=py-ey, dz=pz-ez;
        float r=sqrtf(dx*dx+dy*dy+dz*dz)+1e-4f;
        float radius=wpar[w*6+1];           // rayon courant du front
        float thick=wpar[w*6+2];
        float s=(r-radius)/thick;
        float shell=__expf(-s*s);           // coque gaussienne (1 sur le front)
        float a=st*shell;
        if(a<1e-4f) continue;
        float inv=1.0f/r;
        float nx=dx*inv, ny=dy*inv, nz=dz*inv;
        float push=wpar[w*6+3];
        float curl=wpar[w*6+4];
        tvx+=nx*push*a; tvy+=ny*push*a; tvz+=nz*push*a;   // poussée radiale
        if(curl!=0.0f){                     // tournoiement (cisaillement autour de Y)
            float ttx=-nz, ttz=nx;          // = cross(n, up=(0,1,0)) horizontal
            float tl=rsqrtf(ttx*ttx+ttz*ttz+1e-8f);
            tvx+=ttx*tl*curl*a; tvz+=ttz*tl*curl*a;
        }
        wave_bright+=wpar[w*6+5]*a;         // éclat lumineux du front
    }

    // ----- PAYSAGE TONAL : relief radial stable (les notes TENUES le sculptent)
    // On remonte DOUCEMENT le gradient du relief -> striations concentriques
    // (graves au cœur, aigus en périphérie). Fenêtré + plafonné pour préserver le
    // remplissage de l'espace ; l'advection continue de brasser -> jamais figé.
    float prad=sqrtf(px*px+py*py+pz*pz)+1e-4f;
    float ir=1.0f/prad;
    float u=prad/L; if(u>1.0f)u=1.0f;
    float tonal_here=relief_at(relief, nrel, u);
    if(tonal_strength>0.0f){
        float du=2.0f/(float)nrel;
        float g=relief_at(relief,nrel,u+du)-relief_at(relief,nrel,u-du);
        float win=u*(1.0f-u)*4.0f;          // 0 aux extrêmes, 1 au milieu
        float tf=tonal_strength*g*win;
        if(tf>tonal_cap)tf=tonal_cap; else if(tf<-tonal_cap)tf=-tonal_cap;
        tvx+=px*ir*tf; tvy+=py*ir*tf; tvz+=pz*ir*tf;
    }

    // ----- RESPIRATION (pouls anticipé) : inspir AVANT le temps fort (vers le
    // cœur), expir SUR le beat (vers l'extérieur). breath<0 -> converge, >0 ->
    // s'épanouit. Transitoire et oscillant -> pas d'amas ; porté par le groove.
    tvx+=px*ir*breath; tvy+=py*ir*breath; tvz+=pz*ir*breath;

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
    // basses -> + lumineux ; + ÉCLAT des fronts d'onde ; + LUEUR des strates tonales.
    float val=0.50f+amp*0.25f+beat*0.30f+bass*0.30f
              + wave_bright*0.6f + tonal_glow*tonal_here;
    val=val*(1.0f-0.35f*build) + 0.80f*drop;   // build ASSOMBRIT (charge), drop FLASHE
    val=fminf(val, 1.6f);
    float rgb[3];
    spectral_color(hash_f((unsigned int)i), local, t, bass, mid, high, beat, centroid, harm_hue, val, rgb);
    // ÉTINCELLE DE CISAILLEMENT : |a| = Dv/Dt (différence finie vs vitesse n-1),
    // compressée par tanh -> brille aux nœuds violents du flot ET au passage des fronts.
    float ax=(tvx-ovx)/dt, ay=(tvy-ovy)/dt, az=(tvz-ovz)/dt;
    float ahat=tanhf(sqrtf(ax*ax+ay*ay+az*az)*accel_inv_scale);
    float brightness=0.55f + wave_bright*0.5f + accel_gain*ahat + 0.60f*drop;   // fronts + cisaillement + FLASH du drop
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
    const float t, const float bass, const float mid, const float high, const float beat,
    const float harm_hue)
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
        spectral_color(hash_f((unsigned int)i), local, t, bass, mid, high, beat, centroid, harm_hue, 0.75f, rgb);
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

_FIELD_STRENGTH = _CFG["_FIELD_STRENGTH"]
_WAVE = 1.2
_TURB_BASE = _CFG["_TURB_BASE"]
_EMIT_BRIGHT = 0.45        # luminosité de base d'une particule émise (fond à 0)

_LZ_SIGMA, _LZ_RHO, _LZ_BETA = 10.0, 28.0, 2.6666667
_LZ_ZC = 25.0
_LZ_NX, _LZ_NY, _LZ_NZ = 1.0/18.0, 1.0/24.0, 1.0/24.0
_GUIDE_RATE = 3.0
_GUIDE_HMAX = 0.01

# --- ONDES DE CHOC PERCUSSIVES (kit-aware) -----------------------------------
# Anneau de fronts sphériques qui TRAVERSENT la nuée à vitesse finie. Chaque
# onset (kick/snare/charley) en engendre un, dont l'ÉPICENTRE est l'état courant
# du Lorenz caché (jamais dessiné — il décide juste OÙ naît le rythme). Les
# traînées héritent du geste car on perturbe la VITESSE des origines.
_MAX_WAVES = 24                  # fronts simultanés (anneau ; le plus vieux est écrasé)
# Table par registre : (vitesse, épaisseur, poussée_radiale, cisaillement, éclat, tau_s)
#   kick    : lent, coque épaisse, forte poussée vers l'extérieur (souffle).
#   snare   : vif, coque fine, poussée modérée + fort TOURNOIEMENT (cisaillement).
#   charley : très rapide, coque ténue, peu de poussée mais SCINTILLE (éclat haut).
_WAVE_KINDS = {
    0: (5.5, 0.55, 2.2, 0.0, 0.55, 0.55),   # kick
    1: (8.5, 0.28, 1.2, 1.9, 0.40, 0.38),   # snare
    2: (12.0, 0.16, 0.45, 0.7, 0.75, 0.22),  # charley/hat
}

# --- PAYSAGE TONAL (relief radial sculpté par les notes TENUES) ---------------
# Le spectre 512 (déjà en VRAM) est lissé LENTEMENT en un relief radial stable :
# graves au cœur, aigus en périphérie. On remonte DOUCEMENT son gradient ->
# striations concentriques (la forme vient de l'harmonie) que les ondes animent.
# Couplage volontairement FAIBLE + PLAFONNÉ : on biaise la densité sans casser le
# remplissage de l'espace (jamais d'effondrement en amas).
_N_REL = 512                     # = AudioEngine.N_SPECTRUM
_TONAL_STRENGTH = _CFG["_TONAL_STRENGTH"]   # force du relief (0 = paysage tonal OFF)
_TONAL_CAP = 1.3                 # plafond de la force radiale (anti-collapse)
_TONAL_GLOW = _CFG["_TONAL_GLOW"]   # lueur des strates énergétiques (les voir briller)
_TONAL_TAU = 0.7                 # s — lissage du relief (grand = plus stable)

# --- RESPIRATION (pouls anticipé) --------------------------------------------
# La nuée INSPIRE (converge un peu vers le cœur) sur l'anticipation qui PRÉCÈDE
# le temps fort, puis EXPIRE (s'épanouit vers l'extérieur) sur le beat. Geste
# RADIAL transitoire et oscillant -> aucune accumulation nette (jamais d'amas) ;
# porté par la confiance de groove (s'efface sur la musique sans pulsation nette).
_BREATH_IN = 0.8                 # force de l'inspir (converge avant le temps fort)
_BREATH_OUT = _CFG["_BREATH_OUT"]   # force de l'expir (s'épanouit sur le beat)

# --- CINÉMATIQUE : ÉTINCELLE DE CISAILLEMENT (accélération matérielle) ---------
# |a| = Dv/Dt (différence finie de la vitesse des ORIGINES, quasi gratuite : la
# vitesse de la frame n-1 est encore dans vel[]) révèle où le flot change le plus
# VIOLEMMENT — les nœuds du champ ABC ET le passage des fronts d'onde. Normalisée
# par tanh (bornée) et AJOUTÉE à la luminosité PAR-PARTICULE (.w) -> scintillement
# local aux nœuds, SANS toucher la respiration audio (qui vit dans `val`, pas dans
# `.w`). Origines seules (les émises sont balistiques, a≈0). inv_scale calé sur la
# distribution mesurée de |a| (p50≈39, p90≈106, fronts≈250+).
_ACCEL_GAIN = _CFG["_ACCEL_GAIN"]   # intensité du scintillement (0 = OFF)
_ACCEL_INV_SCALE = 0.005         # échelle tanh de |a| (~1/p99 : bulk discret, fronts saturent)

# --- PHRASE : build (charge) + drop (relâche viscérale) -----------------------
# Le build RAMÈNE doucement la nuée vers le cœur, accélère le flux et l'assombrit
# (le calme avant la tempête) ; le DROP la DÉTONE — bloom radial massif + onde de
# choc CENTRALE + flash. Porté par groove_conf côté audio (pas de drop sur de
# l'arythmique). Réglables (l'utilisateur a demandé « viscéral »).
_BUILD_CONVERGE = _CFG["_BUILD_CONVERGE"]   # attraction vers le cœur pendant le build (charge)
_DROP_BLOOM = _CFG["_DROP_BLOOM"]           # épanouissement radial violent au drop (relâche)

# --- HARMONIE : teinte GLOBALE de la palette (modalité + tonalité) ------------
# tonal_warmth (−1 mineur … +1 majeur) décale la TEMPÉRATURE : mineur -> froid
# (teinte +), majeur -> chaud (teinte −) ; key_hue donne une teinte-maison par
# tonalité. Lent (l'harmonie change sur des mesures). Réglables.
_WARMTH_HUE = _CFG["_WARMTH_HUE"]       # ampleur du décalage chaud/froid selon la modalité
_KEY_HUE_SPAN = _CFG["_KEY_HUE_SPAN"]   # étalement de teinte selon la tonalité (subtil)


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
        # 'shape' est RÉSERVÉ : la nuée est toujours un remplissage de boîte advecté
        # (cf. init_field) ; l'ancien sélecteur de forme (_SHAPE_MODES) a été retiré.
        self.shape = shape
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

        # --- ONDES DE CHOC : anneau de fronts (état CPU minuscule + staging GPU) -
        # On gère l'évolution temporelle (âge, fondu, rayon) sur CPU — 24 fronts,
        # négligeable — puis on uploade deux petits tableaux par frame.
        self._wave_pos = np.zeros((_MAX_WAVES, 3), dtype=np.float32)
        self._wave_age = np.full(_MAX_WAVES, 1e9, dtype=np.float32)  # grand => éteinte
        self._wave_str0 = np.zeros(_MAX_WAVES, dtype=np.float32)
        self._wave_speed = np.zeros(_MAX_WAVES, dtype=np.float32)
        self._wave_thick = np.ones(_MAX_WAVES, dtype=np.float32)
        self._wave_push = np.zeros(_MAX_WAVES, dtype=np.float32)
        self._wave_curl = np.zeros(_MAX_WAVES, dtype=np.float32)
        self._wave_brt = np.zeros(_MAX_WAVES, dtype=np.float32)
        self._wave_tau = np.ones(_MAX_WAVES, dtype=np.float32)
        self._wave_head = 0
        # Anti-doublon d'onset : l'instantané audio peut être RÉ-UTILISÉ sur
        # plusieurs frames de rendu (ré-analyse tous les ~blocksize/2 échantillons).
        # On ne déclenche les ondes que sur un instantané NEUF (samples_written a
        # changé) -> un onset = un seul front, quel que soit le framerate.
        self._last_samples = -1
        self._prev_drop = 0.0        # front montant du drop -> onde de choc centrale
        self._wpar_cpu = np.zeros(_MAX_WAVES * 6, dtype=np.float32)  # staging CPU
        self._wpos_gpu = cp.zeros(_MAX_WAVES * 3, dtype=f32)         # épicentres (VRAM)
        self._wpar_gpu = cp.zeros(_MAX_WAVES * 6, dtype=f32)         # paramètres (VRAM)

        # --- PAYSAGE TONAL : relief radial lissé, vit en VRAM (lu par le kernel) -
        self._relief_gpu = cp.zeros(_N_REL, dtype=f32)

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
        # Le REPLI alloue 2 buffers CuPy de plus (n_total*4 floats chacun) en VRAM.
        # Sur une carte serrée où le zéro-copie a échoué, ça peut manquer de mémoire
        # -> on échoue PROPREMENT (message actionnable) plutôt qu'avec un OOM brut.
        self._interop = False
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

    # --------------------------------------------- Lorenz caché (CPU) -> coeffs
    def _advance_lorenz(self, dt, amp, beat, bass):
        # Les BASSES accélèrent la dérive des coefficients -> le champ se réorganise
        # au rythme des graves (mouvement d'ensemble calé sur les basses).
        rate = _GUIDE_RATE * (1.0 + 1.0*amp + 1.5*beat + 2.0*bass)
        total = min(dt*rate, 8.0*_GUIDE_HMAX)
        nsub = max(1, min(8, int(math.ceil(total/_GUIDE_HMAX))))
        h = total/nsub
        x, y, z = self._lx, self._ly, self._lz
        # RK4 (scalaire, CPU) : robuste au pas — le pas est audio-variable, donc on
        # veut une trajectoire quasi insensible au découpage, sans la dérive d'énergie
        # qu'Euler explicite injecte sur Lorenz. Coût négligeable (1 appel/frame, ≤8 sous-pas).
        def f(x, y, z):
            return (_LZ_SIGMA*(y-x), x*(_LZ_RHO-z)-y, x*y-_LZ_BETA*z)
        for _ in range(nsub):
            k1 = f(x, y, z)
            k2 = f(x+0.5*h*k1[0], y+0.5*h*k1[1], z+0.5*h*k1[2])
            k3 = f(x+0.5*h*k2[0], y+0.5*h*k2[1], z+0.5*h*k2[2])
            k4 = f(x+h*k3[0],     y+h*k3[1],     z+h*k3[2])
            x += (h/6.0)*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
            y += (h/6.0)*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
            z += (h/6.0)*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])
        self._lx, self._ly, self._lz = x, y, z
        return (x*_LZ_NX, y*_LZ_NY, (z-_LZ_ZC)*_LZ_NZ)

    # ----------------------------------------------- ondes de choc / relief tonal
    def _spawn_drop_wave(self):
        """DROP : une onde de choc CENTRALE géante (depuis le cœur), bien plus forte
        et épaisse qu'un onset normal -> un mur de lumière qui balaie toute la boîte."""
        slot = self._wave_head
        self._wave_head = (slot + 1) % _MAX_WAVES
        self._wave_pos[slot, 0] = 0.0
        self._wave_pos[slot, 1] = 0.0
        self._wave_pos[slot, 2] = 0.0
        self._wave_age[slot] = 0.0
        self._wave_str0[slot] = 1.6
        self._wave_speed[slot] = 7.0          # traverse toute la boîte (~0.8 s)
        self._wave_thick[slot] = 0.9          # coque épaisse = mur de choc
        self._wave_push[slot] = 3.5           # poussée massive vers l'extérieur
        self._wave_curl[slot] = 0.0
        self._wave_brt[slot] = 1.0            # éclat fort
        self._wave_tau[slot] = 0.8

    def _spawn_wave(self, kind, strength, la, lb, lc):
        """Allume un nouveau front (anneau : écrase le plus ancien). L'épicentre
        est l'état du Lorenz CACHÉ (la,lb,lc, ~[-1,1]) projeté dans la boîte, avec
        un léger décalage déterministe pour que les fronts ne naissent pas tous au
        même point. `kind` : 0=kick, 1=snare, 2=charley."""
        slot = self._wave_head
        self._wave_head = (slot + 1) % _MAX_WAVES
        L = self.radius
        # Décalage déterministe (pas de RNG) variant avec le slot ET le temps.
        b = slot * 1.7 + self._t * 3.1
        jx = math.sin(b * 1.70) * 0.22 * L
        jy = math.sin(b * 2.30 + 1.1) * 0.22 * L
        jz = math.sin(b * 1.30 + 2.7) * 0.22 * L
        self._wave_pos[slot, 0] = min(L, max(-L, la * L + jx))
        self._wave_pos[slot, 1] = min(L, max(-L, lb * L + jy))
        self._wave_pos[slot, 2] = min(L, max(-L, lc * L + jz))
        self._wave_age[slot] = 0.0
        # La force du front suit l'intensité de l'onset (bornée).
        self._wave_str0[slot] = float(min(1.5, max(0.0, strength))) + 0.15
        speed, thick, push, curl, brt, tau = _WAVE_KINDS.get(int(kind), _WAVE_KINDS[0])
        self._wave_speed[slot] = speed
        self._wave_thick[slot] = thick
        self._wave_push[slot] = push
        self._wave_curl[slot] = curl
        self._wave_brt[slot] = brt
        self._wave_tau[slot] = tau

    def _advance_waves(self, dt):
        """Fait vieillir les fronts (rayon = vitesse·âge ; fondu exp(-âge/tau)) et
        prépare les deux petits tableaux GPU lus par le kernel. Un front qui a
        quitté la boîte est éteint (force 0) -> le kernel le saute."""
        self._wave_age += dt
        radius = self._wave_speed * self._wave_age
        fade = np.exp(-self._wave_age / np.maximum(self._wave_tau, 1e-3))
        str_faded = self._wave_str0 * fade
        # Au-delà de ~2.2·L le front a traversé toute la boîte -> on l'éteint.
        str_faded[radius > (2.2 * self.radius)] = 0.0
        par = self._wpar_cpu.reshape(_MAX_WAVES, 6)
        par[:, 0] = str_faded
        par[:, 1] = radius
        par[:, 2] = self._wave_thick
        par[:, 3] = self._wave_push
        par[:, 4] = self._wave_curl
        par[:, 5] = self._wave_brt
        # Upload (minuscule : 24×6 + 24×3 floats).
        self._wpar_gpu.set(self._wpar_cpu)
        self._wpos_gpu.set(self._wave_pos.reshape(-1))

    def _update_relief(self, dt, features):
        """Lisse LENTEMENT le spectre 512 (déjà en VRAM) en un relief radial
        stable. EMA de constante _TONAL_TAU : assez lent pour que seules les notes
        TENUES sculptent le relief (le rythme, lui, l'ANIME via les ondes)."""
        spec = getattr(features, "spectrum_gpu", None) if features is not None else None
        if spec is not None and spec.shape[0] == _N_REL:
            a = 1.0 - math.exp(-dt / _TONAL_TAU)
            self._relief_gpu *= np.float32(1.0 - a)
            self._relief_gpu += np.float32(a) * spec
        else:
            self._relief_gpu *= np.float32(0.98)   # plus de spectre -> s'efface

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

        # --- ONDES DE CHOC : chaque onset (kick/snare/charley) engendre un front,
        #     dont l'ÉPICENTRE est l'état courant du Lorenz CACHÉ (la,lb,lc). Le
        #     contrat audio est lu défensivement (getattr) -> tourne même sans ces
        #     champs (vieux moteur audio) ou sans audio du tout.
        sw = getattr(features, "samples_written", None) if features is not None else None
        fresh = (sw is None) or (sw != self._last_samples)  # instantané audio NEUF ?
        self._last_samples = sw
        if features is not None and fresh:
            if getattr(features, "kick_hit", False):
                self._spawn_wave(0, float(getattr(features, "kick", 0.0)), la, lb, lc)
            if getattr(features, "snare_hit", False):
                self._spawn_wave(1, float(getattr(features, "snare", 0.0)), la, lb, lc)
            if getattr(features, "hat_hit", False):
                self._spawn_wave(2, float(getattr(features, "hat", 0.0)), la, lb, lc)
        self._advance_waves(dt)          # âge/fondu/rayon des fronts -> staging GPU
        self._update_relief(dt, features)  # relief tonal lissé (EMA lente, en VRAM)

        k = _WAVE / max(self.radius, 1e-3) * math.pi
        # Taux d'émission = CAPACITÉ du ring (n_emit/lifetime). PAS de boost audio :
        # dépasser cette capacité recyclerait des particules ENCORE VISIBLES ->
        # coupure franche. À capacité fixe, le cycle de recyclage = lifetime, donc
        # chaque particule a le temps de s'éteindre AVANT d'être réécrite.
        emit_per_sec = self.n_emit / max(self.lifetime, 1e-3)
        E = int(emit_per_sec * dt)
        E = max(0, min(E, self.n_emit))

        # RESPIRATION : expir (vers l'extérieur) sur le beat, inspir (vers le cœur)
        # sur l'anticipation. anticipation est DÉJÀ pondérée par la confiance côté
        # audio ; on porte aussi l'expir par la confiance -> rien ne « pompe » sur
        # une musique sans pulsation nette (groove_conf -> 0).
        conf = float(getattr(features, "groove_conf", 0.0)) if features is not None else 0.0
        antic = float(getattr(features, "anticipation", 0.0)) if features is not None else 0.0
        build = float(getattr(features, "build", 0.0)) if features is not None else 0.0
        drop = float(getattr(features, "drop", 0.0)) if features is not None else 0.0
        warmth = float(getattr(features, "tonal_warmth", 0.0)) if features is not None else 0.0
        key_hue = float(getattr(features, "key_hue", 0.0)) if features is not None else 0.0
        # HARMONIE -> décalage de teinte GLOBAL : modalité (mineur -> froid/teinte+,
        # majeur -> chaud/teinte−) + teinte-maison de la tonalité. Un seul scalaire
        # passé aux deux kernels (couleur lente, sur des mesures).
        harm_hue = -_WARMTH_HUE * warmth + _KEY_HUE_SPAN * key_hue
        # Respiration par-battement + PHRASE : le build RAMÈNE vers le cœur (charge),
        # le drop ÉPANOUIT violemment (relâche) ET engendre une onde de choc CENTRALE
        # au front montant -> impact viscéral.
        breath = (_BREATH_OUT * beat * conf - _BREATH_IN * antic
                  + _DROP_BLOOM * drop - _BUILD_CONVERGE * build)
        if drop > 0.5 and self._prev_drop <= 0.5:
            self._spawn_drop_wave()
        self._prev_drop = drop

        self._cur = dict(dt=dt, la=la, lb=lb, lc=lc, k=k, amp=amp, beat=beat, E=E,
                         head=self._emit_head, bass=bass, mid=mid, high=high,
                         breath=breath, build=build, drop=drop, harm_hue=harm_hue)

        if self._interop:
            self._update_interop()
        else:
            self._update_fallback()

        if self.n_emit:                  # EMIT_RATE=0 -> aucune émise -> pas de modulo par 0
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
                f32(c["bass"]), f32(c["mid"]), f32(c["high"]),
                self._wpos_gpu, self._wpar_gpu, i32(_MAX_WAVES),
                self._relief_gpu, i32(_N_REL),
                f32(_TONAL_STRENGTH), f32(_TONAL_CAP), f32(_TONAL_GLOW),
                f32(c["breath"]), f32(_ACCEL_GAIN), f32(_ACCEL_INV_SCALE),
                f32(c["build"]), f32(c["drop"]), f32(c["harm_hue"])))
            if c["E"] > 0:
                g_emit = ((c["E"] + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1, 1)
                self._k_emit(g_emit, self._block, (
                    self.pos_state, self.vel_state, self.emit_state,
                    i32(c["E"]), i32(self.n_origin), i32(self.n_emit), i32(c["head"])))
            if self.n_emit > 0:          # grille _grid_e=(0,..) = lancement CUDA invalide
                self._k_emitted(self._grid_e, self._block, (
                    self.emit_state, gl_pos, gl_col,
                    i32(self.n_emit), i32(self.n_origin), f32(c["dt"]),
                    f32(self.lifetime), f32(_EMIT_BRIGHT), f32(self._centroid),
                    f32(self._t), f32(c["bass"]), f32(c["mid"]), f32(c["high"]), f32(c["beat"]),
                    f32(c["harm_hue"])))

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
    def prefill_emitted(self):
        """Pré-remplit le ring des traînées avec un ÂGE uniformément réparti sur
        [0, lifetime) pour que leur densité soit à son RÉGIME dès la 1re frame —
        sinon elle met EMITTED_LIFETIME secondes à se peupler (18 s en Ambiant,
        24 s en Cosmique), d'où le « chargement » visible à l'ouverture.

        Modèle balistique (pos = origine + vitesse·âge) + fondu (1-u)^2 : les jeunes
        (vives, près des origines) remplissent la boîte, les vieilles (déjà éteintes
        par le fondu) sont invisibles -> remplissage SANS couture. À appeler APRÈS
        >=1 update() (les origines doivent déjà avoir une vitesse). One-shot."""
        if self.n_emit <= 0:
            return
        try:
            with self._stream:
                es = self.emit_state.reshape(self.n_emit, 7)
                opos = self.pos_state.reshape(self.n_origin, 3)
                ovel = self.vel_state.reshape(self.n_origin, 3)
                # Même balayage des origines que le kernel d'émission : slot % n_origin.
                src = cp.arange(self.n_emit, dtype=cp.int32) % self.n_origin
                ages = cp.random.random(self.n_emit, dtype=cp.float32) * np.float32(self.lifetime)
                es[:, 0:3] = opos[src] + ovel[src] * ages[:, None]   # position balistique à cet âge
                es[:, 3:6] = ovel[src]                                # vitesse héritée de l'origine
                es[:, 6] = ages                                       # âge réparti -> densité de régime
            self._stream.synchronize()
        except Exception as exc:
            print(f"[particles] pré-remplissage des traînées ignoré ({exc}).",
                  file=sys.stderr)

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
