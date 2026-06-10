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
Interop CUDA<->GL (zéro-copie + repli upload) : module dédié gl_interop.py — ce
fichier ne s'occupe plus QUE de la simulation + du lancement des kernels.
"""

from __future__ import annotations

import os
import sys
import math

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
# Interop zéro-copie CUDA<->GL (map/unmap, repli upload) -> module dédié.
from gl_interop import GLInteropBuffers


# ===========================================================================
#  Source CUDA C
# ===========================================================================
def _load_cuda_source():
    # Lit le source CUDA C du kernel depuis cuda/particles.cu, resolu
    # relativement a CE fichier -> robuste au CWD ET au build PyInstaller onefile
    # (le dossier cuda/ est embarque via le .spec, comme les shaders GLSL).
    # Compile par NVRTC a l'init du ParticleSystem (cp.RawModule).
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "cuda", "particles.cu"), "r", encoding="utf-8") as f:
        return f.read()


_CUDA_SOURCE = _load_cuda_source()


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

# --- MODE COGNITIF (perceptuel) ----------------------------------------------
# La nuée ne réagit QU'À CE QUE L'OREILLE PERÇOIT : la SONIE pondérée d'isosonie +
# compressée en loudness (champs `loudness`/`p_bass`/`p_mid`/`p_high` calculés côté
# audio_engine), au lieu de l'énergie physique brute. _PERCEPTUAL mélange 0..1 :
#   0 -> comportement d'origine, STRICTEMENT inchangé (énergie physique) ;
#   1 -> sonie perçue pure (preset « Cognitive »).
# On ne TOUCHE PAS au kernel CUDA : on substitue seulement ce qu'on lui DONNE
# (amp/bass/mid/high). On met aussi en avant l'ANTICIPATION du rythme (la nuée
# inspire plus fort AVANT le temps fort prédit — le « système d'anticipation »).
_PERCEPTUAL = _CFG["_PERCEPTUAL"]   # 0 = OFF (physique) … 1 = sonie perçue (cognitif)
_PERCEPT_ANTIC = 1.2                # surpoids de l'inspir d'anticipation en cognitif

# --- ONDES GRAVITATIONNELLES (basse PROFONDE -> ondulation globale de la matière) --
# Un front radial qui VOYAGE en continu à travers toute la nuée, piloté par le SUB
# (basse profonde « ressentie », champ audio `sub`). C'est la force viscérale du grave
# rendue VISIBLE : on voit les ondes parcourir la matière. _GRAV_WAVE = force globale
# (0 = OFF, défaut -> rien ne change pour les autres presets). Plus la basse est deep
# (sub soutenu), plus la longueur d'onde S'ALLONGE (grav_k diminue) -> toute la matière
# bouge ENSEMBLE : l'impact devient GÉNÉRAL. La phase est repliée mod 2π (précision +
# pas de saut visible : sin est 2π-périodique). NON soumis à l'isosonie : le grave qui
# s'ENTEND faible se RESSENT fort.
_GRAV_WAVE = _CFG.get("_GRAV_WAVE", 0.0)   # force des ondes gravitationnelles (0 = OFF)
_GRAV_OMEGA = 4.2          # vitesse de propagation des fronts (rad/s de phase)
_GRAV_K0 = 8.5             # radians de phase sur le rayon de la boîte (basse peu deep)
_GRAV_REACH = 1.4          # allonge la longueur d'onde quand le grave est deep (-> général)

# --- TUNNEL HYPERSPACE (vol infini le long de l'axe Z) -----------------------
# La nuée FONCE le long de l'axe (dérive axiale) et se met en forme de PAROI
# cylindrique autour d'un AXE COURBE : le tube SERPENTE, ses virages étant l'état
# LISSÉ du Lorenz caché (l'attracteur tient le manche — jamais à l'écran, fidèle à
# l'esprit du projet). L'enroulement périodique en Z rend le tunnel INFINI — et il
# s'applique AUSSI aux traînées (cf. update_emitted) : toutes les stries restent
# dans le tube. Le rythme y devient des ANNEAUX axiaux (kick gonfle la paroi,
# snare la tord, hat scintille ; le drop lance un MUR de lumière depuis le fond).
# Tout est gated par _TUNNEL (0 = OFF, défaut -> rien ne change pour les autres
# presets). La vitesse s'emballe sur les basses/loudness/drop (hyperspeed), le
# rayon se resserre sur l'anticipation/build (le tube aspire avant le temps fort)
# et s'ouvre sur le drop (le saut en lumière — qui REDRESSE aussi le tunnel).
# _TUNNEL_WALL : 0 = volumétrique (cœur pas vide) … grand = parois nettes + cœur
# creux ; la paroi se DENSIFIE avec le groove et le build. _TUNNEL_TWIST : gain de
# la vrille (signée par le Lorenz lissé -> elle s'inverse organiquement).
# Cf. caméra mode "tunnel" (renderer.update_camera + tunnel_axis_at ci-dessous).
_TUNNEL = _CFG.get("_TUNNEL", 0.0)            # 0 = OFF … >0 = mode tunnel (intensité)
_TUNNEL_WALL = _CFG.get("_TUNNEL_WALL", 0.0)  # netteté des parois (0 = flou … grand = net)
_TUNNEL_TWIST = _CFG.get("_TUNNEL_TWIST", 1.0)  # gain de la vrille (0 = tube sans rotation)
_TUNNEL_SPEED = 9.0      # vitesse de base de la dérive axiale (× modulation audio)
_TUNNEL_RADIUS = 0.50    # rayon-cible de la paroi (× rayon de la boîte)
_TUNNEL_CURVE = (0.30, 0.22)   # amplitude du serpent (× rayon) : X (1 période/boîte), Y (2)
_TUNNEL_PH_DRIFT = (0.05, 0.08)  # rad/s — dérive lente des phases des virages
_TUNNEL_LZ_TAU = 6.0     # s — lissage du Lorenz qui conduit virages & vrille

# --- MANDELBULB (fractale 3D VIVANTE, faite de particules) --------------------
# La nuée se CONDENSE sur la surface d'un Mandelbulb (formule triplex z -> z^n+c)
# évalué PAR PARTICULE dans le kernel (champ de distance + orbit trap). Le flot
# ABC glisse LE LONG de la surface (composante normale effacée près de la
# coquille) -> la fractale est un courant, pas une statue. La puissance n MUTE
# avec la musique : dérive lente portée par l'énergie, PRÉCIPITÉE au drop -> la
# forme se restructure sous les yeux. Couleur = orbit trap (chaque lobe sa
# teinte) ; ondes de choc/respiration/build-drop existants jouent sur la coquille.
# Tout est gated par _BULB (0 = OFF, défaut -> rien ne change pour les autres
# presets). Cf. preset « Mandelbulb » (config_window) et bulb_de (particles.cu).
_BULB = _CFG.get("_BULB", 0.0)              # 0 = OFF … >0 = force de condensation
_BULB_POWER = _CFG.get("_BULB_POWER", 8.0)  # puissance n de base (8 = bulbe classique)
_BULB_MORPH = _CFG.get("_BULB_MORPH", 1.0)  # amplitude de la métamorphose musicale de n
_BULB_SCALE = 0.56       # taille du bulbe (× rayon de la boîte ; surface à ~1.2×scale)
_BULB_MORPH_SPAN = 2.5   # excursion max de n autour de la base (× _BULB_MORPH)
_BULB_RESEED_TAU = 7.0   # s — durée de vie moyenne avant ré-ensemencement (pluie)


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
        self._t = 0.0
        self._centroid = 0.5
        self._grav_phase = 0.0   # phase des ondes gravitationnelles (repliée mod 2π)
        self._grav_depth = 0.0   # profondeur du grave lissée (allonge la longueur d'onde)
        # TUNNEL : état du serpent (Lorenz lissé -> amplitudes des virages + sens de
        # la vrille) et phases des courbes. _tun_axis = (ampx, phx, ampy, phy)
        # EFFECTIFS de la frame courante, lus par la caméra via tunnel_axis_at().
        self._tun_sx = 0.0; self._tun_sy = 0.0; self._tun_sw = 0.0
        self._tun_phx = 0.0; self._tun_phy = 2.1
        self._tun_axis = (0.0, 0.0, 0.0, 0.0)
        # MANDELBULB : phase de la métamorphose de la puissance n (repliée mod 2π).
        self._bulb_phase = 0.0
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
        self._k_prefill = self._module.get_function("prefill_emitted")

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

        # Interop CUDA<->GL (zéro-copie + repli transparent) -> module dédié. On lui
        # confie les deux buffers GL du Renderer + le stream ; il gère map/unmap/upload
        # AUTOUR du lancement des kernels (cf. update -> self._interop.run(self._launch)).
        self._interop = GLInteropBuffers(gl_pos_buffer, gl_col_buffer,
                                         self.n_total, self._stream)

        print(f"[particles] {self.n_origin:,} origines + {self.n_emit:,} émises "
              f"= {self.n_total:,} (champ Lorenz caché + traînées) | "
              f"{self._interop.path_label}".replace(",", " "),
              file=sys.stderr)

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
        et épaisse qu'un onset normal -> un mur de lumière qui balaie toute la boîte.
        EN TUNNEL : le mur naît au FOND du tube (z=-L) et balaie TOUT le tunnel
        jusqu'à la caméra — le saut en lumière vient de l'horizon et vous traverse."""
        slot = self._wave_head
        self._wave_head = (slot + 1) % _MAX_WAVES
        self._wave_pos[slot, 0] = 0.0
        self._wave_pos[slot, 1] = 0.0
        self._wave_pos[slot, 2] = -self.radius if _TUNNEL > 0.0 else 0.0
        self._wave_age[slot] = 0.0
        self._wave_str0[slot] = 1.6
        self._wave_speed[slot] = 10.0 if _TUNNEL > 0.0 else 7.0  # traverse toute la boîte
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
        if _TUNNEL > 0.0:
            # En tunnel les fronts sont des ANNEAUX axiaux : plus VITES (ils doivent
            # remonter le tube jusqu'à la caméra) et un peu plus tenaces.
            speed *= 1.5
            tau *= 1.25
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
        # Plafond du pas : garde-fou contre les HOQUETS (drag de fenêtre, stall de
        # l'encodeur...) — un pas géant téléporterait la matière et viderait le ring
        # d'émission d'un coup. 1/20 s et PAS 1/30 : la 3D frame packing cadence à
        # 24 Hz (dt ≈ 41,7 ms) ; à 1/30 chaque frame stéréo était écrêtée et la
        # simulation tournait à ~80 % du temps réel. À 50 ms, le pas reste très en
        # deçà des limites de stabilité (le terme le plus raide, le rappel de paroi
        # du tunnel, tolère ~1 s) ; le Lorenz, lui, sous-découpe déjà (_GUIDE_HMAX).
        dt = min(dt, 1.0/20.0)
        self._t += dt

        amp = float(getattr(features, "amplitude", 0.0)) if features is not None else 0.0
        beat = float(getattr(features, "beat", 0.0)) if features is not None else 0.0
        bass = float(getattr(features, "bass", 0.0)) if features is not None else 0.0
        mid = float(getattr(features, "mid", 0.0)) if features is not None else 0.0
        high = float(getattr(features, "high", 0.0)) if features is not None else 0.0
        # BASSE PROFONDE ressentie (force des ondes gravitationnelles) : RAW, jamais
        # pondérée par l'isosonie -> le grave qu'on entend faible se RESSENT fort.
        sub = float(getattr(features, "sub", 0.0)) if features is not None else 0.0
        centroid_attr = getattr(features, "centroid", None) if features is not None else None
        target_centroid = float(centroid_attr) if centroid_attr is not None else 0.5
        self._centroid += 0.15*(target_centroid - self._centroid)

        # --- MODE COGNITIF : remplace l'énergie PHYSIQUE par la SONIE PERÇUE.
        # On mélange (_PERCEPTUAL) amp/bass/mid/high vers loudness/p_bass/p_mid/p_high
        # (pondérées d'isosonie + compressées côté audio_engine) AVANT toute la suite :
        # le champ Lorenz, le flux, les couleurs et la luminosité réagissent alors à ce
        # que l'humain ENTEND (le sub inaudible n'emballe plus tout ; la zone de présence
        # ~2–5 kHz prend le dessus). On NE touche PAS aux onsets/beat (transitoires =
        # déjà perceptuels). _PERCEPTUAL=0 -> ces lignes sont des no-op (amp+0·… = amp).
        if _PERCEPTUAL > 0.0 and features is not None:
            P = _PERCEPTUAL
            amp  = amp  + P * (float(getattr(features, "loudness", amp)) - amp)
            bass = bass + P * (float(getattr(features, "p_bass", bass)) - bass)
            mid  = mid  + P * (float(getattr(features, "p_mid",  mid))  - mid)
            high = high + P * (float(getattr(features, "p_high", high)) - high)

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
                # Kick PROFOND -> front plus fort (impact « avec force ») : la force de
                # l'onde est dopée par le sub ressenti.
                self._spawn_wave(0, float(getattr(features, "kick", 0.0)) * (1.0 + 1.5 * sub),
                                 la, lb, lc)
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
        # En mode COGNITIF, l'INSPIR d'anticipation est surpondéré (×(1+_PERCEPT_ANTIC·P))
        # -> le « système d'anticipation » devient le geste central : la nuée se ramasse
        # nettement AVANT le temps fort prédit, puis expire dessus. P=0 -> facteur 1 (inchangé).
        breath = (_BREATH_OUT * beat * conf
                  - _BREATH_IN * (1.0 + _PERCEPT_ANTIC * _PERCEPTUAL) * antic
                  + _DROP_BLOOM * drop - _BUILD_CONVERGE * build)
        if drop > 0.5 and self._prev_drop <= 0.5:
            self._spawn_drop_wave()
        self._prev_drop = drop

        # ONDES GRAVITATIONNELLES : la phase AVANCE (fronts qui voyagent), repliée mod 2π
        # (précision sur de longs runs + AUCUN saut visible : sin est 2π-périodique).
        # Amplitude ∝ sub ; la longueur d'onde s'allonge avec la PROFONDEUR lissée du
        # grave (grav_k plus petit) -> sur les basses deep, toute la matière ondule
        # ENSEMBLE (impact GÉNÉRAL), au lieu de petites rides locales.
        self._grav_depth += 0.04 * (sub - self._grav_depth)
        self._grav_phase = (self._grav_phase + _GRAV_OMEGA * dt) % (2.0 * math.pi)
        grav_amp = _GRAV_WAVE * sub
        grav_k = (_GRAV_K0 / max(self.radius, 1e-3)) / (1.0 + _GRAV_REACH * self._grav_depth)

        # TUNNEL : dérive axiale (HYPERSPEED sur bass/loudness/drop) + rayon de paroi
        # qui se RESSERRE sur l'anticipation/build (le tube aspire avant le temps fort)
        # et s'OUVRE sur le drop (le saut en lumière). En tunnel, le radial 3D (la
        # respiration) est REMPLACÉ par la mise en forme cylindrique -> breath = 0.
        if _TUNNEL > 0.0:
            loud = float(getattr(features, "loudness", amp)) if features is not None else 0.0
            tunnel_speed = _TUNNEL * _TUNNEL_SPEED * (1.0 + 1.5 * bass + 0.6 * loud + 3.0 * drop)
            tunnel_radius = max(0.05 * self.radius,
                                _TUNNEL_RADIUS * self.radius
                                * (1.0 - 0.25 * build - 0.30 * antic + 0.35 * drop))
            # PAROI VIVANTE : elle se densifie quand le groove se verrouille et se
            # tend pendant le build (le tube se cristallise avant le saut) ; sans
            # pulsation nette elle redevient brume volumétrique.
            tunnel_wall = _TUNNEL_WALL * (0.75 + 0.50 * conf + 0.90 * build)
            # LE SERPENT : l'état du Lorenz caché, LISSÉ (τ≈6 s), donne les
            # amplitudes des virages (x: 1 période/boîte, y: 2) et le SENS de la
            # vrille -> le tunnel méandre et tournoie sans jamais se répéter,
            # conduit par l'attracteur. Le drop REDRESSE le tube (le saut en
            # lumière file droit). Phases en dérive lente, repliées mod 2π.
            a_lz = 1.0 - math.exp(-dt / _TUNNEL_LZ_TAU)
            self._tun_sx += a_lz * (la - self._tun_sx)
            self._tun_sy += a_lz * (lb - self._tun_sy)
            self._tun_sw += a_lz * (lc - self._tun_sw)
            self._tun_phx = (self._tun_phx + _TUNNEL_PH_DRIFT[0] * dt) % (2.0 * math.pi)
            self._tun_phy = (self._tun_phy + _TUNNEL_PH_DRIFT[1] * dt) % (2.0 * math.pi)
            straight = 1.0 - 0.55 * drop
            self._tun_axis = (_TUNNEL_CURVE[0] * self.radius * self._tun_sx * straight,
                              self._tun_phx,
                              _TUNNEL_CURVE[1] * self.radius * self._tun_sy * straight,
                              self._tun_phy)
            # VRILLE signée : sens et intensité portés par le Lorenz lissé (elle
            # s'inverse organiquement), dopée par le build, gain utilisateur.
            sw = self._tun_sw
            tunnel_swirl = (_TUNNEL_TWIST * math.copysign(0.04 + 0.10 * abs(sw), sw)
                            * (1.0 + 0.6 * build))
            breath = 0.0
        else:
            tunnel_speed = tunnel_radius = tunnel_wall = tunnel_swirl = 0.0
            self._tun_axis = (0.0, 0.0, 0.0, 0.0)

        # MANDELBULB : la puissance n MUTE avec la musique — dérive lente dont la
        # vitesse suit l'ÉNERGIE (amp), et le DROP la précipite (la fractale se
        # restructure sous les yeux, ~+1,5 rad pendant la relâche). En silence la
        # forme évolue à peine (période ~100 s) : une statue qui respire. Phase
        # repliée mod 2π ; n borné >= 2 (en deçà le DE dégénère).
        if _BULB > 0.0:
            self._bulb_phase = (self._bulb_phase
                                + dt * (0.06 + 0.12 * amp + 1.5 * drop)) % (2.0 * math.pi)
            bulb_power = max(2.0, _BULB_POWER
                             + _BULB_MORPH_SPAN * _BULB_MORPH * math.sin(self._bulb_phase))
            bulb_scale = _BULB_SCALE * self.radius
            bulb_reseed = min(0.25, dt / _BULB_RESEED_TAU)   # pluie d'accrétion (cf. kernel)
        else:
            bulb_power = 8.0
            bulb_scale = 1.0        # placeholder sûr (kernel gated par bulb>0)
            bulb_reseed = 0.0

        self._cur = dict(dt=dt, la=la, lb=lb, lc=lc, k=k, amp=amp, beat=beat, E=E,
                         head=self._emit_head, bass=bass, mid=mid, high=high,
                         breath=breath, build=build, drop=drop, harm_hue=harm_hue,
                         grav_amp=grav_amp, grav_k=grav_k, grav_phase=self._grav_phase,
                         tunnel=_TUNNEL, tunnel_speed=tunnel_speed,
                         tunnel_radius=tunnel_radius, tunnel_wall=tunnel_wall,
                         tunnel_ax=self._tun_axis[0], tunnel_phx=self._tun_axis[1],
                         tunnel_ay=self._tun_axis[2], tunnel_phy=self._tun_axis[3],
                         tunnel_swirl=tunnel_swirl,
                         bulb=_BULB, bulb_scale=bulb_scale, bulb_power=bulb_power,
                         bulb_reseed=bulb_reseed)

        # Interop : map/unmap zéro-copie (ou upload de repli) AUTOUR du lancement des
        # kernels. _launch écrit dans les deux buffers GL fournis ; le repli sur erreur
        # d'interop est transparent (géré dans gl_interop.GLInteropBuffers.run).
        self._interop.run(self._launch)

        if self.n_emit:                  # EMIT_RATE=0 -> aucune émise -> pas de modulo par 0
            self._emit_head = (self._emit_head + E) % self.n_emit

    def tunnel_axis_at(self, z):
        """(x, y) de l'AXE COURBE du tunnel à la cote z (unités monde) — la ligne
        que la caméra suit (cf. renderer._tunnel_camera, câblé par main.py via
        renderer.tunnel_axis_fn). Même formule que le kernel : sinusoïdes à
        périodes ENTIÈRES de la boîte (kc = π/L -> enroulement sans couture),
        amplitudes/phases = état lissé du Lorenz caché (cf. update). (0,0) si le
        tunnel est OFF ou tant qu'update n'a pas tourné."""
        ax, phx, ay, phy = self._tun_axis
        kc = math.pi / max(self.radius, 1e-6)
        return (ax * math.sin(kc * z + phx), ay * math.sin(2.0 * kc * z + phy))

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
                f32(c["build"]), f32(c["drop"]), f32(c["harm_hue"]),
                f32(c["grav_amp"]), f32(c["grav_k"]), f32(c["grav_phase"]),
                f32(c["tunnel"]), f32(c["tunnel_speed"]),
                f32(c["tunnel_radius"]), f32(c["tunnel_wall"]),
                f32(c["tunnel_ax"]), f32(c["tunnel_phx"]),
                f32(c["tunnel_ay"]), f32(c["tunnel_phy"]),
                f32(c["tunnel_swirl"]),
                f32(c["bulb"]), f32(c["bulb_scale"]), f32(c["bulb_power"]),
                f32(c["bulb_reseed"])))
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
                    f32(c["harm_hue"]), f32(self.radius), f32(c["tunnel"])))

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
            # Kernel DÉDIÉ (prefill_emitted dans _CUDA_SOURCE) : écrit le ring EN PLACE,
            # 1 thread par slot, origines lues en mémoire globale -> AUCUN tableau
            # temporaire. L'ancienne version vectorisée (opos[src], ovel[src], broadcast)
            # allouait ~5 tableaux pleins (~50 o × n_emit, soit plusieurs Go sur un
            # preset long proche du plafond) et tombait en OutOfMemory -> prefill avalé
            # PILE sur les presets longs où il sert le plus. Réutilise la grille de
            # update_emitted (dimensionnée pour n_emit).
            with self._stream:
                self._k_prefill(self._grid_e, self._block, (
                    self.pos_state, self.vel_state, self.emit_state,
                    np.int32(self.n_emit), np.int32(self.n_origin),
                    np.float32(self.lifetime),
                    np.float32(self.radius), np.float32(_TUNNEL)))
            self._stream.synchronize()
        except Exception as exc:
            print(f"[particles] pré-remplissage des traînées ignoré ({exc}).",
                  file=sys.stderr)

    def release(self):
        try:
            if self._stream is not None: self._stream.synchronize()
        except Exception: pass
        if self._interop is not None:
            self._interop.release()        # désenregistre l'interop / libère le repli
        self._interop = None
        self.pos_state = None; self.vel_state = None; self.emit_state = None


if __name__ == "__main__":
    from gl_interop import _CudaGLDriver
    print("=== particles.py (nuée + émission, Lorenz caché) : auto-vérification ===")
    print(f"CuPy disponible      : {'OUI' if _HAS_CUPY else 'NON'}")
    _drv = _CudaGLDriver._load_driver()
    print(f"driver CUDA (interop): {'trouvé' if _drv is not None else 'absent'}")
