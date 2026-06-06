# -*- coding: utf-8 -*-
"""
main.py
=======
Sythm — Visualiseur de particules 3D audio-réactif temps réel (RTX 4090).

Ce fichier est LE POINT D'ENTRÉE et LA COLLE de tout le programme. Il assemble
les quatre grandes tranches développées séparément :

    AudioEngine   (audio_engine.py)   — capture loopback + FFT GPU -> features
    Renderer      (renderer.py)       — rendu des particules en HDR offscreen
    ParticleSystem(particles.py)      — simulation GPU (CUDA/compute) des positions
    PostProcessor (postfx.py)         — bloom + motion blur + tonemapping -> écran

RÔLE DE main.py (tranche « Intégrateur ») :
    - Exposer EN HAUT DE FICHIER un en-tête de PARAMÈTRES RÉGLABLES (ci-dessous),
      seul endroit que l'utilisateur final a besoin de toucher.
    - Créer la fenêtre + le contexte OpenGL/moderngl (via window.py).
    - Construire les modules DANS LE BON ORDRE (l'interop CUDA<->GL impose que les
      buffers GL existent AVANT que le système de particules s'y enregistre).
    - Faire tourner la BOUCLE DE RENDU (audio -> particules -> rendu -> post-effets
      -> présentation), gérer le clavier (toggles), le redimensionnement, les FPS.
    - Garantir un ARRÊT PROPRE (try/finally) : on libère l'audio, les particules
      et les ressources GL quoi qu'il arrive.

Lancement :
    uv run python main.py
    (ou)  python main.py
"""

from __future__ import annotations

import sys
import time

# Le PARAMÉTRAGE par défaut vit dans config_window.DEFAULTS (SOURCE DE VÉRITÉ
# unique, présentée dans la fenêtre de config au lancement). Les constantes de
# l'en-tête ci-dessous le LISENT — elles ne codent plus aucune valeur en dur.
# Import léger : config_window ne tire ni cupy ni moderngl (juste os/sys/json).
from config_window import DEFAULTS as _CFG


# #############################################################################
# #                                                                           #
# #   EN-TÊTE DE PARAMÈTRES RÉGLABLES  —  MODIFIEZ ICI ET NULLE PART AILLEURS  #
# #                                                                           #
# #############################################################################

# --- PARTICULES --------------------------------------------------------------
# Nombre de particules simulées. La RTX 4090 (24 Go VRAM, 16384 cœurs CUDA)
# encaisse très largement quelques millions de particules. 5 millions donne un
# nuage très dense ; on peut pousser à 8_000_000+ sur cette carte.
N_PARTICLES = _CFG["N_PARTICLES"]

# Nuée qui REMPLIT l'espace visible, advectée par un champ dont les coefficients
# sont un système de Lorenz CACHÉ (jamais affiché — il sous-tend la dynamique).
# Cf. particles.py. 'CLOUD_SHAPE' est ignoré.
CLOUD_SHAPE = "lorenz"

# Demi-côté de la BOÎTE de particules en unités monde (= l'espace visible rempli).
# La caméra est cadrée pour ~3 ; l'augmenter agrandit la boîte.
CLOUD_RADIUS = _CFG["CLOUD_RADIUS"]

# Taille apparente des particules en PIXELS écran (constante : ne varie ni avec
# la distance ni avec l'audio). Plus petit = grain fin façon champ d'étoiles ET
# moins de recouvrement additif -> moins de saturation blanche. ~3 est un bon
# point de départ pour plusieurs millions de particules. Petit = grain fin, on
# distingue chaque particule (anti-agrégation).
PARTICLE_SIZE = _CFG["PARTICLE_SIZE"]

# --- ÉMISSION DE TRAÎNÉES ÉPHÉMÈRES ------------------------------------------
# Chaque particule d'ORIGINE émet des particules de courte durée de vie, lancées
# dans la direction de son déplacement immédiat -> traînées qui révèlent le flux.
# Total affiché ≈ N_PARTICLES * (1 + EMIT_RATE * EMITTED_LIFETIME) -> dizaines de M.
EMITTED_LIFETIME = _CFG["EMITTED_LIFETIME"]   # durée de vie (s) des particules émises
EMIT_RATE = _CFG["EMIT_RATE"]                 # émissions par particule d'origine et par seconde


# --- FENÊTRE & CONTEXTE ------------------------------------------------------
# True  -> plein écran sur le moniteur primaire (résolution native).
# False -> fenêtré redimensionnable de taille WINDOW_W x WINDOW_H.
FULLSCREEN = _CFG["FULLSCREEN"]
WINDOW_W = _CFG["WINDOW_W"]
WINDOW_H = _CFG["WINDOW_H"]

# SUPERSAMPLING interne : on rend à (résolution_écran * SUPERSAMPLE_FACTOR) puis
# on réduit à l'affichage -> anti-aliasing « gratuit » et particules très nettes.
# 1.0 = natif, 1.5 = ~2.25x de pixels, 2.0 = 4x de pixels (réservé à la 4090).
# La résolution interne est bornée (voir MAX_RENDER_PIXELS) pour rester sain.
SUPERSAMPLE_FACTOR = _CFG["SUPERSAMPLE_FACTOR"]

# Plafond de pixels rendus en interne (garde-fou anti-explosion VRAM/charge).
# ~ 3840*2160*2.25 par défaut ; au-delà on réduit automatiquement le SS effectif.
MAX_RENDER_PIXELS = 10_000_000_000

# MSAA du framebuffer par défaut (présentation finale). Le rendu des particules,
# lui, profite surtout du supersampling ; voir la note dans window.py.
MSAA_SAMPLES = 2

# vsync : True -> synchronisé au moniteur (pas de tearing). False -> FPS max.
VSYNC = _CFG["VSYNC"]


# --- 3D STÉRÉOSCOPIQUE (relief réel, frame packing 1080p) --------------------
# Vraie stéréoscopie : la MÊME simulation est rendue depuis DEUX caméras off-axis
# (axes parallèles + frustums asymétriques -> parallaxe nulle au plan de
# convergence ; PAS de faux-3D reprojeté). Les deux yeux sont empaquetés au format
# FRAME PACKING 1080p de la HDMI 1.4a : œil gauche (haut) | 45 lignes d'« active
# space » | œil droit (bas) = 1920x2205, présenté à 24 Hz. Cf. stereo.py.
STEREO_3D = _CFG["STEREO_3D"]              # master : active le rendu/empaquetage stéréo
STEREO_EYE_SEP = _CFG["STEREO_EYE_SEP"]    # écart inter-oculaire (unités MONDE) ; + grand = + de relief
STEREO_CONVERGENCE = _CFG["STEREO_CONVERGENCE"]  # ×(distance œil->cible) du plan à parallaxe nulle
STEREO_SWAP_EYES = _CFG["STEREO_SWAP_EYES"]      # inverse G/D (confort / matériel à canaux inversés)

# Géométrie EXACTE du frame packing 1080p (HDMI 1.4a). NE PAS modifier : c'est le
# standard (1080 + 45 lignes d'intervalle de garde + 1080).
_FP_EYE_W = 1920
_FP_EYE_H = 1080
_FP_GAP = 45
_FP_TOTAL_H = _FP_EYE_H + _FP_GAP + _FP_EYE_H   # 2205
_FP_FPS = 24
# Taille d'APERÇU (fenêtré, portrait ~moitié) du mode 3D avant le plein écran sur
# l'afficheur 3D. Fenêtre DÉPLAÇABLE ; la géométrie EXACTE vient du plein écran (F).
_FP_PREVIEW = (960, 1040)


# --- POST-TRAITEMENT (look cinématographique) --------------------------------
ENABLE_BLOOM = _CFG["ENABLE_BLOOM"]               # OFF : pas de halo d'agrégation -> chaque particule nette
ENABLE_MOTION_BLUR = _CFG["ENABLE_MOTION_BLUR"]   # OFF : pas de traînée -> particules nettes, distinctes
MOTION_BLUR_STRENGTH = _CFG["MOTION_BLUR_STRENGTH"]  # 0 = aucune traînée, ->1 = traînées très longues
BLOOM_INTENSITY = _CFG["BLOOM_INTENSITY"]         # dosage du halo (réduit -> moins d'amas « plasma »)
BLOOM_THRESHOLD = 1.7          # seuil de luminance HDR (élevé -> seuls les vrais pics saignent)
EXPOSURE = _CFG["EXPOSURE"]                       # exposition basse : ADAPTÉE à ~35M (grain net, ~0% blanc).
#                                Plus de particules = additif plus fort -> baisser l'expo.

# DÉBRUITAGE à-trous : lisse le bruit de speckle (particules éparses) en une
# nébuleuse soyeuse, tout en préservant les filaments. ON par défaut.
ENABLE_DENOISE = _CFG["ENABLE_DENOISE"]
DENOISE_SIGMA = _CFG["DENOISE_SIGMA"]   # force ; + grand = + lisse (mais filaments moins nets)
DENOISE_ITERS = 7             # nb de passes à-trous (+ = rayon large, + lisse)


# --- ENREGISTREMENT (touche R) — H.265 ---------------------------------------
# La touche R démarre/arrête l'enregistrement de la diffusion en HEVC, avec le
# ffmpeg fourni par imageio-ffmpeg (aucune install système). Fichier .mp4 écrit
# dans RECORD_DIR, à cadence fixe. L'écriture se fait dans un thread dédié : la
# boucle de rendu n'est jamais figée par l'encodeur.
RECORD_ENCODER = _CFG["RECORD_ENCODER"]   # "x265" = encodeur LOGICIEL (qualité MAX, + lent ; bien
                               #   meilleur sur notre contenu sombre/granuleux : préserve
                               #   le grain et les filaments au lieu de les lisser).
                               #   "nvenc" = encodeur MATÉRIEL GPU (temps réel garanti).
RECORD_FPS = _CFG["RECORD_FPS"]   # images/s de la vidéo (pacing horloge -> timing correct).
                               #   NB : le visuel n'avance qu'au rythme du RENDU ; au-delà,
                               #   les images sont DUPLIQUÉES.
RECORD_DIR = "."               # dossier de sortie des enregistrements
RECORD_QUALITY = _CFG["RECORD_QUALITY"]   # x265 CRF / NVENC CQ — plus bas = meilleure qualité (+ gros).
                               #   x265 : ~18 = excellent, ~16 = quasi transparent.
RECORD_PRESET = "p5"           # Preset de l'encodeur ACTIF. "p5" = preset NVENC (p1..p7,
                               #   p7 = meilleure qualité) — cohérent avec RECORD_ENCODER
                               #   = "nvenc" par défaut. En x265, un nom NVENC est remplacé
                               #   par "medium" (cf. recorder.py) ; pour x265, mets un vrai
                               #   preset : ultrafast..placebo.
RECORD_PIXFMT = "p010le"       # 10 bits 4:2:0 (Main10, tue le banding)(yuv420p10le). Autres :
                               #   "yuv444p10le" = 4:4:4 (chroma pleine, filaments nets) ;
                               #   "yuv420p" = 8 bits (ancien comportement).
RECORD_AUDIO_BITRATE = _CFG["RECORD_AUDIO_BITRATE"]  # piste AUDIO AAC stéréo 48 kHz muxée dans la vidéo. C'est le son
                               #   SYSTÈME capté en loopback (ce que tu entends), dérivé du moteur
                               #   audio -> synchrone avec l'image. Mets "" (ou None) pour muet.


# --- CAMÉRA ------------------------------------------------------------------
# 'fixed'         -> caméra immobile.
# 'auto_rotate'   -> rotation continue autour du nuage.
# 'beat_reactive' -> rotation + impulsions/zoom rythmés par les beats audio.
CAMERA_MODE = _CFG["CAMERA_MODE"]
CAMERA_ROTATE_SPEED = _CFG["CAMERA_ROTATE_SPEED"]   # vitesse de rotation (rad/s) pour les modes auto.


# --- AUDIO -------------------------------------------------------------------
AUDIO_SAMPLERATE = 48000       # Hz. Standard WASAPI/PulseAudio.
AUDIO_BLOCKSIZE = 1024         # taille de BLOC de capture audio (≈21 ms à 48 kHz). NB : ce
                               #   n'est PAS la taille de FFT (fixée à 4096 dans audio_engine
                               #   pour la résolution spectrale), juste la granularité de capture.

# PALETTE COULEUR HDR éditable par l'utilisateur : liste de « stops » RGB.
# Les valeurs peuvent dépasser 1.0 (HDR) pour des couleurs qui « brillent » et
# nourrissent le bloom. Transmise au Renderer/Particles pour colorer le nuage
# (typiquement : du grave -> aigu, ou selon la position/vitesse).
USER_COLOR_GRADIENT = [
    (0.05, 0.10, 0.40),   # bleu nuit profond (basses fréquences / repos)
    (0.10, 0.55, 0.95),   # cyan électrique
    (0.95, 0.25, 0.85),   # magenta
    (1.40, 0.60, 0.10),   # orange HDR (>1 : brille)
    (1.80, 1.60, 1.20),   # blanc chaud surexposé (pics / aigus)
]


# --- PERFORMANCE / BOUCLE ----------------------------------------------------
# TARGET_FPS sert d'indication pour le mode adaptatif. En vsync, c'est le
# moniteur qui cadence. En mode non-vsync, on peut limiter pour économiser la
# carte (None = illimité).
TARGET_FPS = None              # ex: 144 pour brider ; None = pas de bridage
ADAPTIVE = True                # ajuste dynamiquement la charge si on rame (réservé)


# #############################################################################
# #                        FIN DE L'EN-TÊTE RÉGLABLE                          #
# #############################################################################


# -----------------------------------------------------------------------------
#  Configuration agrégée passée à la fenêtre.
#  window.py attend un objet avec des attributs (width, height, fullscreen,
#  msaa, vsync, title). On lui fournit ce petit conteneur plutôt qu'un dict.
# -----------------------------------------------------------------------------
class _WindowConfig:
    """Petit conteneur d'attributs pour configurer window.Window."""

    def __init__(self):
        self.width = WINDOW_W
        self.height = WINDOW_H
        self.fullscreen = FULLSCREEN
        self.msaa = MSAA_SAMPLES
        self.vsync = VSYNC
        self.title = "Sythm — Visualiseur audio GPU (RTX 4090)"
        # STÉRÉO : fenêtre DÉPLAÇABLE (décorée, redimensionnable) au format portrait
        # du frame packing. On NE fige PAS la géométrie et on NE force PAS le plein
        # écran : l'utilisateur glisse la fenêtre sur son afficheur/projecteur 3D
        # puis appuie sur F -> plein écran EXACT sur CE moniteur (1920x2205 en mode
        # frame packing) -> empaquetage pixel-exact. Taille d'aperçu portrait.
        if STEREO_3D:
            self.width, self.height = _FP_PREVIEW
            self.fullscreen = False


def _compute_render_resolution(screen_w, screen_h):
    """Calcule la résolution de rendu INTERNE (supersamplée) à partir de la
    résolution écran et de SUPERSAMPLE_FACTOR, en respectant MAX_RENDER_PIXELS.

    Retourne (render_w, render_h, ss_effectif).
    """
    ss = max(0.25, float(SUPERSAMPLE_FACTOR))
    render_w = max(1, int(round(screen_w * ss)))
    render_h = max(1, int(round(screen_h * ss)))

    # Garde-fou : si on dépasse le plafond de pixels, on réduit le SS effectif.
    pixels = render_w * render_h
    if pixels > MAX_RENDER_PIXELS:
        scale = (MAX_RENDER_PIXELS / pixels) ** 0.5
        render_w = max(1, int(render_w * scale))
        render_h = max(1, int(render_h * scale))
        ss = render_w / float(screen_w) if screen_w else ss
        print(f"[main] Supersampling bridé pour rester sous "
              f"{MAX_RENDER_PIXELS:,} px -> {render_w}x{render_h}",
              file=sys.stderr)

    return render_w, render_h, ss


# -----------------------------------------------------------------------------
#  Petit compteur de FPS lissé (affiché dans le titre de la fenêtre).
# -----------------------------------------------------------------------------
class _FpsCounter:
    """Moyenne glissante des FPS, rafraîchie ~2x/seconde pour le titre."""

    def __init__(self, smoothing=0.9):
        self._fps = 0.0
        self._smoothing = smoothing
        self._last_title_update = 0.0

    def update(self, dt):
        if dt > 0.0:
            inst = 1.0 / dt
            # Lissage exponentiel pour un affichage stable.
            self._fps = (self._smoothing * self._fps
                         + (1.0 - self._smoothing) * inst)

    @property
    def value(self):
        return self._fps

    def should_refresh_title(self, now, period=0.5):
        if now - self._last_title_update >= period:
            self._last_title_update = now
            return True
        return False


def _max_total_particles(n_origin, render_w, render_h, msaa):
    """Plafond RIGOUREUX de n_total (origines + émises), calculé sur le MATÉRIEL
    courant — donc valable quelle que soit la carte. C'est le MINIMUM de :

      (A) LIMITE D'INDEX 32 BITS des buffers GL : chaque buffer (pos, col) fait
          n_total * 16 octets, et moderngl/GL indexe offset/taille en int32 signé
          -> n_total * 16 doit rester < 2^31 (sinon overflow -> crash). Marge 64 Mo.

      (B) LIMITE VRAM : ce que la mémoire GPU LIBRE permet, en mode zéro-copie.
          Bilan exact par particule (octets) :
            buffers GL pos+col           = 32 * n_total
            état CUDA pos+vel (origines) = 24 * n_origin
            état CUDA emit (émises)      = 28 * (n_total - n_origin)
          => 60 * n_total - 4 * n_origin.
          On retire d'abord les framebuffers de rendu (HDR+MSAA+post-FX, fonction
          de la résolution) et une marge (pool cupy, fragmentation, headroom driver).

    Renvoie le plafond ABSOLU de n_total — qui PEUT être < n_origin si la carte ne
    peut même pas tenir les seules origines : l'appelant clampe alors n_origin."""
    cap_int32 = (2**31 - 1 - 64 * 1024 * 1024) // 16            # ~130,1 M

    try:
        import cupy as cp
        with cp.cuda.Device(0):
            free, total = cp.cuda.runtime.memGetInfo()
    except Exception:
        return cap_int32                                       # pas de query -> int32 seule

    px = max(1, int(render_w) * int(render_h))
    m = max(1, int(msaa))
    # Framebuffers de rendu (indép. du nb de particules) : RGBA16F HDR + resolve,
    # renderbuffers MSAA couleur+depth, tampons post-FX (bloom/denoise) — généreux.
    fb_bytes = px * (64 + 16 * m)
    # Marge : pool/fragmentation cupy, matrices audio, headroom driver.
    margin = max(int(total * 0.10), 512 * 1024 * 1024)
    budget = max(0, free - fb_bytes - margin)
    # 60*n_total - 4*n_origin <= budget  ->  n_total <= (budget + 4*n_origin)/60
    cap_vram = (budget + 4 * n_origin) // 60

    # MIN strict des deux limites — PAS de max(n_origin, …) : si n_origin lui-même
    # dépasse, on DOIT pouvoir renvoyer < n_origin pour que l'appelant le clampe.
    return int(max(1, min(cap_int32, cap_vram)))


# Préchauffe : nb de frames et plafond de temps. Quelques frames suffisent à
# finaliser les pipelines GL/post-FX et à grossir le pool mémoire cupy ; la
# fenêtre de temps laisse les horloges GPU monter, tout en bornant la durée sur
# une carte modeste (chaque frame de préchauffe y étant plus lente).
WARMUP_FRAMES = 16
WARMUP_MAX_SECONDS = 0.8


def _warmup(window, ctx, particles, renderer, present_frame):
    """Exécute le pipeline COMPLET (sim -> rendu HDR -> post-FX, mono OU stéréo)
    quelques frames AVANT la boucle visible, HORS-ÉCRAN (aucun swap), puis force la
    fin du travail GPU. On paie ICI (bref instant noir) ce qui rendait les premières
    frames vues lentes : finalisation des pipelines GL au 1er draw de millions de
    points, 1ers passes post-FX, montée du pool mémoire cupy et des horloges GPU. La
    1re frame VISIBLE est alors déjà « chaude ». `present_frame` dessine une frame
    complète (mono : post-FX plein écran ; stéréo : empaquetage frame packing des
    deux yeux). Borné par WARMUP_FRAMES / WARMUP_MAX_SECONDS. Tout échec est avalé :
    la préchauffe est un bonus, jamais bloquante."""
    try:
        dt = 1.0 / 60.0
        t0 = time.perf_counter()
        n = 0
        while n < WARMUP_FRAMES and (time.perf_counter() - t0) < WARMUP_MAX_SECONDS:
            particles.update(dt, None)
            renderer.update_camera(n * dt, None)
            present_frame()             # rendu COMPLET (mono ou stéréo), hors-écran
            window.poll_events()        # garde la fenêtre réactive (pas « ne répond pas »)
            n += 1
        # Pré-remplit le ring des traînées -> densité de RÉGIME dès la 1re frame vue
        # (sinon elle met EMITTED_LIFETIME s à se peupler : 18 s en Ambiant, 24 s en
        # Cosmique). Les origines ont une vitesse, établie par les updates ci-dessus.
        if hasattr(particles, "prefill_emitted"):
            particles.prefill_emitted()
            particles.update(dt, None)              # écrit l'état pré-rempli dans les buffers GL
            renderer.update_camera(n * dt, None)
            present_frame()
            window.poll_events()
        # Force la fin de TOUT le travail GPU différé MAINTENANT (pas en frame 1).
        try:
            import cupy as cp
            cp.cuda.Device().synchronize()
        except Exception:
            pass
        try:
            ctx.finish()
        except Exception:
            pass
        print(f"[main] Préchauffe : {n} frames en {time.perf_counter() - t0:.2f}s "
              f"(pipelines GL + pool cupy + horloges).", file=sys.stderr)
    except Exception as exc:
        print(f"[main] Préchauffe ignorée ({exc}).", file=sys.stderr)


def main():
    """Point d'entrée : BOUCLE « réglages -> simulation -> réglages ».

    La simulation ne QUITTE jamais le programme : l'arrêt en plein run (ESC ou
    croix de la fenêtre) REVIENT à l'écran de configuration, prêt à relancer
    aussitôt. On ne quitte VRAIMENT que depuis cette fenêtre (la fermer / l'annuler).
    """
    # Import (léger : os/sys/json) de la fenêtre de config. Différé pour que
    # py_compile et les outils statiques marchent même sans la lib d'UI.
    run_config = None
    _MAIN_KEYS, _PARTICLE_KEYS = [], []
    try:
        from config_window import (run_config,
                                    MAIN_KEYS as _MAIN_KEYS,
                                    PARTICLE_KEYS as _PARTICLE_KEYS)
    except Exception as exc:
        print(f"[main] Fenêtre de config indisponible ({exc}) -> défauts.",
              file=sys.stderr)
        run_config = None

    while True:
        # ----- 1) FENÊTRE DE CONFIGURATION (thémée TKinterModernThemes) -------
        #   dict  -> lancer avec ces réglages ; None -> fermée (on QUITTE) ;
        #   False -> UI indisponible/HS -> une seule session avec les défauts.
        cfg = run_config() if run_config is not None else False
        if cfg is None:
            print("[main] Configuration fermée — au revoir.", file=sys.stderr)
            return 0
        can_return = cfg is not False          # un dict => UI OK => on pourra revenir
        if cfg:                                 # applique les réglages choisis
            _g = globals()
            for _k in _MAIN_KEYS:
                if _k in cfg:
                    _g[_k] = cfg[_k]
            import particles as _pmod           # clés « _ » -> module particles
            for _k in _PARTICLE_KEYS:
                if _k in cfg:
                    setattr(_pmod, _k, cfg[_k])

        # ----- 2) SIMULATION — revient quand la fenêtre est fermée -----------
        if not _run_session():
            return 1                            # dépendances GPU manquantes

        # ----- 3) RETOUR AUX RÉGLAGES ----------------------------------------
        # Sans UI de config exploitable, impossible d'y revenir -> on quitte après
        # cette session unique (sinon on rejouerait en boucle infinie).
        if not can_return:
            return 0
        print("[main] Simulation arrêtée — retour aux réglages.", file=sys.stderr)


def _run_session():
    """Construit le moteur, fait tourner la boucle de rendu, puis NETTOIE TOUT
    (ordre inverse de construction) pour que la session suivante reparte d'un état
    propre : GLFW est ré-initialisable (Window.close() fait glfw.terminate()), le
    contexte GL/CUDA et le thread audio sont libérés. Revient à main() quand la
    fenêtre se ferme.

    Renvoie False si les dépendances GPU manquent (rien n'a démarré), True sinon
    (session menée à son terme puis nettoyée — y compris après une erreur attrapée).
    """
    # Imports « lourds » différés (cachés après le 1er appel) — gardés hors du
    # module pour que py_compile/outils statiques marchent même sans les libs GPU
    # (moderngl, cupy, glfw, soundcard...).
    try:
        from window import Window
        from renderer import Renderer
        from particles import ParticleSystem
        from postfx import PostProcessor
        from audio_engine import AudioEngine
        from recorder import Recorder
    except ImportError as exc:
        print("[main] Import d'un module du projet impossible :", exc,
              file=sys.stderr)
        print("       Vérifiez que renderer.py / particles.py / postfx.py /"
              " audio_engine.py / window.py sont présents et que les dépendances"
              " (moderngl, glfw, cupy, soundcard...) sont installées :",
              file=sys.stderr)
        print("           uv pip install -r requirements.txt", file=sys.stderr)
        return False

    # Références déclarées à None pour un nettoyage sûr dans le finally même si
    # une étape de construction échoue à mi-chemin.
    window = None
    renderer = None
    particles = None
    postfx = None
    audio = None
    loop_state = None

    try:
        # =====================================================================
        #  1) FENÊTRE + CONTEXTE OpenGL/moderngl
        # =====================================================================
        window = Window(_WindowConfig())
        ctx = window.ctx
        screen_w, screen_h = window.size

        # Mode 3D stéréoscopique (frame packing) : rendu PAR ŒIL à 1920x1080 natif
        # (pas de supersampling -> coût borné pour 2 yeux @ 24 Hz), empaqueté ensuite
        # en 1920x2205. Sinon, résolution interne (supersamplée) habituelle.
        stereo = bool(STEREO_3D)
        if stereo:
            render_w, render_h, ss_eff = _FP_EYE_W, _FP_EYE_H, 1.0
        else:
            render_w, render_h, ss_eff = _compute_render_resolution(screen_w, screen_h)

        # Comptes : origines (advectées) + émises (traînées éphémères, ring buffer).
        n_origin = N_PARTICLES
        n_emit = int(n_origin * EMIT_RATE * EMITTED_LIFETIME)
        # GARDE-FOU CALCULÉ SUR LE MATÉRIEL : plafonne n_total au MIN de la limite
        # d'index 32 bits des buffers GL et de ce que la VRAM LIBRE permet (cf.
        # _max_total_particles). Réduire n_emit ne casse PAS le flux : le taux
        # d'émission est dérivé de n_emit/lifetime -> pas de coupure franche.
        _max_total = _max_total_particles(n_origin, render_w, render_h, MSAA_SAMPLES)
        if n_origin + n_emit > _max_total:
            _req = n_origin + n_emit
            # On sacrifie d'abord les émises (traînées) ; si même les origines seules
            # dépassent (N_PARTICLES hors bornes via un sythm_config.json édité à la
            # main), on CLAMPE AUSSI n_origin -> jamais d'overflow d'index 32 bits/OOM.
            n_origin = min(n_origin, _max_total)
            n_emit = max(0, _max_total - n_origin)
            print(f"[main] Particules plafonnées (matériel) : {_req:,} demandées -> "
                  f"{n_origin + n_emit:,} (VRAM libre + index 32 bits des buffers GL).",
                  file=sys.stderr)
        n_total = n_origin + n_emit
        print(f"[main] Écran {screen_w}x{screen_h} | Rendu interne "
              f"{render_w}x{render_h} (SS x{ss_eff:.2f}) | {n_origin:,} origines + "
              f"{n_emit:,} émises = {n_total:,} particules",
              file=sys.stderr)

        # =====================================================================
        #  2) RENDERER  — crée les buffers GL pos/col (AVANT les particules !)
        # =====================================================================
        # L'ORDRE est critique : Renderer alloue les VBOs de positions/couleurs,
        # que ParticleSystem va ensuite enregistrer pour l'interop CUDA. Si on
        # créait les particules d'abord, il n'y aurait aucun buffer GL à mapper.
        renderer = Renderer(
            ctx,
            render_w,
            render_h,
            n_total,            # buffers GL = origines + émises (tout est rendu)
            MSAA_SAMPLES,
            color_gradient=USER_COLOR_GRADIENT,
        )
        # Taille des particules (px écran) réglée depuis l'en-tête réglable.
        renderer.point_size = float(PARTICLE_SIZE)
        # Active la STÉRÉO off-axis sur le renderer si la session est en 3D : les
        # deux yeux partageront ces mêmes buffers (une seule simulation), rendus via
        # renderer.render(eye=-1/+1) depuis deux frustums asymétriques.
        if stereo and hasattr(renderer, "enable_stereo"):
            renderer.enable_stereo(True, iod=STEREO_EYE_SEP,
                                   convergence_scale=STEREO_CONVERGENCE)

        # =====================================================================
        #  3) PARTICLE SYSTEM  — s'enregistre sur les buffers GL du Renderer
        # =====================================================================
        particles = ParticleSystem(
            n_origin,
            n_emit,
            renderer.pos_buffer,
            renderer.col_buffer,
            CLOUD_SHAPE,
            CLOUD_RADIUS,
            EMITTED_LIFETIME,
        )

        # =====================================================================
        #  4) POST-PROCESSOR  — bloom / motion blur / tonemapping -> écran
        # =====================================================================
        # Fabrique d'une chaîne post-FX (réglages affinés appliqués), réutilisée :
        # une fois en mono (plein écran), deux fois en stéréo (un œil chacune ->
        # historiques motion-blur SÉPARÉS, pas de fantôme inter-yeux).
        def _build_postfx(scr_w, scr_h):
            p = PostProcessor(
                ctx, render_w, render_h, scr_w, scr_h,
                enable_bloom=ENABLE_BLOOM,
                enable_motion_blur=ENABLE_MOTION_BLUR,
                exposure=EXPOSURE,
            )
            _safe_set_params(
                p,
                bloom_intensity=BLOOM_INTENSITY,
                bloom_threshold=BLOOM_THRESHOLD,
                motion_blur_strength=MOTION_BLUR_STRENGTH,
                exposure=EXPOSURE,
                enable_denoise=ENABLE_DENOISE,
                denoise_sigma=DENOISE_SIGMA,
                denoise_iters=DENOISE_ITERS,
            )
            return p

        if stereo:
            # StereoRig : deux chaînes 1920x1080 (render = screen, pas de downscale)
            # empaquetées en frame packing 1920x2205. Expose set_params/release ->
            # interchangeable avec PostProcessor pour les toggles et le nettoyage.
            from stereo import StereoRig
            postfx = StereoRig(ctx, renderer,
                               lambda: _build_postfx(render_w, render_h),
                               swap_eyes=STEREO_SWAP_EYES)
            print(f"[main] Mode 3D STÉRÉOSCOPIQUE — frame packing "
                  f"{_FP_EYE_W}x{_FP_TOTAL_H}@{_FP_FPS} (œil {_FP_EYE_W}x{_FP_EYE_H} + "
                  f"{_FP_GAP} lignes de garde + œil {_FP_EYE_W}x{_FP_EYE_H}) | "
                  f"IOD={STEREO_EYE_SEP}, conv x{STEREO_CONVERGENCE}"
                  f"{', G/D inversés' if STEREO_SWAP_EYES else ''}.", file=sys.stderr)
            print("[main] 3D : glisse la fenêtre sur ton afficheur/projecteur 3D, "
                  "puis F -> plein écran pixel-exact. ESC revient aux réglages.",
                  file=sys.stderr)
        else:
            postfx = _build_postfx(screen_w, screen_h)

        # =====================================================================
        #  5) AUDIO ENGINE  — démarre le thread de capture loopback + FFT GPU
        # =====================================================================
        audio = AudioEngine(samplerate=AUDIO_SAMPLERATE, blocksize=AUDIO_BLOCKSIZE)
        try:
            audio.start()
        except Exception as exc:
            # Pas d'audio (pas de loopback / pas de carte) ne doit pas tuer le
            # visu : on continue avec des features nulles (nuage « au repos »).
            print(f"[main] Audio indisponible, on continue sans : {exc}",
                  file=sys.stderr)

        # ----- État mutable de la boucle (modifiable au clavier) -------------
        loop_state = _LoopState()
        loop_state.camera_mode = CAMERA_MODE
        loop_state.enable_bloom = ENABLE_BLOOM
        loop_state.enable_motion_blur = ENABLE_MOTION_BLUR

        # Câblage des touches clavier (B/M/C/...) -> via window si dispo.
        _install_key_toggles(window, loop_state, postfx, renderer)

        # Mode caméra initial choisi dans l'en-tête (le renderer mappe les noms
        # "auto_rotate"/"beat_reactive" vers ses modes internes "auto"/"beat").
        if hasattr(renderer, "set_camera_mode"):
            renderer.set_camera_mode(CAMERA_MODE)

        _print_controls()

        # Fermeture de rendu d'UNE frame complète (mono ou stéréo), réutilisée par
        # la préchauffe ET la boucle. NE fait PAS update_camera (l'appelant le fait
        # avec le bon temps t). Mono -> post-FX plein écran ; stéréo -> empaquetage
        # frame packing des deux yeux dans le framebuffer par défaut (window.size).
        if stereo:
            def _present_frame():
                postfx.render_frame(ctx.screen, *window.size)
        else:
            def _present_frame():
                postfx.process(renderer.render(), ctx.screen)

        # =====================================================================
        #  PRÉCHAUFFE — avant la boucle visible : on absorbe ICI le vrai warmup
        #  GPU (1er draw de millions de points, 1ers post-FX, pool cupy, horloges)
        #  en rendant hors-écran, pour que la 1re frame VUE soit déjà chaude.
        # =====================================================================
        _warmup(window, ctx, particles, renderer, _present_frame)

        # =====================================================================
        #  BOUCLE PRINCIPALE
        # =====================================================================
        fps = _FpsCounter()
        t_start = time.perf_counter()
        t_prev = t_start

        while not window.doit_fermer():
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            # Temps écoulé depuis le démarrage (alimente caméra & animation).
            t = now - t_start

            # --- Bascules différées (traitées hors callback GLFW) ------------
            # F : plein écran borderless (change la taille -> resize géré juste après).
            if loop_state.toggle_fullscreen:
                loop_state.toggle_fullscreen = False
                # F : plein écran borderless sur le moniteur SOUS la fenêtre. En 3D,
                # couverture EXACTE (exact=True) -> sur un afficheur en frame packing
                # 1920x2205 le framebuffer fait pile cette taille -> empaquetage
                # pixel-exact. En mono, débordement 1px habituel (HDR préservé).
                window.toggle_fullscreen(exact=stereo)
            # R : démarre / arrête l'enregistrement HEVC (x265/NVENC).
            if loop_state.toggle_record:
                loop_state.toggle_record = False
                if loop_state.recorder is None:
                    # Taille RÉELLE du framebuffer par défaut (= taille GLFW du
                    # framebuffer). SURTOUT PAS ctx.screen.size : moderngl ne met
                    # pas à jour la taille de l'écran au resize -> en plein écran
                    # elle resterait à la taille fenêtrée initiale et la capture
                    # ne prendrait qu'un coin de l'image.
                    sw, sh = window.size
                    try:
                        # Dérivation du flux audio loopback -> piste AAC muxée
                        # (synchrone). RECORD_AUDIO_BITRATE vide -> vidéo muette.
                        audio_q = (audio.start_recording_tap()
                                   if (audio is not None and RECORD_AUDIO_BITRATE)
                                   else None)
                        a_rate = (getattr(audio, "samplerate", 48000)
                                  if audio is not None else 48000)
                        loop_state.recorder = Recorder(
                            sw, sh, RECORD_FPS, RECORD_DIR, RECORD_ENCODER,
                            RECORD_QUALITY, RECORD_PRESET, RECORD_PIXFMT,
                            audio_queue=audio_q, audio_rate=a_rate,
                            audio_bitrate=RECORD_AUDIO_BITRATE)
                        _atxt = (f", AAC {RECORD_AUDIO_BITRATE}" if audio_q
                                 else ", muet")
                        print(f"[record] ● REC {sw}x{sh} {RECORD_ENCODER}/"
                              f"{RECORD_PIXFMT} (HEVC q{RECORD_QUALITY}{_atxt}) -> "
                              f"{loop_state.recorder.path}", file=sys.stderr)
                    except Exception as exc:
                        print(f"[record] démarrage impossible : {exc}", file=sys.stderr)
                        loop_state.recorder = None
                else:
                    rec = loop_state.recorder
                    loop_state.recorder = None
                    if audio is not None:
                        audio.stop_recording_tap()   # stoppe la dérivation audio
                    path, nframes = rec.close()      # draine le reste + muxe AAC
                    extra = (f", {rec.dropped} droppées"
                             if getattr(rec, "dropped", 0) else "")
                    print(f"[record] ■ sauvegardé : {path} "
                          f"({nframes} frames{extra})", file=sys.stderr)
                    if getattr(rec, "dropped", 0):
                        print("[record] ⚠ encodeur en retard : baisse "
                              "RECORD_PRESET (ex. 'fast') ou la résolution.",
                              file=sys.stderr)
                    if getattr(rec, "last_error", None):
                        print(f"[record] ⚠ ffmpeg : {rec.last_error}",
                              file=sys.stderr)

            # --- Redimensionnement éventuel (fenêtré ; jamais en 3D figée) ----
            _resized = window.resized            # lecture consommatrice du drapeau
            if _resized and not stereo:
                new_w, new_h = window.size
                r_w, r_h, _ = _compute_render_resolution(new_w, new_h)
                renderer.resize(r_w, r_h)
                _safe_resize_postfx(postfx, r_w, r_h, new_w, new_h)

            # --- 1) Récupération des features audio (thread-safe, non bloquant)
            if audio is not None:
                features = audio.get_features()
            else:
                features = None

            # --- 2) Simulation des particules (CUDA écrit dans les buffers GL)
            particles.update(dt, features)

            # --- 3+4) Caméra + rendu + post-traitement -> framebuffer écran --
            # Mono : 1 rendu HDR -> post-FX plein écran. Stéréo : 2 rendus off-axis
            # (un par œil) empaquetés en frame packing (cf. _present_frame).
            renderer.update_camera(t, features)
            _present_frame()

            # --- 4b) Capture vidéo (si enregistrement actif), AVANT le swap ---
            if loop_state.recorder is not None:
                loop_state.recorder.maybe_capture(ctx.screen, window.size)

            # --- 5) Présentation + événements + FPS --------------------------
            window.swap_buffers()
            window.poll_events()

            fps.update(dt)
            if fps.should_refresh_title(now):
                _update_title(window, fps.value, loop_state)

            # Bridage des FPS : 24 Hz IMPOSÉ en 3D (cadence frame packing 1080p),
            # sinon TARGET_FPS optionnel (mode non-vsync).
            _maybe_limit_fps(now, _FP_FPS if stereo else TARGET_FPS)

    except KeyboardInterrupt:
        print("\n[main] Interruption clavier — arrêt.", file=sys.stderr)
    except Exception as exc:
        # On laisse remonter une trace utile, mais on passe quand même par le
        # nettoyage du finally.
        print(f"[main] Erreur fatale dans la boucle : {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        # =====================================================================
        #  NETTOYAGE  — toujours exécuté, dans l'ordre inverse de construction.
        # =====================================================================
        print("[main] Nettoyage des ressources...", file=sys.stderr)
        _safe_call(lambda: loop_state.recorder.close()
                   if (loop_state is not None and loop_state.recorder is not None)
                   else None, "recorder.close()")
        _safe_call(lambda: audio.stop() if audio is not None else None,
                   "audio.stop()")
        _safe_call(lambda: particles.release() if particles is not None else None,
                   "particles.release()")
        _safe_call(lambda: _release_renderer(renderer), "renderer.release()")
        _safe_call(lambda: _release_postfx(postfx), "postfx.release()")
        _safe_call(lambda: window.close() if window is not None else None,
                   "window.close()")
        print("[main] Terminé.", file=sys.stderr)

    return True


# -----------------------------------------------------------------------------
#  État de la boucle (modifié par les toggles clavier).
# -----------------------------------------------------------------------------
class _LoopState:
    def __init__(self):
        self.camera_mode = CAMERA_MODE
        self.enable_bloom = ENABLE_BLOOM
        self.enable_motion_blur = ENABLE_MOTION_BLUR
        self.toggle_fullscreen = False   # demande de bascule plein écran (touche F)
        self.toggle_record = False       # demande de bascule enregistrement (touche R)
        self.recorder = None             # Recorder actif (None = pas d'enregistrement)


# Liste ordonnée des modes caméra pour le cyclage avec la touche C.
_CAMERA_MODES = ("fixed", "auto_rotate", "beat_reactive")


def _install_key_toggles(window, state, postfx, renderer):
    """Installe les bascules clavier si window expose un point d'extension.

    window.py de base ne gère que ESC. On tente d'enregistrer un callback
    additionnel via une API optionnelle `window.set_extra_key_callback(fn)` ;
    si elle n'existe pas, on l'ignore proprement (les toggles seront inertes,
    mais le programme tourne). Ceci permet à window.py d'évoluer sans casser
    main.py, et inversement.
    """
    import importlib

    try:
        glfw = importlib.import_module("glfw")
    except Exception:
        glfw = None

    def on_extra_key(key, action):
        # On ne réagit qu'à l'appui (PRESS), pas au relâchement/répétition.
        if glfw is not None and action != glfw.PRESS:
            return

        # B -> bascule bloom
        if glfw is not None and key == glfw.KEY_B:
            state.enable_bloom = not state.enable_bloom
            _safe_set_params(postfx, enable_bloom=state.enable_bloom)
            print(f"[touche] Bloom : {'ON' if state.enable_bloom else 'OFF'}")

        # M -> bascule motion blur
        elif glfw is not None and key == glfw.KEY_M:
            state.enable_motion_blur = not state.enable_motion_blur
            _safe_set_params(postfx, enable_motion_blur=state.enable_motion_blur)
            print(f"[touche] Motion blur : "
                  f"{'ON' if state.enable_motion_blur else 'OFF'}")

        # C -> cycle des modes caméra
        elif glfw is not None and key == glfw.KEY_C:
            idx = _CAMERA_MODES.index(state.camera_mode) \
                if state.camera_mode in _CAMERA_MODES else 0
            state.camera_mode = _CAMERA_MODES[(idx + 1) % len(_CAMERA_MODES)]
            # On informe le renderer si une API existe (sinon inerte).
            if hasattr(renderer, "set_camera_mode"):
                _safe_call(lambda: renderer.set_camera_mode(state.camera_mode),
                           "renderer.set_camera_mode")
            print(f"[touche] Caméra : {state.camera_mode}")

        # F -> plein écran borderless (différé : appliqué dans la boucle de rendu).
        elif glfw is not None and key == glfw.KEY_F:
            state.toggle_fullscreen = True

        # R -> démarre / arrête l'enregistrement HEVC x265/NVENC (différé).
        elif glfw is not None and key == glfw.KEY_R:
            state.toggle_record = True

    # Branche le callback si l'API optionnelle est disponible.
    if hasattr(window, "set_extra_key_callback"):
        window.set_extra_key_callback(on_extra_key)
    else:
        # window.py minimal : on ne peut pas brancher les toggles. On le note.
        print("[main] (info) window.set_extra_key_callback absent : les touches "
              "B/M/C/F/R sont inactives. ESC fonctionne toujours.", file=sys.stderr)


def _update_title(window, fps_value, state):
    """Met à jour le titre de la fenêtre avec les FPS et l'état des effets."""
    import importlib
    try:
        glfw = importlib.import_module("glfw")
    except Exception:
        return
    bloom = "B" if state.enable_bloom else "-"
    mblur = "M" if state.enable_motion_blur else "-"
    rec = "  ● REC" if getattr(state, "recorder", None) is not None else ""
    fs = "FS " if getattr(window, "is_fullscreen", False) else ""
    d3 = "3D " if STEREO_3D else ""
    title = (f"Sythm | {fps_value:5.1f} FPS | {N_PARTICLES:,} part. | "
             f"[{bloom}{mblur}] {d3}{fs}cam:{state.camera_mode}{rec}")
    if getattr(window, "handle", None) is not None:
        glfw.set_window_title(window.handle, title)


def _maybe_limit_fps(frame_start, target_fps):
    """Bride le framerate à `target_fps` (None/<=0 = pas de bridage).

    Sert à IMPOSER 24 Hz en 3D (cadence frame packing 1080p) et à appliquer
    TARGET_FPS en mono non-vsync."""
    if target_fps is None or target_fps <= 0:
        return
    target_dt = 1.0 / float(target_fps)
    elapsed = time.perf_counter() - frame_start
    remaining = target_dt - elapsed
    if remaining > 0:
        time.sleep(remaining)


def _print_controls():
    """Affiche les contrôles dans la console au démarrage."""
    print(
        "\n=== Sythm — contrôles ===\n"
        "  ESC : quitter\n"
        "  B   : activer/désactiver le bloom\n"
        "  M   : activer/désactiver le motion blur\n"
        "  C   : changer de mode caméra (fixed / auto_rotate / beat_reactive)\n"
        "  F   : plein écran borderless (HDR préservé) (bascule)\n"
        "  R   : démarrer/arrêter l'enregistrement HEVC (x265/NVENC)\n"
        "============================\n",
        file=sys.stderr,
    )


# -----------------------------------------------------------------------------
#  Helpers de robustesse : ne jamais crasher sur une API optionnelle absente.
# -----------------------------------------------------------------------------
def _safe_call(fn, label):
    """Exécute fn() en avalant les erreurs (utilisé au nettoyage)."""
    try:
        fn()
    except Exception as exc:
        print(f"[main] (nettoyage) {label} a échoué : {exc}", file=sys.stderr)


def _safe_set_params(obj, **kwargs):
    """Appelle obj.set_params(**kwargs) si la méthode existe, sinon ignore."""
    if obj is not None and hasattr(obj, "set_params"):
        try:
            obj.set_params(**kwargs)
        except Exception as exc:
            print(f"[main] set_params{tuple(kwargs)} ignoré : {exc}",
                  file=sys.stderr)


def _safe_resize_postfx(postfx, render_w, render_h, screen_w, screen_h):
    """Appelle postfx.resize en s'adaptant à sa signature (4 args ou 2 args)."""
    if postfx is None or not hasattr(postfx, "resize"):
        return
    try:
        postfx.resize(render_w, render_h, screen_w, screen_h)
    except TypeError:
        # Signature alternative resize(w, h) -> on passe la résolution interne.
        try:
            postfx.resize(render_w, render_h)
        except Exception as exc:
            print(f"[main] postfx.resize ignoré : {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[main] postfx.resize ignoré : {exc}", file=sys.stderr)


def _release_renderer(renderer):
    """Libère le renderer si une méthode de libération existe."""
    if renderer is None:
        return
    for name in ("release", "close", "destroy"):
        if hasattr(renderer, name):
            getattr(renderer, name)()
            return


def _release_postfx(postfx):
    """Libère le post-processor si une méthode de libération existe."""
    if postfx is None:
        return
    for name in ("release", "close", "destroy"):
        if hasattr(postfx, name):
            getattr(postfx, name)()
            return


if __name__ == "__main__":
    sys.exit(main())
