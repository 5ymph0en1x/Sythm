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


# #############################################################################
# #                                                                           #
# #   EN-TÊTE DE PARAMÈTRES RÉGLABLES  —  MODIFIEZ ICI ET NULLE PART AILLEURS  #
# #                                                                           #
# #############################################################################

# --- PARTICULES --------------------------------------------------------------
# Nombre de particules simulées. La RTX 4090 (24 Go VRAM, 16384 cœurs CUDA)
# encaisse très largement quelques millions de particules. 5 millions donne un
# nuage très dense ; on peut pousser à 8_000_000+ sur cette carte.
N_PARTICLES = 2_500_000

# Nuée qui REMPLIT l'espace visible, advectée par un champ dont les coefficients
# sont un système de Lorenz CACHÉ (jamais affiché — il sous-tend la dynamique).
# Cf. particles.py. 'CLOUD_SHAPE' est ignoré.
CLOUD_SHAPE = "lorenz"

# Demi-côté de la BOÎTE de particules en unités monde (= l'espace visible rempli).
# La caméra est cadrée pour ~3 ; l'augmenter agrandit la boîte.
CLOUD_RADIUS = 2.5

# Taille apparente des particules en PIXELS écran (constante : ne varie ni avec
# la distance ni avec l'audio). Plus petit = grain fin façon champ d'étoiles ET
# moins de recouvrement additif -> moins de saturation blanche. ~3 est un bon
# point de départ pour plusieurs millions de particules. Petit = grain fin, on
# distingue chaque particule (anti-agrégation).
PARTICLE_SIZE = 0.5

# --- ÉMISSION DE TRAÎNÉES ÉPHÉMÈRES ------------------------------------------
# Chaque particule d'ORIGINE émet des particules de courte durée de vie, lancées
# dans la direction de son déplacement immédiat -> traînées qui révèlent le flux.
# Total affiché ≈ N_PARTICLES * (1 + EMIT_RATE * EMITTED_LIFETIME) -> dizaines de M.
EMITTED_LIFETIME = 10.0    # durée de vie (s) des particules émises
EMIT_RATE = 5.0           # émissions par particule d'origine et par seconde


# --- FENÊTRE & CONTEXTE ------------------------------------------------------
# True  -> plein écran sur le moniteur primaire (résolution native).
# False -> fenêtré redimensionnable de taille WINDOW_W x WINDOW_H.
FULLSCREEN = False
WINDOW_W = 1280
WINDOW_H = 720

# SUPERSAMPLING interne : on rend à (résolution_écran * SUPERSAMPLE_FACTOR) puis
# on réduit à l'affichage -> anti-aliasing « gratuit » et particules très nettes.
# 1.0 = natif, 1.5 = ~2.25x de pixels, 2.0 = 4x de pixels (réservé à la 4090).
# La résolution interne est bornée (voir MAX_RENDER_PIXELS) pour rester sain.
SUPERSAMPLE_FACTOR = 1.0

# Plafond de pixels rendus en interne (garde-fou anti-explosion VRAM/charge).
# ~ 3840*2160*2.25 par défaut ; au-delà on réduit automatiquement le SS effectif.
MAX_RENDER_PIXELS = 10_000_000_000

# MSAA du framebuffer par défaut (présentation finale). Le rendu des particules,
# lui, profite surtout du supersampling ; voir la note dans window.py.
MSAA_SAMPLES = 4

# vsync : True -> synchronisé au moniteur (pas de tearing). False -> FPS max.
VSYNC = True


# --- POST-TRAITEMENT (look cinématographique) --------------------------------
ENABLE_BLOOM = False           # OFF : pas de halo d'agrégation -> chaque particule nette
ENABLE_MOTION_BLUR = False     # OFF : pas de traînée -> particules nettes, distinctes
MOTION_BLUR_STRENGTH = 0.15    # 0 = aucune traînée, ->1 = traînées très longues
BLOOM_INTENSITY = 0.3          # dosage du halo (réduit -> moins d'amas « plasma »)
BLOOM_THRESHOLD = 1.7          # seuil de luminance HDR (élevé -> seuls les vrais pics saignent)
EXPOSURE = 0.15                # exposition basse : ADAPTÉE à ~35M (grain net, ~0% blanc).
#                                Plus de particules = additif plus fort -> baisser l'expo.

# DÉBRUITAGE à-trous : lisse le bruit de speckle (particules éparses) en une
# nébuleuse soyeuse, tout en préservant les filaments. ON par défaut.
ENABLE_DENOISE = True
DENOISE_SIGMA = 0.01            # force ; + grand = + lisse (mais filaments moins nets)
DENOISE_ITERS = 7             # nb de passes à-trous (+ = rayon large, + lisse)


# --- ENREGISTREMENT (touche R) — H.265 ---------------------------------------
# La touche R démarre/arrête l'enregistrement de la diffusion en HEVC, avec le
# ffmpeg fourni par imageio-ffmpeg (aucune install système). Fichier .mp4 écrit
# dans RECORD_DIR, à cadence fixe. L'écriture se fait dans un thread dédié : la
# boucle de rendu n'est jamais figée par l'encodeur.
RECORD_ENCODER = "x265"        # "x265" = encodeur LOGICIEL (qualité MAX, + lent ; bien
                               #   meilleur sur notre contenu sombre/granuleux : préserve
                               #   le grain et les filaments au lieu de les lisser).
                               #   "nvenc" = encodeur MATÉRIEL GPU (temps réel garanti).
RECORD_FPS = 30                # images/s de la vidéo (pacing horloge -> timing correct).
                               #   NB : le visuel n'avance qu'au rythme du RENDU ; au-delà,
                               #   les images sont DUPLIQUÉES.
RECORD_DIR = "."               # dossier de sortie des enregistrements
RECORD_QUALITY = 16            # x265 CRF / NVENC CQ — plus bas = meilleure qualité (+ gros).
                               #   x265 : ~18 = excellent, ~16 = quasi transparent.
RECORD_PRESET = "faster"       # x265 : ultrafast..placebo (medium = bon équilibre ; si trop
                               #   de frames sont droppées, passe à "fast"/"faster").
                               #   nvenc : p1..p7 (p7 = meilleure qualité).
RECORD_PIXFMT = "p010"  # 10 bits 4:2:0 (Main10, tue le banding). Autres :
                               #   "yuv444p10le" = 4:4:4 (chroma pleine, filaments nets) ;
                               #   "yuv420p" = 8 bits (ancien comportement).


# --- CAMÉRA ------------------------------------------------------------------
# 'fixed'         -> caméra immobile.
# 'auto_rotate'   -> rotation continue autour du nuage.
# 'beat_reactive' -> rotation + impulsions/zoom rythmés par les beats audio.
CAMERA_MODE = "beat_reactive"
CAMERA_ROTATE_SPEED = 0.15     # vitesse de rotation (rad/s) pour les modes auto.


# --- AUDIO -------------------------------------------------------------------
AUDIO_SAMPLERATE = 48000       # Hz. Standard WASAPI/PulseAudio.
FFT_SIZE = 1024                # taille de bloc / FFT (≈21 ms à 48 kHz). Puissance de 2.

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


def main():
    """Point d'entrée : construit tout, fait tourner la boucle, nettoie."""

    # ----- Imports « lourds » différés ---------------------------------------
    # On importe ici (et non en tête de module) pour que `python -m py_compile`
    # et les outils statiques restent fonctionnels même si les dépendances GPU
    # (moderngl, cupy, glfw, soundcard...) ne sont pas installées sur la machine
    # de développement. L'échec d'import est alors signalé clairement à l'exécution.
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
        return 1

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

        # Résolution de rendu interne (supersamplée), bornée pour la sécurité.
        render_w, render_h, ss_eff = _compute_render_resolution(screen_w, screen_h)

        # Comptes : origines (advectées) + émises (traînées éphémères, ring buffer).
        n_origin = N_PARTICLES
        n_emit = int(n_origin * EMIT_RATE * EMITTED_LIFETIME)
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
        postfx = PostProcessor(
            ctx,
            render_w,
            render_h,
            screen_w,
            screen_h,
            enable_bloom=ENABLE_BLOOM,
            enable_motion_blur=ENABLE_MOTION_BLUR,
            exposure=EXPOSURE,
        )
        # Paramètres affinés (intensités/seuils) si le PostProcessor les accepte.
        _safe_set_params(
            postfx,
            bloom_intensity=BLOOM_INTENSITY,
            bloom_threshold=BLOOM_THRESHOLD,
            motion_blur_strength=MOTION_BLUR_STRENGTH,
            exposure=EXPOSURE,
            enable_denoise=ENABLE_DENOISE,
            denoise_sigma=DENOISE_SIGMA,
            denoise_iters=DENOISE_ITERS,
        )

        # =====================================================================
        #  5) AUDIO ENGINE  — démarre le thread de capture loopback + FFT GPU
        # =====================================================================
        audio = AudioEngine(samplerate=AUDIO_SAMPLERATE, blocksize=FFT_SIZE)
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
                window.toggle_fullscreen()
            # R : démarre / arrête l'enregistrement HEVC (x265/NVENC).
            if loop_state.toggle_record:
                loop_state.toggle_record = False
                if loop_state.recorder is None:
                    # Dimensions du framebuffer FINAL réellement lu par le recorder.
                    sw, sh = ctx.screen.size
                    try:
                        loop_state.recorder = Recorder(
                            sw, sh, RECORD_FPS, RECORD_DIR, RECORD_ENCODER,
                            RECORD_QUALITY, RECORD_PRESET, RECORD_PIXFMT)
                        print(f"[record] ● REC {sw}x{sh} {RECORD_ENCODER}/"
                              f"{RECORD_PIXFMT} (HEVC q{RECORD_QUALITY}) -> "
                              f"{loop_state.recorder.path}", file=sys.stderr)
                    except Exception as exc:
                        print(f"[record] démarrage impossible : {exc}", file=sys.stderr)
                        loop_state.recorder = None
                else:
                    rec = loop_state.recorder
                    loop_state.recorder = None
                    path, nframes = rec.close()
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

            # --- Redimensionnement éventuel (fenêtré) ------------------------
            if window.resized:
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

            # --- 3) Caméra + rendu HDR offscreen -----------------------------
            renderer.update_camera(t, features)
            hdr_tex = renderer.render()

            # --- 4) Post-traitement -> framebuffer écran ---------------------
            postfx.process(hdr_tex, ctx.screen)

            # --- 4b) Capture vidéo (si enregistrement actif), AVANT le swap ---
            if loop_state.recorder is not None:
                loop_state.recorder.maybe_capture(ctx.screen)

            # --- 5) Présentation + événements + FPS --------------------------
            window.swap_buffers()
            window.poll_events()

            fps.update(dt)
            if fps.should_refresh_title(now):
                _update_title(window, fps.value, loop_state)

            # Bridage optionnel des FPS (mode non-vsync).
            _maybe_limit_fps(now)

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

    return 0


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
    title = (f"Sythm | {fps_value:5.1f} FPS | {N_PARTICLES:,} part. | "
             f"[{bloom}{mblur}] {fs}cam:{state.camera_mode}{rec}")
    if getattr(window, "handle", None) is not None:
        glfw.set_window_title(window.handle, title)


def _maybe_limit_fps(frame_start):
    """Bride le framerate si TARGET_FPS est défini (mode non-vsync)."""
    if TARGET_FPS is None or TARGET_FPS <= 0:
        return
    target_dt = 1.0 / float(TARGET_FPS)
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
