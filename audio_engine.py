# -*- coding: utf-8 -*-
"""
audio_engine.py
================

MODULE DE CAPTURE AUDIO LOOPBACK + ANALYSE SPECTRALE GPU
--------------------------------------------------------
Tranche « audio » d'un visualiseur de particules 3D temps réel (RTX 4090),
en Python + GLSL. Ce module ne s'occupe QUE du son : il capture ce qui sort
des haut-parleurs du système et en extrait des « features » que le moteur de
particules (autre module, « Particles ») consomme à 120+ fps.

CHAINE DE TRAITEMENT (vue d'ensemble) :
  1. Un thread daemon capture en LOOPBACK le flux qui joue sur la sortie
     audio par défaut (Spotify, YouTube, jeu...) — sans micro physique :
       * Windows  -> WASAPI loopback (via la librairie `soundcard`).
       * Linux    -> moniteur PulseAudio/PipeWire (sink ".monitor").
  2. Chaque petit bloc audio (~512 échantillons, < ~11 ms de latence) est
     écrit dans un RING BUFFER numpy sans verrou lourd (le thread de capture
     n'attend jamais le rendu, et inversement).
  3. À chaque appel de get_features() (une fois par frame de rendu), on prend
     la dernière fenêtre `fft_size` du ring buffer, on la fenêtre (Hann) et on
     calcule la FFT réelle EN CPU (numpy.fft.rfft) — quelques µs sur ~4096
     points — puis on uploade la SEULE magnitude sur le GPU. Tout le reste du DSP
     lourd (regroupement en bandes, spectre 512 bins, centroïde) reste sur le
     GPU ; on ne rapatrie vers le CPU que ~10 scalaires. NB : la rfft est faite
     en CPU pour NE PAS embarquer cuFFT (~284 Mo) dans le build standalone.
  4. Le SPECTRE 512 bins reste un tableau CuPy EN VRAM (`spectrum_gpu`) : le
     module de particules le lit directement device-to-device, sans aller-retour
     CPU. C'est le point clé de perf de tout le pipeline.

POURQUOI CALCULER LA FFT DANS get_features() (côté rendu) ET PAS DANS LE
THREAD DE CAPTURE ?
  Le contexte CUDA et les tableaux GPU vivent naturellement dans le thread qui
  fait le rendu (celui qui possède le contexte OpenGL/CuPy). En faisant la FFT
  GPU dans get_features(), `spectrum_gpu` est produit sur le bon thread/contexte,
  prêt à être partagé device-to-device avec OpenGL via CuPy, sans transfert ni
  synchronisation inter-contexte hasardeuse. Le thread de capture, lui, ne fait
  que de l'I/O audio et des écritures numpy CPU triviales -> il ne touche jamais
  le GPU. C'est l'architecture la plus simple et la plus robuste.

CONTRAT PARTAGÉ (implémenté ici À LA LETTRE — d'autres modules en dépendent) :
  class AudioFeatures:
      bass, low_mid, mid, high : float   # lissés, normalisés ~0..1
      amplitude : float                  # RMS 0..1
      beat : float                       # force d'onset 0..1
      is_beat : bool
      centroid : float                   # 0..1 (grave->aigu) pour la teinte
      spectrum_gpu : cupy.ndarray        # float32, (N_SPECTRUM,), 0..1, SUR GPU
      t : float                          # secondes depuis le démarrage
  class AudioEngine:
      N_SPECTRUM = 512
      __init__(self, samplerate=48000, fft_size=4096, blocksize=512)
      start(); stop(); get_features() -> AudioFeatures

DÉPENDANCES (à ajouter au requirements partagé) :
    soundcard           # capture loopback multiplateforme
    cupy-cuda13x        # FFT GPU (RTX 4090, toolkit système CUDA 13.x)
    numpy               # ring buffer + petites opérations CPU

CuPy est OBLIGATOIRE : le contrat exige que `spectrum_gpu` vive sur le GPU.
Si CuPy est absent, AudioEngine.__init__ lève une RuntimeError explicite avec
le conseil d'installation. `soundcard`, lui, peut manquer : on dégrade alors
proprement (features à zéro) pour que le visualiseur tourne quand même.
"""

from __future__ import annotations

import sys
import math
import time
import threading

import numpy as np

# DONNÉE partagée + ANALYSE musicale de haut niveau, désormais DÉCOUPLÉES de ce
# moteur (qui ne fait plus QUE l'acquisition : capture + FFT GPU + réductions).
# On RÉ-EXPORTE AudioFeatures -> `from audio_engine import AudioFeatures` reste
# valide (contrat figé inchangé pour particles/renderer). Les constantes de
# lissage (ATTACK/RELEASE/MAX_DECAY) sont partagées : on les réutilise pour le
# lissage du spectre GPU afin que CPU (analyseur) et GPU (ici) restent calés.
from audio_features import AudioFeatures
from musical_analyzer import MusicalAnalyzer, RawFrame, ATTACK, RELEASE, MAX_DECAY

# ---------------------------------------------------------------------------
# Import de CuPy (OBLIGATOIRE pour ce module).
# On l'importe de façon défensive UNIQUEMENT pour pouvoir donner un message
# d'erreur propre au moment de la construction de l'AudioEngine : le module
# reste importable (utile pour py_compile / CI sans GPU), mais instancier
# AudioEngine sans CuPy lèvera une RuntimeError claire.
# ---------------------------------------------------------------------------
try:
    import cupy as cp            # type: ignore
    _HAS_CUPY = True
    _CUPY_IMPORT_ERROR: Exception | None = None
except Exception as _exc:        # ImportError, ou CuPy présent mais sans CUDA
    cp = None                    # type: ignore
    _HAS_CUPY = False
    _CUPY_IMPORT_ERROR = _exc

# ---------------------------------------------------------------------------
# Import de soundcard (capture loopback). OPTIONNEL : si absent, on dégrade
# vers des features nulles (le visualiseur continue de tourner « à vide »).
# ---------------------------------------------------------------------------
try:
    import soundcard as _sc      # type: ignore
    _HAS_SOUNDCARD = True
    _SOUNDCARD_IMPORT_ERROR: Exception | None = None
except Exception as _exc:
    _sc = None                   # type: ignore
    _HAS_SOUNDCARD = False
    _SOUNDCARD_IMPORT_ERROR = _exc


# ===========================================================================
#  AudioFeatures + _OnsetDetector ont DÉMÉNAGÉ (découplage).
#    - AudioFeatures  -> audio_features.py   (ré-exportée ci-dessus -> contrat ok)
#    - _OnsetDetector -> musical_analyzer.py  (utilisé par RhythmTracker)
#  Ce module ne garde QUE l'acquisition (capture + FFT GPU + réductions).
# ===========================================================================


# ===========================================================================
#  Sélection du périphérique de capture loopback (Windows / Linux / autre).
# ===========================================================================
def _find_loopback_microphone():
    """Renvoie (mic, is_loopback) : le « microphone » qui capte la sortie
    système, et un booléen indiquant si c'est bien un loopback.

    Stratégie :
      * Windows (WASAPI) : on prend le haut-parleur par défaut puis le
        microphone loopback correspondant
        (soundcard.get_microphone(id=str(default_speaker.id),
                                   include_loopback=True)).
        -> C'est exactement l'API recommandée par `soundcard`.
        FALLBACK Windows documenté (NON implémenté ici pour rester sur une
        seule dépendance) : `pyaudiowpatch`, un fork de PyAudio qui expose les
        périphériques WASAPI loopback. On ouvrirait alors le flux loopback via
        pyaudio.PyAudio().get_default_wasapi_loopback() ; même logique de ring
        buffer ensuite.
      * Linux (PulseAudio/PipeWire) : on cherche un micro dont le nom contient
        ".monitor" (le moniteur du sink par défaut).
      * Autres OS : on tente un loopback générique, sinon on signale l'échec.

    En cas d'échec total, renvoie (None, False) : l'appelant basculera alors
    sur le micro par défaut (avertissement) ou sur le mode silencieux.
    """
    if not _HAS_SOUNDCARD:
        return None, False

    plat = sys.platform

    # ---- Windows : WASAPI loopback ----------------------------------------
    if plat.startswith("win"):
        try:
            speaker = _sc.default_speaker()
        except Exception:
            return None, False

        # Tentative directe : le micro loopback partage l'id du haut-parleur.
        try:
            mic = _sc.get_microphone(id=str(speaker.id), include_loopback=True)
            if mic is not None:
                return mic, True
        except Exception:
            pass  # on bascule sur la recherche par nom ci-dessous.

        # Repli : on parcourt les micros loopback et on tente de matcher le nom.
        try:
            loopback_mics = _sc.all_microphones(include_loopback=True)
        except Exception:
            return None, False

        for mic in loopback_mics:
            if getattr(mic, "isloopback", False) and speaker.name in mic.name:
                return mic, True
        for mic in loopback_mics:               # n'importe quel loopback
            if getattr(mic, "isloopback", False):
                return mic, True
        return None, False

    # ---- Linux : moniteur PulseAudio / PipeWire ---------------------------
    elif plat.startswith("linux"):
        try:
            default_name = _sc.default_speaker().name
        except Exception:
            default_name = None
        try:
            all_mics = _sc.all_microphones(include_loopback=True)
        except Exception:
            return None, False

        # a) moniteur correspondant au sink par défaut.
        if default_name:
            for mic in all_mics:
                if ".monitor" in mic.name.lower() and default_name in mic.name:
                    return mic, True
        # b) n'importe quel ".monitor".
        for mic in all_mics:
            if ".monitor" in mic.name.lower():
                return mic, True
        # c) repli sur un loopback générique.
        for mic in all_mics:
            if getattr(mic, "isloopback", False):
                return mic, True
        return None, False

    # ---- Autres OS (macOS, etc.) ------------------------------------------
    else:
        try:
            for mic in _sc.all_microphones(include_loopback=True):
                if getattr(mic, "isloopback", False):
                    return mic, True
        except Exception:
            pass
        return None, False


def _default_microphone():
    """Micro physique par défaut (repli si aucun loopback trouvé)."""
    if not _HAS_SOUNDCARD:
        return None
    try:
        return _sc.default_microphone()
    except Exception:
        return None


# ===========================================================================
#  AudioEngine : capture (thread) + analyse spectrale GPU (à la demande).
# ===========================================================================
class AudioEngine:
    """Moteur de capture loopback + analyse spectrale GPU temps réel.

    Utilisation typique (depuis la boucle de rendu) :
        engine = AudioEngine(samplerate=48000, fft_size=4096, blocksize=512)
        engine.start()
        while running:
            feats = engine.get_features()   # 120+ fps, non bloquant
            ... # piloter les particules avec feats.bass, feats.spectrum_gpu, ...
        engine.stop()
    """

    # Longueur du spectre downsamplé exposé sur GPU (constante du contrat).
    N_SPECTRUM = 512

    # Bornes des 4 bandes d'énergie (Hz). [début, fin[.
    BAND_EDGES = (
        ("bass",    20.0,   150.0),
        ("low_mid", 150.0,  500.0),
        ("mid",     500.0,  2000.0),
        ("high",    2000.0, 16000.0),
    )

    # Bornes log du spectre downsamplé (Hz) : l'oreille perçoit ~log.
    SPECTRUM_FMIN = 30.0
    SPECTRUM_FMAX = 18000.0

    def __init__(
        self,
        samplerate: int = 48000,
        fft_size: int = 4096,
        blocksize: int = 512,
    ):
        """
        :param samplerate: fréquence d'échantillonnage de la capture (Hz).
        :param fft_size:   taille de la FFT (puissance de 2 conseillée, p.ex.
                           2048–4096). Plus grand = meilleure résolution
                           fréquentielle (utile pour les basses) mais un peu
                           plus de lissage temporel.
        :param blocksize:  taille de bloc de capture en échantillons. Petit
                           -> faible latence. 512 @ 48 kHz ~= 10.7 ms.

        Lève RuntimeError si CuPy est indisponible (spectrum_gpu impose le GPU).
        """
        # --- CuPy OBLIGATOIRE : on échoue tôt avec un message actionnable. ---
        if not _HAS_CUPY:
            raise RuntimeError(
                "CuPy est requis par audio_engine (le spectre doit vivre sur "
                "le GPU pour le pipeline de particules), mais son import a "
                f"échoué : {_CUPY_IMPORT_ERROR!r}\n"
                "Installe-le pour ta version de CUDA, par exemple :\n"
                "    pip install cupy-cuda13x      # GPU RTX 4090 -> CUDA 13.x\n"
                "(adapte la roue cupy-cudaXXx a ta version de toolkit)."
            )

        self.samplerate = int(samplerate)
        self.fft_size = int(fft_size)
        self.blocksize = int(blocksize)

        # ------------------------------------------------------------------ #
        #  RING BUFFER CPU (numpy)                                           #
        # ------------------------------------------------------------------ #
        # Le thread de capture ÉCRIT des blocs mono ici ; get_features() LIT
        # la dernière fenêtre fft_size. On dimensionne le buffer à plusieurs
        # fois fft_size pour absorber la jitter d'ordonnancement sans jamais
        # bloquer. La synchro est assurée par un petit verrou tenu très
        # brièvement (juste la copie d'un bloc / d'une fenêtre).
        self._ring_size = max(self.fft_size * 4, self.blocksize * 8)
        self._ring = np.zeros(self._ring_size, dtype=np.float32)
        self._write_pos = 0                  # index d'écriture (mod ring_size)
        self._frames_written = 0             # total d'échantillons jamais écrits
        self._ring_lock = threading.Lock()   # protège ring + positions

        # ------------------------------------------------------------------ #
        #  Verrou des FEATURES publiques                                     #
        # ------------------------------------------------------------------ #
        # get_features() calcule dans le thread appelant (rendu) puis range le
        # résultat ici ; on garde aussi une dernière copie valide à renvoyer si
        # un appel arrive avant le premier bloc audio.
        self._feat_lock = threading.Lock()
        self._last_features = AudioFeatures(
            spectrum_gpu=cp.zeros(self.N_SPECTRUM, dtype=cp.float32)
        )

        # --- Thread de capture ---
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._mic = None
        self._is_loopback = False
        self._t0 = time.perf_counter()
        # Dérivation OPTIONNELLE du flux audio BRUT pour l'enregistrement vidéo
        # (cf. start_recording_tap). None tant qu'aucun enregistrement n'est en
        # cours -> aucun impact sur le chemin get_features (contrat inchangé).
        self._rec_q = None

        # ------------------------------------------------------------------ #
        #  Pré-calcul GPU : fenêtre de Hann + axe fréquentiel + matrices.    #
        # ------------------------------------------------------------------ #
        # Fenêtre de Hann sur fft_size points (réduit les fuites spectrales),
        # gardée en CPU (numpy) : la rfft est calculée côté CPU dans
        # _analyze_window (voir la note là-bas) pour NE PAS embarquer cuFFT
        # (cufft64_*.dll, ~284 Mo) dans le build standalone.
        self._window_cpu = np.hanning(self.fft_size).astype(np.float32)
        # Compensation de gain cohérente (somme de la fenêtre) pour des
        # magnitudes ~indépendantes de fft_size.
        self._win_gain = float(np.sum(self._window_cpu)) or 1.0

        # Axe des fréquences de la rfft : fft_size points -> fft_size//2+1 bins.
        freqs_np = np.fft.rfftfreq(self.fft_size, d=1.0 / self.samplerate)
        self._n_bins = freqs_np.shape[0]
        self._freqs_gpu = cp.asarray(freqs_np.astype(np.float32))  # pour centroïde

        # Matrices de regroupement (pré-calc CPU -> upload VRAM une seule fois).
        # band_matrix (4 x Nbins) : énergie moyenne par bande.
        # spectrum_matrix (N_SPECTRUM x Nbins) : magnitude moyenne par bin log.
        self._band_matrix = cp.asarray(self._build_band_matrix(freqs_np))
        self._spectrum_matrix = cp.asarray(self._build_spectrum_matrix(freqs_np))
        # Analyse HARMONIQUE : matrice chroma 12 classes + profils de tonalité.
        self._build_harmony()

        # ------------------------------------------------------------------ #
        #  PSYCHOACOUSTIQUE : pondération d'ISOSONIE (sonie perçue).          #
        # ------------------------------------------------------------------ #
        # Courbe de pondération A (forme close, normalisée à 1 @ 1 kHz) : approxime la
        # sensibilité fréquentielle de l'oreille (contour d'isosonie ~40 phon de
        # Fletcher-Munson — sourde au sub-grave et aux très aigus, ultra-sensible vers
        # 2–5 kHz). Sert à dériver une SONIE perçue et un ÉQUILIBRE de bandes perçu au
        # lieu de l'énergie physique brute : le préset « Cognitive » ne fait réagir la
        # nuée qu'à ce que l'humain ENTEND vraiment. Poids en ÉNERGIE (A²) pré-uploadé
        # en VRAM -> une réduction pondérée par frame (coût négligeable).
        a_lin = self._a_weighting(freqs_np)              # gain ~sensibilité (1 @ 1 kHz)
        self._loud_w2_gpu = cp.asarray((a_lin * a_lin).astype(np.float32))   # poids énergie
        # Poids perceptuel MOYEN par bande (RMS de l'énergie A-pondérée), normalisé à 1
        # sur la bande la + sensible -> donne l'équilibre de bandes tel que l'oreille
        # le perçoit (le grave pèse bien moins que sa seule énergie ne le dirait).
        a2 = (a_lin * a_lin).astype(np.float64)
        pb = np.zeros(4, dtype=np.float64)
        for i, (_n, f_lo, f_hi) in enumerate(self.BAND_EDGES):
            m = (freqs_np >= f_lo) & (freqs_np < f_hi)
            pb[i] = math.sqrt(a2[m].mean()) if m.any() else 0.0
        self._perc_band_w = (pb / (pb.max() + 1e-12)).astype(np.float32)      # (4,) max=1

        # BASSE PROFONDE (sub ~20–60 Hz) : poids de réduction (moyenne sur les bins du
        # sub), pré-uploadé en VRAM. NON pondéré isosonie -> c'est la force RESSENTIE.
        sub_mask = (freqs_np >= 20.0) & (freqs_np < 60.0)
        sub_w = np.zeros_like(freqs_np, dtype=np.float32)
        if sub_mask.any():
            sub_w[sub_mask] = 1.0 / float(sub_mask.sum())
        self._sub_w_gpu = cp.asarray(sub_w)

        # ------------------------------------------------------------------ #
        #  ANALYSEUR MUSICAL (sémantique de haut niveau) — DÉCOUPLÉ.          #
        # ------------------------------------------------------------------ #
        # Tout le suivi de bandes/amplitude + beat + onsets + tempo + phrase +
        # harmonie + perceptuel vit désormais dans MusicalAnalyzer (musical_analyzer.py)
        # : testable sans micro. On lui injecte les poids d'isosonie par bande (CPU)
        # calculés ci-dessus. Le moteur ne garde que l'acquisition + le spectre GPU.
        self._analyzer = MusicalAnalyzer(self._perc_band_w,
                                         self.SPECTRUM_FMIN, self.SPECTRUM_FMAX)

        # Lissage + peak-hold du spectre, ENTIÈREMENT sur GPU (persistant).
        self._env_spectrum_gpu = cp.zeros(self.N_SPECTRUM, dtype=cp.float32)
        self._spectrum_max_gpu = cp.full(self.N_SPECTRUM, 1e-3, dtype=cp.float32)
        # Tampon de SORTIE GPU réutilisé : on écrit le spectre normalisé dedans
        # puis on en publie une COPIE par frame (pas de course avec la frame
        # suivante). 512 floats -> copie négligeable.
        self._spectrum_out_gpu = cp.zeros(self.N_SPECTRUM, dtype=cp.float32)

        # (Beat, onsets, oscillateur de tempo/phase et traqueur de phrase ont
        #  déménagé dans MusicalAnalyzer.RhythmTracker — cf. self._analyzer.)

        # Cadence d'analyse : on évite de relancer une FFT plus souvent que
        # nécessaire (si get_features est appelé à 240 fps alors qu'un nouveau
        # bloc audio n'arrive que toutes les ~10 ms, on réutilise le dernier
        # résultat). On force une FFT dès qu'au moins blocksize/2 nouveaux
        # échantillons sont disponibles.
        self._last_analyzed_frame = -1
        self._min_new_frames = max(1, self.blocksize // 2)

    # ------------------------------------------------------------------ #
    #  Construction des matrices de regroupement (CPU, une seule fois).  #
    # ------------------------------------------------------------------ #
    def _build_band_matrix(self, freqs: np.ndarray) -> np.ndarray:
        """Matrice (4 x Nbins) : moyenne d'énergie (puissance) par bande.

        Chaque ligne porte des poids 1/N sur les bins de la bande, 0 ailleurs.
        band_matrix @ puissance -> énergie moyenne par bande (on prendra sqrt
        ensuite pour une RMS).
        """
        mat = np.zeros((4, freqs.shape[0]), dtype=np.float32)
        for i, (_name, f_lo, f_hi) in enumerate(self.BAND_EDGES):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            n = int(mask.sum())
            if n > 0:
                mat[i, mask] = 1.0 / n
        return mat

    def _build_spectrum_matrix(self, freqs: np.ndarray) -> np.ndarray:
        """Matrice (N_SPECTRUM x Nbins) : magnitude moyenne par bin log-espacé.

        On découpe [SPECTRUM_FMIN, SPECTRUM_FMAX] en N_SPECTRUM intervalles
        répartis logarithmiquement. Chaque ligne moyenne les bins FFT tombant
        dans l'intervalle. Pour les bins log plus étroits que la résolution FFT
        (bas du spectre), on rattache le bin FFT le plus proche -> pas de trou.
        """
        edges = np.logspace(
            np.log10(self.SPECTRUM_FMIN),
            np.log10(self.SPECTRUM_FMAX),
            self.N_SPECTRUM + 1,
        )
        mat = np.zeros((self.N_SPECTRUM, freqs.shape[0]), dtype=np.float32)
        for i in range(self.N_SPECTRUM):
            f_lo, f_hi = edges[i], edges[i + 1]
            mask = (freqs >= f_lo) & (freqs < f_hi)
            n = int(mask.sum())
            if n > 0:
                mat[i, mask] = 1.0 / n
            else:
                idx = int(np.argmin(np.abs(freqs - 0.5 * (f_lo + f_hi))))
                mat[i, idx] = 1.0
        return mat

    @staticmethod
    def _a_weighting(freqs: np.ndarray) -> np.ndarray:
        """Pondération A (IEC 61672, forme close), en gain LINÉAIRE normalisé à 1 @
        1 kHz. Approxime la sensibilité de l'oreille (contour d'isosonie ~40 phon) :
        ~0.10 (−20 dB) vers 50 Hz, pic ~1.0 vers 2–3 kHz, redescend dans l'extrême
        aigu. Le bin DC (f=0) reçoit un poids nul (numérateur ∝ f⁴)."""
        f = np.asarray(freqs, dtype=np.float64)
        f2 = f * f
        c1, c2, c3, c4 = 20.6 ** 2, 107.7 ** 2, 737.9 ** 2, 12194.0 ** 2
        num = (c4 * f2 * f2)
        den = (f2 + c1) * np.sqrt((f2 + c2) * (f2 + c3)) * (f2 + c4) + 1e-30
        ra = num / den
        fk = 1000.0 ** 2                          # normalisation à 1 kHz
        ra_1k = (c4 * fk * fk) / ((fk + c1) * math.sqrt((fk + c2) * (fk + c3)) * (fk + c4))
        return (ra / (ra_1k + 1e-30)).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Cycle de vie : start / stop                                       #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Résout le périphérique loopback et lance le thread de capture.

        Idempotent : un second appel ne fait rien. Ne lève PAS si l'audio est
        indisponible -> on bascule en mode silencieux (features à zéro) pour ne
        jamais empêcher le visualiseur de tourner.
        """
        if self._running.is_set():
            return  # déjà démarré

        self._t0 = time.perf_counter()

        # Résolution du périphérique de capture (loopback de préférence).
        self._mic, self._is_loopback = _find_loopback_microphone()

        if self._mic is None:
            # Pas de loopback -> on tente le micro physique par défaut.
            self._mic = _default_microphone()
            self._is_loopback = False
            if self._mic is None:
                if not _HAS_SOUNDCARD:
                    print(
                        "[audio_engine] AVERTISSEMENT : la librairie 'soundcard' "
                        f"est introuvable ({_SOUNDCARD_IMPORT_ERROR!r}). "
                        "Aucune capture -> features à zéro. "
                        "Installe-la avec : pip install soundcard",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "[audio_engine] AVERTISSEMENT : aucun périphérique de "
                        "capture (loopback ni micro) trouvé. "
                        "Le visualiseur tourne mais sans réaction audio.",
                        file=sys.stderr,
                    )
                return  # mode silencieux : on ne démarre même pas le thread.
            else:
                print(
                    "[audio_engine] AVERTISSEMENT : loopback indisponible, repli "
                    f"sur le MICRO par défaut « {self._mic.name} ». "
                    "Le visualiseur réagira au micro, pas au son système.",
                    file=sys.stderr,
                )
        else:
            print(f"[audio_engine] Capture loopback : « {self._mic.name} »",
                  file=sys.stderr)

        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop, name="AudioCaptureThread", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Arrête proprement le thread de capture. Sûr même si start() n'a
        jamais été appelé, ou appelé deux fois."""
        self._running.clear()
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)    # thread daemon -> jamais bloquant fatal
        self._thread = None

    # ------------------------------------------------------------------ #
    #  THREAD DE CAPTURE (daemon) : I/O audio + écriture ring buffer.    #
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        """Boucle robuste : ne lève jamais vers l'extérieur, ne bloque jamais
        le rendu. Ce thread ne touche PAS le GPU : il ne fait que capturer des
        blocs audio et les écrire (downmixés mono) dans le ring buffer."""
        # soundcard émet un SoundcardRuntimeWarning « data discontinuity in
        # recording » à chaque trou WASAPI (fréquent en loopback, surtout sur les
        # passages silencieux). C'est bénin, mais à 60+ fps ça noierait stderr :
        # on filtre CE message précis (les autres warnings passent normalement).
        import warnings
        warnings.filterwarnings("ignore", message="data discontinuity in recording")
        try:
            with self._mic.recorder(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=None,     # None = nombre natif de canaux
            ) as rec:
                while self._running.is_set():
                    try:
                        block = rec.record(numframes=self.blocksize)
                    except Exception as exc:
                        print(f"[audio_engine] erreur record(): {exc}",
                              file=sys.stderr)
                        time.sleep(0.005)   # évite une boucle d'erreur à 100% CPU
                        continue
                    self._push_block(block)
        except Exception as exc:
            # Échec fatal d'ouverture du flux : on log et on s'arrête
            # proprement (mode silencieux) plutôt que de crasher le visualiseur.
            print(f"[audio_engine] flux audio interrompu : {exc}",
                  file=sys.stderr)
            self._running.clear()

    def _push_block(self, block: np.ndarray) -> None:
        """Downmix mono + écriture dans le ring buffer (thread de capture)."""
        # --- Dérivation enregistrement (AVANT downmix : on conserve les canaux
        #     NATIFS, p.ex. stéréo). Non bloquant : si la file est pleine on
        #     saute le bloc -> la capture (et donc get_features) n'est JAMAIS
        #     ralentie par l'enregistrement.
        q = self._rec_q
        if q is not None:
            try:
                q.put_nowait(np.ascontiguousarray(block, dtype=np.float32))
            except Exception:
                pass

        # Downmix mono (moyenne des canaux) : énergie globale, et on évite de
        # garder plusieurs canaux dans le ring.
        if block.ndim == 2 and block.shape[1] > 1:
            mono = block.mean(axis=1).astype(np.float32, copy=False)
        else:
            mono = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)

        n = mono.shape[0]
        if n == 0:
            return

        with self._ring_lock:
            # Écriture circulaire (gère le wrap-around en deux tronçons).
            end = self._write_pos + n
            if end <= self._ring_size:
                self._ring[self._write_pos:end] = mono
            else:
                first = self._ring_size - self._write_pos
                self._ring[self._write_pos:] = mono[:first]
                self._ring[: n - first] = mono[first:]
            self._write_pos = end % self._ring_size
            self._frames_written += n

    # ------------------------------------------------------------------ #
    #  Dérivation d'enregistrement (audio BRUT pour le muxage vidéo).     #
    #  AJOUT additif : n'affecte ni AudioFeatures ni get_features().      #
    # ------------------------------------------------------------------ #
    def start_recording_tap(self, maxsize: int = 256):
        """Active une dérivation du flux audio BRUT (canaux NATIFS, au samplerate
        de capture `self.samplerate`) et renvoie une `queue.Queue` thread-safe où
        le thread de capture pousse une COPIE de chaque bloc. Le Recorder la
        draine pour muxer une piste audio synchronisée. `stop_recording_tap()`
        pour arrêter."""
        import queue
        q = queue.Queue(maxsize=int(max(8, maxsize)))
        self._rec_q = q
        return q

    def stop_recording_tap(self) -> None:
        """Désactive la dérivation (le thread de capture cesse d'alimenter la
        file ; les blocs déjà présents restent drainables par le Recorder)."""
        self._rec_q = None

    def _read_latest_window(self) -> tuple[np.ndarray | None, int]:
        """Lit les `fft_size` derniers échantillons du ring buffer.

        Renvoie (fenêtre_contigue, frames_written_au_moment_de_la_lecture).
        Renvoie (None, frames_written) tant qu'il n'y a pas assez de données.
        """
        with self._ring_lock:
            written = self._frames_written
            if written < self.fft_size:
                return None, written
            # La fenêtre se termine à write_pos (dernier échantillon écrit).
            start = (self._write_pos - self.fft_size) % self._ring_size
            end = start + self.fft_size
            if end <= self._ring_size:
                win = self._ring[start:end].copy()
            else:
                first = self._ring_size - start
                win = np.empty(self.fft_size, dtype=np.float32)
                win[:first] = self._ring[start:]
                win[first:] = self._ring[: self.fft_size - first]
        return win, written

    # ------------------------------------------------------------------ #
    #  get_features() : appelé une fois par frame de rendu.              #
    # ------------------------------------------------------------------ #
    def get_features(self) -> AudioFeatures:
        """Renvoie les dernières features audio. Lance la FFT GPU si de
        nouvelles données sont disponibles, sinon renvoie le dernier résultat.

        Conçu pour être appelé à 120+ fps depuis la boucle de rendu :
          * si moins de `blocksize/2` nouveaux échantillons -> on réutilise le
            dernier résultat (zéro travail GPU) ;
          * sinon -> une seule rfft CPU + quelques produits matriciels GPU (< 2 ms
            sur une 4090).

        Si l'audio est indisponible / silencieux, renvoie des features nulles
        (avec un spectrum_gpu de zéros valide) : le visualiseur tourne quand
        même.
        """
        win, written = self._read_latest_window()

        # Pas (encore) assez de données -> dernier instantané connu (zéros au
        # tout début). On met juste `t` à jour pour que l'animation avance.
        if win is None:
            with self._feat_lock:
                feats = self._last_features
            feats.t = time.perf_counter() - self._t0
            self._analyzer.apply_tempo(feats, feats.t, False)  # phase continue (pas de beat frais)
            return feats

        # Anti-redondance : si aucun bloc neuf significatif depuis la dernière
        # analyse, on renvoie le dernier résultat (économise du temps GPU).
        if (written - self._last_analyzed_frame) < self._min_new_frames \
                and self._last_analyzed_frame >= 0:
            with self._feat_lock:
                feats = self._last_features
            feats.t = time.perf_counter() - self._t0
            self._analyzer.apply_tempo(feats, feats.t, False)  # phase continue (snapshot recyclé)
            return feats
        self._last_analyzed_frame = written

        # --- Réductions GPU (ici) -> sémantique musicale (analyseur) -------------
        # `now` est calculé UNE fois et injecté : timing unifié sur la frame (au lieu
        # de 3 lectures perf_counter dispersées) + déterminisme pour les tests.
        now = time.perf_counter() - self._t0
        feats = self._analyze_window(win, now)
        feats.t = now
        # Expose la forme d'onde brute (GPU) + le compteur d'échantillons : la
        # « donnée » que le module de particules plonge en retards pour
        # reconstruire l'attracteur du son (device-to-device, pas de copie CPU).
        feats.waveform_gpu = cp.asarray(win, dtype=cp.float32)
        feats.samples_written = int(written)
        # Oscillateur de tempo : corrige période/phase sur ce battement frais.
        self._analyzer.apply_tempo(feats, now, bool(feats.is_beat))
        with self._feat_lock:
            self._last_features = feats
        return feats

    # ------------------------------------------------------------------ #
    #  Cœur DSP : analyse d'une fenêtre fft_size -> AudioFeatures.       #
    #  Tout le lourd est sur GPU ; on ne rapatrie que ~10 scalaires.     #
    # ------------------------------------------------------------------ #
    def _analyze_window(self, win: np.ndarray, now: float) -> AudioFeatures:
        """ACQUISITION d'une frame : RMS + FFT + réductions pondérées (GPU). Produit
        une RawFrame que le MusicalAnalyzer interprète (sémantique musicale, CPU).
        `now` (s) est INJECTÉ -> timing unifié sur la frame + déterminisme pour les
        tests. Tout le lourd reste sur GPU ; on ne rapatrie que ~7 scalaires + chroma."""
        # RMS temporel (amplitude brute) — petit calcul CPU avant l'upload.
        block_rms = float(np.sqrt(np.mean(win.astype(np.float64) ** 2)) + 1e-12)

        # Silence quasi total : l'analyseur décroît ses enveloppes ; on lui fournit un
        # spectrum_gpu valide (zéros lissés) -> on ne casse pas le consommateur GPU.
        if block_rms < 1e-5:
            return self._analyzer.silent(self._decay_spectrum_silent())

        # --- FFT réelle + magnitude (CPU/numpy), PUIS upload de la magnitude ---
        # La rfft d'une fenêtre de fft_size points (~4096) coûte quelques µs en
        # CPU et évite d'embarquer cuFFT (cufft64_*.dll, ~284 Mo) dans le build
        # standalone. On n'uploade QUE la magnitude (Nbins floats, quelques Ko) :
        # le DSP lourd (réductions pondérées, spectre 512) reste sur GPU et
        # `spectrum_gpu` est bien produit en VRAM — le contrat est inchangé.
        win_w = win * self._window_cpu                   # fenêtrage Hann (CPU)
        mag = np.abs(np.fft.rfft(win_w)).astype(np.float32)
        mag *= np.float32(2.0 / self._win_gain)          # magnitude normalisée
        g_mag = cp.asarray(mag)                          # (Nbins,) -> VRAM
        g_pow = g_mag * g_mag                            # puissance (pour bandes)

        # --- Énergie RMS par bande : somme pondérée (4 x Nbins) ---
        # On évite l'opérateur @ (qui chargerait cuBLAS + cuBLASLt, ~500 Mo) :
        # une multiplication DIFFUSÉE + réduction donne le même résultat via des
        # kernels NVRTC, sur des matrices minuscules -> coût négligeable.
        g_band_rms = cp.sqrt(
            cp.sum(self._band_matrix * g_pow[None, :], axis=1) + 1e-12)   # (4,)

        # --- SONIE perçue : niveau large bande A-PONDÉRÉ (isosonie). Même réduction
        # diffusée que les bandes (pas de @, donc pas de cuBLAS) -> 1 scalaire. ---
        g_loud = cp.sqrt(cp.sum(self._loud_w2_gpu * g_pow) + 1e-12)
        # --- BASSE PROFONDE ressentie : énergie du sub (20–60 Hz), SANS isosonie. ---
        g_sub = cp.sqrt(cp.sum(self._sub_w_gpu * g_pow) + 1e-12)

        # --- Centroïde spectral (barycentre des fréquences pondéré par mag) --
        # centroid_hz = sum(f * mag) / sum(mag). Reste sur GPU.
        mag_sum = cp.sum(g_mag) + 1e-9
        g_centroid_hz = cp.sum(self._freqs_gpu * g_mag) / mag_sum

        # --- Spectre downsamplé 512 bins (magnitude moyenne par bin log) ---
        # Même raison qu'au-dessus : somme pondérée diffusée plutôt que @, pour
        # NE PAS charger cuBLAS. (512 x Nbins) -> (N_SPECTRUM,), reste en VRAM.
        g_spec = cp.sum(self._spectrum_matrix * g_mag[None, :], axis=1)
        # Échelle « musicale » : compression douce (racine) pour densifier le
        # bas niveau visuellement, sans le coût d'un log complet.
        g_spec = cp.sqrt(g_spec + 1e-12)

        # --- DIAPASON : estime l'accordage global et re-centre le repliement chroma
        #     dessus (invariance au diapason : La=432, vinyle ralenti…). Lent, gardé.
        self._update_tuning(g_spec)
        # --- CHROMA 12 classes (réduction GPU) -> corrélé par l'analyseur (HarmonyAnalyzer).
        g_chroma = cp.sum(self._chroma_matrix_gpu * g_spec[None, :], axis=1)   # (12,)

        # --- Spectre GPU : normalisation adaptative + lissage, EN VRAM (jamais CPU). ---
        spectrum_gpu_frame = self._smooth_spectrum(g_spec)

        # ---- Frame BRUTE (scalaires CPU + chroma + spectre GPU) -> ANALYSEUR ----
        # On ne rapatrie que ~7 scalaires + le vecteur chroma (12) ; la sémantique
        # musicale (normalisation, beat, onsets, tempo, phrase, harmonie, perceptuel)
        # vit dans MusicalAnalyzer.analyze (CPU, testable). spectrum_gpu reste en VRAM.
        raw = RawFrame(
            band_rms=cp.asnumpy(g_band_rms).astype(np.float32),     # (4,)
            block_rms=block_rms,
            centroid_hz=float(cp.asnumpy(g_centroid_hz)),
            loud_lin=float(cp.asnumpy(g_loud)),                     # niveau A-pondéré
            sub_lin=float(cp.asnumpy(g_sub)),                       # niveau sub-grave
            chroma=cp.asnumpy(g_chroma).astype(np.float64),
            spectrum_gpu=spectrum_gpu_frame,
        )
        return self._analyzer.analyze(raw, now)

    # ------------------------------------------------------------------ #
    #  Lissage du SPECTRE 512 bins (GPU). Reste dans le moteur car c'est du
    #  travail GPU sur le thread de rendu (jamais rapatrié côté CPU).
    # ------------------------------------------------------------------ #
    def _smooth_spectrum(self, g_spec):
        """Normalisation adaptative (max glissant) + lissage (attaque/release) du
        spectre 512, ENTIÈREMENT en VRAM. Renvoie une COPIE indépendante par frame
        (le consommateur la lit pendant que la frame suivante se calcule)."""
        self._spectrum_max_gpu *= MAX_DECAY
        cp.maximum(self._spectrum_max_gpu, g_spec, out=self._spectrum_max_gpu)
        cp.maximum(self._spectrum_max_gpu, 1e-3, out=self._spectrum_max_gpu)
        norm_spec = g_spec / self._spectrum_max_gpu      # (N_SPECTRUM,) ~0..1
        up = norm_spec > self._env_spectrum_gpu
        coeff = cp.where(up, np.float32(ATTACK), np.float32(RELEASE))
        self._env_spectrum_gpu = (
            self._env_spectrum_gpu + coeff * (norm_spec - self._env_spectrum_gpu)
        )
        cp.clip(self._env_spectrum_gpu, 0.0, 1.0, out=self._spectrum_out_gpu)
        return self._spectrum_out_gpu.copy()

    def _decay_spectrum_silent(self):
        """Cas silence : le spectre GPU décroît doucement vers 0 (toujours valide en
        VRAM). Renvoie la copie par frame fournie à MusicalAnalyzer.silent()."""
        self._env_spectrum_gpu *= np.float32(0.85)
        self._spectrum_max_gpu *= MAX_DECAY
        cp.clip(self._env_spectrum_gpu, 0.0, 1.0, out=self._spectrum_out_gpu)
        return self._spectrum_out_gpu.copy()

    # (Le suivi de bandes/amplitude (AGC + enveloppe) et la détection de beat ont
    #  déménagé dans MusicalAnalyzer ; cf. self._analyzer.)

    # ------------------------------------------------------------------ #
    #  Analyse HARMONIQUE : chroma 12 classes + diapason (GPU). La corrélation
    #  tonale (Krumhansl -> warmth/key) vit dans MusicalAnalyzer.HarmonyAnalyzer.
    # ------------------------------------------------------------------ #
    def _build_harmony(self):
        """Pré-calcule la matrice chroma (12 x N_SPECTRUM) + l'estimateur de diapason
        (GPU). La corrélation aux profils Krumhansl vit dans HarmonyAnalyzer."""
        # Centres (Hz) des N_SPECTRUM bins log du spectre downsamplé.
        edges = np.logspace(np.log10(self.SPECTRUM_FMIN),
                            np.log10(self.SPECTRUM_FMAX), self.N_SPECTRUM + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        # Chroma : chaque bin de la PLAGE HARMONIQUE contribue à sa classe de hauteur
        # (pc = round(12·log2(f/C0) − τ) % 12, τ = offset de diapason estimé, cf.
        # _update_tuning). On exclut le très grave et les aigus (cymbales/bruit) qui
        # brouilleraient l'estimation tonale.
        HARM_FMIN, HARM_FMAX, C0 = 55.0, 2000.0, 16.351597831287414
        self._C0_base = C0
        # Position réelle (demi-tons, grille A=440) de chaque bin + masque harmonique.
        self._bin_semitone = 12.0 * np.log2(np.maximum(centers, 1e-6) / C0)
        self._harm_mask = (centers >= HARM_FMIN) & (centers <= HARM_FMAX)
        # --- Estimateur de DIAPASON (invariance à l'accordage) -------------------
        # Écart de chaque bin au demi-ton le plus proche (grille 440), dans [-0.5,0.5[.
        # La moyenne CIRCULAIRE de ces écarts PONDÉRÉE PAR L'ÉNERGIE donne l'offset de
        # diapason global (p.ex. −0.318 demi-ton pour un morceau en La=432). On
        # pré-calcule les phaseurs cos/sin (période = 1 demi-ton) + le masque en VRAM
        # pour une réduction pondérée quasi gratuite par frame (cf. _update_tuning).
        _frac = self._bin_semitone - np.round(self._bin_semitone)         # [-0.5, 0.5]
        _two_pi = 2.0 * np.pi
        self._tune_cos_gpu = cp.asarray(
            (np.cos(_two_pi * _frac) * self._harm_mask).astype(np.float32))
        self._tune_sin_gpu = cp.asarray(
            (np.sin(_two_pi * _frac) * self._harm_mask).astype(np.float32))
        self._harm_mask_gpu = cp.asarray(self._harm_mask.astype(np.float32))
        # Réglages de l'estimateur (le diapason est CONSTANT sur un morceau -> lent).
        self._TUNE_R_MIN = 0.30      # concentration mini (contenu nettement tonal) pour ajuster
        self._TUNE_ALPHA = 0.02      # lissage circulaire (lent)
        self._TUNE_REBUILD = 0.02    # re-replie la chroma si l'offset bouge de >0.02 demi-ton
        # État du diapason : résultante lissée sur le cercle (angle -> offset demi-tons).
        self._tune_zr, self._tune_zi = 0.0, 0.0
        self._tuning_semitones = 0.0
        self._chroma_tau_applied = 0.0
        # Matrice chroma initiale (τ = 0 -> grille A=440 standard).
        self._rebuild_chroma_matrix(0.0)
        # (Les profils Krumhansl + la corrélation tonale (warmth/key) ont déménagé
        #  dans MusicalAnalyzer.HarmonyAnalyzer. Ce moteur ne produit QUE le vecteur
        #  chroma 12 classes — réduction GPU — que l'analyseur corrèle ensuite.)

    def _rebuild_chroma_matrix(self, tau):
        """(Re)construit la matrice chroma 12xN : chaque bin de la plage harmonique
        est replié vers sa classe de hauteur la plus proche, sur une grille DÉCALÉE
        de `tau` demi-tons (l'offset de diapason estimé). tau = 0 => grille A=440."""
        s = self._bin_semitone
        pc = np.mod(np.rint(s - tau).astype(np.int64), 12)
        cm = np.zeros((12, self.N_SPECTRUM), dtype=np.float32)
        idx = np.where(self._harm_mask)[0]
        cm[pc[idx], idx] = 1.0
        self._chroma_matrix_gpu = cp.asarray(cm)
        self._chroma_tau_applied = float(tau)

    def _update_tuning(self, g_spec):
        """Estime le DIAPASON global (offset en demi-tons vs A=440) et re-centre le
        repliement chroma dessus -> reconnaissance tonale INVARIANTE AU DIAPASON
        (La=432, vinyle légèrement ralenti…) tant que l'écart reste < ~50 cents (un
        quart de ton ; au-delà, le demi-ton « le plus proche » bascule et l'accordage
        devient intrinsèquement ambigu — limite de toute méthode chroma).

        Méthode : moyenne CIRCULAIRE, pondérée par l'énergie, des écarts bin->demi-ton
        le plus proche sur la plage harmonique — l'analogue léger de
        librosa.estimate_tuning. On regarde TOUT le spectre (pas une crête isolée à
        440, fragile). Gating par la concentration R (> _TUNE_R_MIN) : on n'ajuste que
        sur un contenu nettement tonal, jamais sur la percussion/le bruit. Lissé TRÈS
        lentement (le diapason est constant sur un morceau)."""
        cw = float(cp.sum(g_spec * self._tune_cos_gpu))
        sw = float(cp.sum(g_spec * self._tune_sin_gpu))
        w = float(cp.sum(g_spec * self._harm_mask_gpu)) + 1e-9
        cw /= w
        sw /= w
        R = (cw * cw + sw * sw) ** 0.5                    # concentration 0..1
        if R > self._TUNE_R_MIN:
            a = self._TUNE_ALPHA
            self._tune_zr += a * (cw - self._tune_zr)
            self._tune_zi += a * (sw - self._tune_zi)
            self._tuning_semitones = float(
                np.arctan2(self._tune_zi, self._tune_zr)) / (2.0 * np.pi)
            # Re-centre le repliement uniquement si le diapason a bougé sensiblement.
            if abs(self._tuning_semitones - self._chroma_tau_applied) > self._TUNE_REBUILD:
                self._rebuild_chroma_matrix(self._tuning_semitones)

    # (La corrélation tonale (_update_harmony), le traqueur de phrase
    #  (_phrase_detect), l'oscillateur de tempo (_apply_tempo) et le chemin
    #  « silence » (_silent_features) ont déménagé dans musical_analyzer.py
    #  (HarmonyAnalyzer / RhythmTracker / MusicalAnalyzer). Le moteur ne fait
    #  plus QUE l'acquisition + le lissage du spectre GPU.)


# ===========================================================================
#  Bloc de test isolé (lance de la musique sur ta sortie par défaut !).
# ===========================================================================
if __name__ == "__main__":
    print("=== Test audio_engine ===")
    print(f"CuPy (GPU)  : {'OUI' if _HAS_CUPY else 'NON (AudioEngine lèvera)'}")
    print(f"soundcard   : {'OUI' if _HAS_SOUNDCARD else 'NON (mode silencieux)'}")

    try:
        engine = AudioEngine(samplerate=48000, fft_size=4096, blocksize=512)
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}")
        sys.exit(1)

    engine.start()
    print("Capture en cours pendant 8 s... (joue du son sur ta sortie)\n")

    t0 = time.time()
    try:
        while time.time() - t0 < 8.0:
            f = engine.get_features()
            bar = lambda v: "#" * int(max(0.0, min(v, 1.5)) * 20)
            beat_flag = "BEAT" if f.is_beat else "    "
            sys.stdout.write(
                f"\rt={f.t:5.1f}s bass {f.bass:4.2f} low {f.low_mid:4.2f} "
                f"mid {f.mid:4.2f} high {f.high:4.2f} amp {f.amplitude:4.2f} "
                f"cen {f.centroid:4.2f} beat {f.beat:4.2f} {beat_flag} "
                f"| {bar(f.bass):<20}"
            )
            sys.stdout.flush()
            time.sleep(1.0 / 60.0)   # simule ~60 fps de rendu
    except KeyboardInterrupt:
        pass
    finally:
        print("\nArrêt...")
        engine.stop()
        print("Terminé.")
