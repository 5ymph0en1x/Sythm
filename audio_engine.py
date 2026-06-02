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
     calcule la FFT réelle SUR LE GPU avec CuPy (cupy.fft.rfft). Tout le DSP
     lourd (magnitude, regroupement en bandes, spectre 512 bins, centroïde)
     reste sur le GPU ; on ne rapatrie vers le CPU que ~10 scalaires.
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
import time
import threading
from dataclasses import dataclass, field

import numpy as np

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
#  AudioFeatures : photo instantanée des caractéristiques audio.
# ===========================================================================
@dataclass
class AudioFeatures:
    """Instantané thread-safe des features audio pour UNE frame de rendu.

    Tous les scalaires sont des floats Python (lissés, normalisés ~0..1).
    Le seul champ GPU est `spectrum_gpu` : un tableau CuPy float32 de longueur
    AudioEngine.N_SPECTRUM, normalisé 0..1, qui RESTE EN VRAM pour être lu
    directement par le module de particules (device-to-device, pas de
    rapatriement CPU).

    Voir le contrat partagé en tête de fichier : noms et unités sont figés.
    """
    bass: float = 0.0          # 20–150 Hz     (lissé, 0..1)
    low_mid: float = 0.0       # 150–500 Hz    (lissé, 0..1)
    mid: float = 0.0           # 500–2000 Hz   (lissé, 0..1)
    high: float = 0.0          # 2000–16000 Hz (lissé, 0..1)
    amplitude: float = 0.0     # RMS global    (0..1)
    beat: float = 0.0          # force d'onset (0..1, décroît dans le temps)
    is_beat: bool = False      # True sur la frame où un onset est détecté
    centroid: float = 0.0      # centroïde spectral normalisé 0..1 (grave->aigu)
    # Spectre downsamplé EN VRAM (CuPy float32, (N_SPECTRUM,), 0..1).
    # default_factory=None -> rempli par AudioEngine ; reste None si GPU absent.
    spectrum_gpu: "cp.ndarray | None" = None
    t: float = 0.0             # secondes depuis AudioEngine.start()
    # --- EXTENSIONS (non-cassantes) : la forme d'onde brute récente, « donnée »
    # source pour une reconstruction d'attracteur par plongement (cf. particles).
    waveform_gpu: "cp.ndarray | None" = None   # derniers fft_size échantillons bruts (GPU)
    samples_written: int = 0                   # total d'échantillons capturés (compteur)


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

        # ------------------------------------------------------------------ #
        #  Pré-calcul GPU : fenêtre de Hann + axe fréquentiel + matrices.    #
        # ------------------------------------------------------------------ #
        # Fenêtre de Hann sur fft_size points (réduit les fuites spectrales),
        # pré-chargée en VRAM une fois pour toutes.
        self._window = cp.asarray(
            np.hanning(self.fft_size).astype(np.float32)
        )
        # Compensation de gain cohérente (somme de la fenêtre) pour des
        # magnitudes ~indépendantes de fft_size.
        self._win_gain = float(np.sum(np.hanning(self.fft_size))) or 1.0

        # Axe des fréquences de la rfft : fft_size points -> fft_size//2+1 bins.
        freqs_np = np.fft.rfftfreq(self.fft_size, d=1.0 / self.samplerate)
        self._n_bins = freqs_np.shape[0]
        self._freqs_gpu = cp.asarray(freqs_np.astype(np.float32))  # pour centroïde

        # Matrices de regroupement (pré-calc CPU -> upload VRAM une seule fois).
        # band_matrix (4 x Nbins) : énergie moyenne par bande.
        # spectrum_matrix (N_SPECTRUM x Nbins) : magnitude moyenne par bin log.
        self._band_matrix = cp.asarray(self._build_band_matrix(freqs_np))
        self._spectrum_matrix = cp.asarray(self._build_spectrum_matrix(freqs_np))

        # ------------------------------------------------------------------ #
        #  État de lissage / normalisation (CPU, scalaires) — persistant     #
        #  entre frames. Tout ceci ne vit QUE dans le thread de rendu.       #
        # ------------------------------------------------------------------ #
        # Enveloppes lissées des 4 bandes + amplitude (attaque rapide /
        # release lent -> rendu fluide mais réactif).
        self._env_bands = np.zeros(4, dtype=np.float32)
        self._env_amp = 0.0
        self._attack = 0.6     # 0..1 : plus grand = montée plus rapide
        self._release = 0.10   # 0..1 : plus petit = descente plus lente

        # Normalisation adaptative (AGC doux) : max glissant par bande + ampl.
        self._running_max_bands = np.full(4, 1e-3, dtype=np.float32)
        self._running_max_amp = 1e-3
        self._max_decay = 0.9995   # le max glissant redescend lentement

        # Lissage + peak-hold du spectre, ENTIÈREMENT sur GPU (persistant).
        self._env_spectrum_gpu = cp.zeros(self.N_SPECTRUM, dtype=cp.float32)
        self._spectrum_max_gpu = cp.full(self.N_SPECTRUM, 1e-3, dtype=cp.float32)
        # Tampon de SORTIE GPU réutilisé : on écrit le spectre normalisé dedans
        # puis on en publie une COPIE par frame (pas de course avec la frame
        # suivante). 512 floats -> copie négligeable.
        self._spectrum_out_gpu = cp.zeros(self.N_SPECTRUM, dtype=cp.float32)

        # Détection de beat (flux spectral sur bass+low_mid, seuil adaptatif).
        self._prev_lowband_mag = np.zeros(2, dtype=np.float32)  # [bass, low_mid]
        self._flux_avg = 1e-6     # moyenne glissante du flux (seuil adaptatif)
        self._flux_var = 1e-6     # variance glissante (seuil = moy + k*ecart-type)
        self._beat_env = 0.0      # impulsion de beat qui décroît
        self._beat_decay = 0.86   # décroissance par frame de l'impulsion
        self._last_beat_t = -1.0  # anti-rebond : pas deux beats trop rapprochés
        self._beat_refractory = 0.10  # 100 ms mini entre deux onsets

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
          * sinon -> une seule rfft GPU + quelques produits matriciels (< 2 ms
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
            return feats

        # Anti-redondance : si aucun bloc neuf significatif depuis la dernière
        # analyse, on renvoie le dernier résultat (économise du temps GPU).
        if (written - self._last_analyzed_frame) < self._min_new_frames \
                and self._last_analyzed_frame >= 0:
            with self._feat_lock:
                feats = self._last_features
            feats.t = time.perf_counter() - self._t0
            return feats
        self._last_analyzed_frame = written

        # --- Analyse complète sur GPU --------------------------------------
        feats = self._analyze_window(win)
        feats.t = time.perf_counter() - self._t0
        # Expose la forme d'onde brute (GPU) + le compteur d'échantillons : la
        # « donnée » que le module de particules plonge en retards pour
        # reconstruire l'attracteur du son (device-to-device, pas de copie CPU).
        feats.waveform_gpu = cp.asarray(win, dtype=cp.float32)
        feats.samples_written = int(written)
        with self._feat_lock:
            self._last_features = feats
        return feats

    # ------------------------------------------------------------------ #
    #  Cœur DSP : analyse d'une fenêtre fft_size -> AudioFeatures.       #
    #  Tout le lourd est sur GPU ; on ne rapatrie que ~10 scalaires.     #
    # ------------------------------------------------------------------ #
    def _analyze_window(self, win: np.ndarray) -> AudioFeatures:
        # RMS temporel (amplitude brute) — petit calcul CPU avant l'upload.
        block_rms = float(np.sqrt(np.mean(win.astype(np.float64) ** 2)) + 1e-12)

        # Silence quasi total : on décroît les enveloppes et on publie ~0 sans
        # toucher au GPU lourdement (on garde quand même un spectrum_gpu valide).
        if block_rms < 1e-5:
            return self._silent_features()

        # --- Upload de la fenêtre sur le GPU (SEUL gros transfert CPU->GPU) ---
        # fft_size floats (p.ex. 4096) -> quelques Ko, négligeable sur PCIe 4.0.
        g_win = cp.asarray(win) * self._window

        # --- FFT réelle + magnitude (sur GPU) ---
        g_fft = cp.fft.rfft(g_win)                       # complexe (Nbins,)
        g_mag = cp.abs(g_fft) * (2.0 / self._win_gain)   # magnitude normalisée
        g_pow = g_mag * g_mag                            # puissance (pour bandes)

        # --- Énergie RMS par bande : un produit matriciel (4 x Nbins) ---
        g_band_rms = cp.sqrt(self._band_matrix @ g_pow + 1e-12)   # (4,)

        # --- Centroïde spectral (barycentre des fréquences pondéré par mag) --
        # centroid_hz = sum(f * mag) / sum(mag). Reste sur GPU.
        mag_sum = cp.sum(g_mag) + 1e-9
        g_centroid_hz = cp.sum(self._freqs_gpu * g_mag) / mag_sum

        # --- Spectre downsamplé 512 bins (magnitude moyenne par bin log) ---
        g_spec = self._spectrum_matrix @ g_mag           # (N_SPECTRUM,)
        # Échelle « musicale » : compression douce (racine) pour densifier le
        # bas niveau visuellement, sans le coût d'un log complet.
        g_spec = cp.sqrt(g_spec + 1e-12)

        # ---- RAPATRIEMENT GPU->CPU : seulement 4 + 1 scalaires ----
        band_rms = cp.asnumpy(g_band_rms).astype(np.float32)        # (4,)
        centroid_hz = float(cp.asnumpy(g_centroid_hz))

        # --- Normalisation adaptative (AGC doux) sur bandes + amplitude ---
        norm_bands = self._adaptive_normalize_bands(band_rms)
        norm_amp = self._adaptive_normalize_amp(block_rms)

        # --- Lissage temporel (attaque rapide / release lent) ---
        sm_bands = self._envelope_follow_bands(norm_bands)
        sm_amp = self._envelope_follow_amp(norm_amp)

        # --- Spectre GPU : normalisation adaptative + lissage, EN VRAM ---
        # On ne quitte jamais le GPU pour le spectre 512 bins.
        self._spectrum_max_gpu *= self._max_decay
        cp.maximum(self._spectrum_max_gpu, g_spec, out=self._spectrum_max_gpu)
        cp.maximum(self._spectrum_max_gpu, 1e-3, out=self._spectrum_max_gpu)
        norm_spec = g_spec / self._spectrum_max_gpu      # (N_SPECTRUM,) ~0..1
        # Lissage par bin (attaque rapide / release lent) directement sur GPU.
        up = norm_spec > self._env_spectrum_gpu
        coeff = cp.where(up, np.float32(self._attack), np.float32(self._release))
        self._env_spectrum_gpu = (
            self._env_spectrum_gpu + coeff * (norm_spec - self._env_spectrum_gpu)
        )
        cp.clip(self._env_spectrum_gpu, 0.0, 1.0, out=self._spectrum_out_gpu)
        # COPIE par frame : on remet une copie indépendante pour qu'une frame
        # suivante n'écrase pas le tableau que le consommateur est en train de
        # lire. 512 floats -> copie device-to-device quasi gratuite.
        spectrum_gpu_frame = self._spectrum_out_gpu.copy()

        # --- Centroïde normalisé 0..1 (échelle log entre FMIN et FMAX) ---
        # log-mapping car la perception de hauteur est logarithmique -> teinte
        # plus « naturelle » côté shaders.
        c = (np.log10(max(centroid_hz, self.SPECTRUM_FMIN))
             - np.log10(self.SPECTRUM_FMIN)) / (
                np.log10(self.SPECTRUM_FMAX) - np.log10(self.SPECTRUM_FMIN))
        centroid_norm = float(min(max(c, 0.0), 1.0))

        # --- Détection de beat (flux spectral bass+low_mid, seuil adaptatif) -
        beat_strength, is_beat = self._detect_beat(band_rms[:2])

        return AudioFeatures(
            bass=float(sm_bands[0]),
            low_mid=float(sm_bands[1]),
            mid=float(sm_bands[2]),
            high=float(sm_bands[3]),
            amplitude=float(sm_amp),
            beat=float(beat_strength),
            is_beat=bool(is_beat),
            centroid=centroid_norm,
            spectrum_gpu=spectrum_gpu_frame,
            t=0.0,   # rempli par get_features()
        )

    # ------------------------------------------------------------------ #
    #  Helpers de normalisation / lissage (CPU, scalaires).              #
    # ------------------------------------------------------------------ #
    def _adaptive_normalize_bands(self, raw: np.ndarray) -> np.ndarray:
        """AGC doux par bande : on divise par un max glissant (suit les pics,
        redescend lentement). Garde les features ~0..1 quel que soit le volume
        système, sans pompage brutal."""
        self._running_max_bands *= self._max_decay
        self._running_max_bands = np.maximum(self._running_max_bands, raw)
        self._running_max_bands = np.maximum(self._running_max_bands, 1e-3)
        return raw / self._running_max_bands

    def _adaptive_normalize_amp(self, raw: float) -> float:
        self._running_max_amp *= self._max_decay
        self._running_max_amp = max(self._running_max_amp, raw, 1e-3)
        return raw / self._running_max_amp

    def _envelope_follow_bands(self, target: np.ndarray) -> np.ndarray:
        """Suiveur d'enveloppe : attaque rapide à la montée, release lent à la
        descente -> bandes fluides mais réactives."""
        up = target > self._env_bands
        coeff = np.where(up, self._attack, self._release).astype(np.float32)
        self._env_bands = self._env_bands + coeff * (target - self._env_bands)
        return self._env_bands

    def _envelope_follow_amp(self, target: float) -> float:
        coeff = self._attack if target > self._env_amp else self._release
        self._env_amp = self._env_amp + coeff * (target - self._env_amp)
        return self._env_amp

    def _detect_beat(self, lowband_rms: np.ndarray) -> tuple[float, bool]:
        """Onset/beat via flux spectral half-wave sur (bass + low_mid).

        Principe :
          * flux = somme des hausses de magnitude des bandes graves (half-wave
            rectified) entre la frame précédente et l'actuelle.
          * seuil adaptatif = moyenne glissante + k * écart-type glissant du
            flux -> robuste quel que soit le niveau global.
          * un onset (flux > seuil) déclenche une impulsion `beat=1.0` qui
            décroît géométriquement ensuite ; `is_beat` est True uniquement sur
            la frame de déclenchement, avec une période réfractaire pour éviter
            les doubles déclenchements sur un même coup.
        """
        # Flux half-wave rectifié sur les deux bandes graves.
        diff = lowband_rms - self._prev_lowband_mag
        flux = float(np.sum(np.maximum(diff, 0.0)))
        self._prev_lowband_mag = lowband_rms.astype(np.float32, copy=True)

        # Statistiques glissantes du flux (moyenne + variance) pour le seuil.
        self._flux_avg = 0.97 * self._flux_avg + 0.03 * flux
        dev = flux - self._flux_avg
        self._flux_var = 0.97 * self._flux_var + 0.03 * (dev * dev)
        std = float(np.sqrt(self._flux_var))
        threshold = self._flux_avg + 1.6 * std + 1e-6

        now = time.perf_counter() - self._t0
        is_beat = False
        if flux > threshold and (now - self._last_beat_t) > self._beat_refractory:
            self._beat_env = 1.0          # attaque immédiate
            self._last_beat_t = now
            is_beat = True
        else:
            self._beat_env *= self._beat_decay   # décroissance de l'impulsion

        return float(min(self._beat_env, 1.0)), is_beat

    # ------------------------------------------------------------------ #
    #  Features « silence » : décroissance douce, spectrum_gpu valide.   #
    # ------------------------------------------------------------------ #
    def _silent_features(self) -> AudioFeatures:
        """Cas « rien ne joue » : on fait décroître toutes les enveloppes vers
        0 et on publie des features quasi nulles, en gardant un spectrum_gpu
        valide (zéros lissés) pour ne pas casser le consommateur GPU."""
        self._env_bands *= 0.85
        self._env_amp *= 0.85
        self._beat_env *= self._beat_decay
        self._running_max_bands *= self._max_decay
        self._running_max_amp *= self._max_decay

        # Spectre GPU décroît doucement vers 0 (toujours en VRAM).
        self._env_spectrum_gpu *= np.float32(0.85)
        self._spectrum_max_gpu *= self._max_decay
        cp.clip(self._env_spectrum_gpu, 0.0, 1.0, out=self._spectrum_out_gpu)
        spectrum_gpu_frame = self._spectrum_out_gpu.copy()

        return AudioFeatures(
            bass=float(self._env_bands[0]),
            low_mid=float(self._env_bands[1]),
            mid=float(self._env_bands[2]),
            high=float(self._env_bands[3]),
            amplitude=float(self._env_amp),
            beat=float(min(self._beat_env, 1.0)),
            is_beat=False,
            centroid=0.0,
            spectrum_gpu=spectrum_gpu_frame,
            t=0.0,   # rempli par get_features()
        )


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
