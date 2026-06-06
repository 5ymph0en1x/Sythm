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
    # --- EXTENSIONS (non-cassantes) : ONSETS PERCUSSIFS PAR REGISTRE.
    # Détection d'attaque séparée pour les trois grandes voix de la batterie, afin
    # que chacune déclenche un GESTE visuel distinct (cf. particles : ondes de choc
    # kick/snare/charley). Comme `beat`/`is_beat`, l'enveloppe décroît dans le temps
    # et le booléen `*_hit` n'est vrai QUE sur la frame de déclenchement.
    kick: float = 0.0          # onset GRAVE (sub/basses, ~20–150 Hz)   0..1, décroît
    snare: float = 0.0         # onset MÉDIUM large (caisse/clap, ~150–2000 Hz)
    hat: float = 0.0           # onset AIGU (charley/cymbales, ~2–16 kHz)
    kick_hit: bool = False     # True sur la frame d'attaque grave
    snare_hit: bool = False    # True sur la frame d'attaque médium
    hat_hit: bool = False      # True sur la frame d'attaque aiguë
    # --- EXTENSIONS (non-cassantes) : MODÈLE DE TEMPO / PHASE (entraînement au
    # groove). Un oscillateur adaptatif s'accroche au pouls et PRÉDIT le temps
    # fort -> permet l'ANTICIPATION (la nuée inspire avant le beat) et une houle
    # caméra calée sur la mesure. S'efface seul (groove_conf->0) sur la musique
    # sans pulsation nette. Mis à jour à CHAQUE frame (la phase avance en continu).
    tempo_bpm: float = 0.0       # tempo estimé (battements/min ; 0 = pas verrouillé)
    pulse_phase: float = 0.0     # phase du battement [0,1) (0 = sur le temps)
    bar_phase: float = 0.0       # phase lente sur 4 temps [0,1) (« houle » de mesure)
    groove_conf: float = 0.0     # confiance du verrouillage [0,1] (porte tout l'effet)
    anticipation: float = 0.0    # [0,1] : monte AVANT le temps fort prédit (l'inspir)
    # --- EXTENSIONS (non-cassantes) : ANTICIPATION DE PHRASE (build / drop).
    # Échelle SUPÉRIEURE au battement : la TENSION qui s'accumule pendant un build
    # (filtre qui s'ouvre, roulement qui accélère, sub qui se retire) PUIS le DROP
    # (relâche quand le sub/kick claque de nouveau). Gated par groove_conf.
    build: float = 0.0           # [0,1] : charge de tension qui s'accumule (avant le drop)
    drop: float = 0.0            # [0,1] : impulsion de relâche au drop (décroît)
    phrase_phase: float = 0.0    # [0,1] : phase de phrase, recalée au drop
    # --- EXTENSIONS (non-cassantes) : COULEUR DE L'HARMONIE.
    # Chroma 12 classes -> corrélation aux profils de tonalité (Krumhansl) :
    # tonal_warmth = axe MAJEUR(chaud)/MINEUR(froid) ; key_hue = teinte-maison de la
    # tonalité. Très LENTS (l'harmonie change sur des mesures). Teintent la palette.
    tonal_warmth: float = 0.0    # [-1,1] : +1 majeur (chaud) … -1 mineur (froid)
    key_hue: float = 0.0         # [0,1) : fondamentale estimée / 12 (teinte-maison)


# ===========================================================================
#  Détecteur d'ONSET générique (flux spectral half-wave + seuil adaptatif).
# ===========================================================================
class _OnsetDetector:
    """Détecteur d'attaque sur UN signal d'énergie de bande (scalaire par frame).

    Même principe que la détection de beat historique, mais factorisé pour être
    instancié une fois PAR REGISTRE (grave / médium / aigu) avec ses propres
    constantes — une charley n'a pas la même dynamique qu'un kick.

    Principe :
      * flux = hausse half-wave rectifiée de l'énergie depuis la frame précédente ;
      * seuil ADAPTATIF = moyenne glissante + k·écart-type glissant du flux
        (robuste quel que soit le niveau global) ;
      * un franchissement déclenche une impulsion `env=1.0` qui décroît
        géométriquement ; `hit` n'est vrai QUE sur la frame de déclenchement, avec
        une période réfractaire pour éviter les doubles déclenchements.

    Renvoie (env 0..1, hit bool) à chaque `update`.
    """

    __slots__ = ("_prev", "_avg", "_var", "_env", "_last_t",
                 "refractory", "decay", "k", "_a")

    def __init__(self, refractory=0.08, decay=0.85, k=1.6, smoothing=0.03):
        self._prev = 0.0        # énergie de bande à la frame précédente
        self._avg = 1e-6        # moyenne glissante du flux (seuil)
        self._var = 1e-6        # variance glissante du flux
        self._env = 0.0         # impulsion d'onset (décroît)
        self._last_t = -1.0     # instant du dernier onset (anti-rebond)
        self.refractory = float(refractory)
        self.decay = float(decay)
        self.k = float(k)
        self._a = float(smoothing)   # vitesse d'adaptation des stats glissantes

    def update(self, value: float, now: float) -> tuple[float, bool]:
        flux = value - self._prev
        if flux < 0.0:
            flux = 0.0
        self._prev = value
        a = self._a
        self._avg = (1.0 - a) * self._avg + a * flux
        dev = flux - self._avg
        self._var = (1.0 - a) * self._var + a * (dev * dev)
        std = math.sqrt(self._var)
        threshold = self._avg + self.k * std + 1e-6
        hit = False
        if flux > threshold and (now - self._last_t) > self.refractory:
            self._env = 1.0
            self._last_t = now
            hit = True
        else:
            self._env *= self.decay
        if self._env > 1.0:
            self._env = 1.0
        return self._env, hit

    def decay_only(self) -> float:
        """Cas silence : on fait juste décroître l'impulsion (aucun onset)."""
        self._env *= self.decay
        return self._env


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

        # Onsets PERCUSSIFS par registre (kick/snare/charley) — un détecteur par
        # voix, avec des constantes propres à sa dynamique. Le kick est lent et
        # massif (réfractaire long, décroissance douce) ; la charley est vive et
        # serrée (réfractaire court, décroissance rapide). Le snare écoute une
        # bande médium LARGE (caisse claire/clap = transitoire à large spectre).
        self._onset_kick = _OnsetDetector(refractory=0.11, decay=0.84, k=1.5)
        self._onset_snare = _OnsetDetector(refractory=0.07, decay=0.80, k=1.7)
        self._onset_hat = _OnsetDetector(refractory=0.045, decay=0.72, k=1.9)

        # OSCILLATEUR DE TEMPO/PHASE (entraînement au groove, façon oscillateur
        # adaptatif). Alimenté par les beats `is_beat` ; corrige doucement sa
        # période et sa phase à chaque battement, et avance librement entre deux.
        self._tempo_period = 0.5    # s par battement (120 BPM par défaut)
        self._tempo_phase = 0.0     # phase courante [0,1)
        self._tempo_conf = 0.0      # confiance de verrouillage [0,1]
        self._tempo_last_t = 0.0    # horodatage du dernier update de phase
        self._tempo_last_beat_t = -1.0  # dernier battement encaissé
        self._bar_count = 0         # compteur de temps (0..3) -> phase de mesure

        # TRAQUEUR DE PHRASE (build / drop) — échelle macro, gated par le verrou.
        # Tension = filtre qui s'ouvre (centroïde + aigus) + activité percussive ;
        # un BUILD = tension qui monte sur des secondes (fast EMA > slow EMA) ; un
        # DROP = le sub qui SLAMME de nouveau alors qu'un build était chargé.
        self._phrase_bars = 8       # longueur de phrase (mesures de 4 temps)
        self._tension_fast = 0.0    # EMA rapide de la tension (~1 s)
        self._tension_slow = 0.0    # EMA lente (~plusieurs s) -> ligne de base
        self._bass_slow = 0.0       # EMA lente du grave (pour détecter le SLAM)
        self._build = 0.0           # charge lissée [0,1]
        self._build_target = 0.0    # cible de charge (mise à jour à l'analyse)
        self._drop_env = 0.0        # impulsion de drop (décroît)
        self._last_drop_t = -10.0   # anti-rebond entre deux drops
        self._phrase_beat = 0       # compteur de temps DANS la phrase

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
            self._apply_tempo(feats, feats.t, False)   # phase continue (pas de beat frais)
            return feats

        # Anti-redondance : si aucun bloc neuf significatif depuis la dernière
        # analyse, on renvoie le dernier résultat (économise du temps GPU).
        if (written - self._last_analyzed_frame) < self._min_new_frames \
                and self._last_analyzed_frame >= 0:
            with self._feat_lock:
                feats = self._last_features
            feats.t = time.perf_counter() - self._t0
            self._apply_tempo(feats, feats.t, False)   # phase continue (snapshot recyclé)
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
        # Oscillateur de tempo : corrige période/phase sur ce battement frais.
        self._apply_tempo(feats, feats.t, bool(feats.is_beat))
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

        # --- HARMONIE : chroma 12 classes (réduction diffusée, reste GPU) puis
        #     corrélation aux profils de tonalité (Krumhansl) -> warmth + key_hue.
        g_chroma = cp.sum(self._chroma_matrix_gpu * g_spec[None, :], axis=1)   # (12,)
        warmth_now, key_hue_now = self._update_harmony(
            cp.asnumpy(g_chroma).astype(np.float64))

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

        # --- Onsets PERCUSSIFS par registre (sur les énergies de bande BRUTES) -
        # kick = grave ; snare = médium LARGE (low_mid+mid, transitoire de caisse
        # claire/clap) ; charley = aigu. Chacun alimente un geste visuel distinct.
        now = time.perf_counter() - self._t0
        kick_s, kick_h = self._onset_kick.update(float(band_rms[0]), now)
        snare_s, snare_h = self._onset_snare.update(
            float(band_rms[1] + band_rms[2]), now)
        hat_s, hat_h = self._onset_hat.update(float(band_rms[3]), now)

        # --- Traqueur de PHRASE (build/drop) sur les features lissées/normalisées.
        self._phrase_detect(high=float(sm_bands[3]), centroid=centroid_norm,
                            snare=snare_s, hat=hat_s, bass=float(sm_bands[0]), now=now)

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
            kick=float(kick_s), snare=float(snare_s), hat=float(hat_s),
            kick_hit=bool(kick_h), snare_hit=bool(snare_h), hat_hit=bool(hat_h),
            tonal_warmth=float(warmth_now), key_hue=float(key_hue_now),
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
    #  Analyse HARMONIQUE : chroma 12 classes + profils de tonalité Krumhansl.
    # ------------------------------------------------------------------ #
    def _build_harmony(self):
        """Pré-calcule la matrice chroma (12 x N_SPECTRUM) et les 24 profils de
        tonalité (majeur/mineur × 12 rotations) normalisés pour la corrélation."""
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
        # Profils Krumhansl-Schmuckler (poids par classe, clé en 0).
        ks_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        ks_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        def _normp(v):
            v = v - v.mean()
            return v / (np.linalg.norm(v) + 1e-9)

        rows, major_flag, root_idx = [], [], []
        for root in range(12):
            rows.append(_normp(np.roll(ks_major, root))); major_flag.append(True); root_idx.append(root)
            rows.append(_normp(np.roll(ks_minor, root))); major_flag.append(False); root_idx.append(root)
        self._ks_mat = np.array(rows, dtype=np.float64)        # (24, 12) corrélation
        self._ks_major = np.array(major_flag, dtype=bool)      # (24,)
        self._ks_root = np.array(root_idx, dtype=np.int32)     # (24,)
        # État lissé (l'harmonie est lente).
        self._chroma = np.zeros(12, dtype=np.float64)
        self._tonal_warmth = 0.0
        self._key_root = 0
        self._key_counter = 0

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

    def _update_harmony(self, chroma):
        """Lisse le chroma, corrèle aux profils -> (tonal_warmth, key_hue).
        warmth = meilleure corrélation MAJEUR − meilleure MINEUR (chaud vs froid) ;
        key_hue = fondamentale du meilleur profil, avec hystérésis anti-clignotement."""
        self._chroma += 0.025 * (chroma - self._chroma)        # ~0.5 s
        c = self._chroma - self._chroma.mean()
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-6:                                          # pas d'harmonie nette
            self._tonal_warmth *= 0.98
            return self._tonal_warmth, self._key_root / 12.0
        scores = self._ks_mat @ (c / nrm)                      # (24,) corrélations
        c_maj = float(scores[self._ks_major].max())
        c_min = float(scores[~self._ks_major].max())
        warmth_raw = (c_maj - c_min) * 2.0
        if warmth_raw > 1.0:
            warmth_raw = 1.0
        elif warmth_raw < -1.0:
            warmth_raw = -1.0
        self._tonal_warmth += 0.02 * (warmth_raw - self._tonal_warmth)   # ~0.8 s
        # Tonalité (teinte-maison) : hystérésis pour ne pas clignoter entre clés.
        best = int(np.argmax(scores))
        best_root = int(self._ks_root[best])
        cur_score = float(scores[self._ks_root == self._key_root].max())
        if best_root != self._key_root and float(scores[best]) > cur_score + 0.04:
            self._key_counter += 1
            if self._key_counter >= 3:
                self._key_root = best_root
                self._key_counter = 0
        else:
            self._key_counter = 0
        return self._tonal_warmth, self._key_root / 12.0

    # ------------------------------------------------------------------ #
    #  Traqueur de PHRASE : détecte le BUILD (tension qui monte) et le DROP
    #  (le sub qui claque après un build). Appelé à l'ANALYSE (signaux frais) ;
    #  le lissage par-frame de build/drop vit dans _apply_tempo.
    # ------------------------------------------------------------------ #
    def _phrase_detect(self, high, centroid, snare, hat, bass, now):
        # Tension = filtre ouvert (aigus + centroïde) + activité percussive haute.
        act = 0.5 * (snare + hat)
        tension = 0.40 * high + 0.30 * centroid + 0.30 * act
        self._tension_fast += 0.06 * (tension - self._tension_fast)     # ~1 s
        self._tension_slow += 0.012 * (tension - self._tension_slow)    # ~plusieurs s
        self._bass_slow += 0.02 * (bass - self._bass_slow)

        conf = self._tempo_conf
        # BUILD = tension AU-DESSUS de sa ligne de base (= ça monte), porté par le
        # verrou de tempo (aucun build sur de l'arythmique).
        # Composante « ça monte » (tendance fast>slow) + bonus « tension haute
        # soutenue » -> un build long et énergique charge vraiment, pas juste sa pente.
        bt = (self._tension_fast - self._tension_slow) * 4.0
        bt += 0.4 * max(0.0, self._tension_fast - 0.45)
        if bt < 0.0:
            bt = 0.0
        elif bt > 1.0:
            bt = 1.0
        self._build_target = bt * conf

        # DROP = le grave SLAMME de nouveau (surge vs ligne de base) ALORS qu'un
        # build était chargé. Précondition build + surge + verrou + réfractaire.
        bass_surge = bass - self._bass_slow
        if (self._build > 0.30 and bass_surge > 0.22 and conf > 0.40
                and (now - self._last_drop_t) > 2.0):
            self._drop_env = 1.0
            self._last_drop_t = now
            self._phrase_beat = 0          # le drop = frontière de phrase
            self._build = 0.0              # décharge immédiate
            self._build_target = 0.0

    # ------------------------------------------------------------------ #
    #  Oscillateur de TEMPO/PHASE : entraînement au groove + anticipation.
    #  Appelé à CHAQUE frame (même cache) pour que la phase avance en continu ;
    #  ne corrige période/phase QUE sur un battement frais (`beat_now`).
    # ------------------------------------------------------------------ #
    def _apply_tempo(self, feats: "AudioFeatures", now: float, beat_now: bool) -> None:
        # 1) Avance de la phase au temps réel écoulé depuis le dernier appel.
        dt = now - self._tempo_last_t
        self._tempo_last_t = now
        if dt < 0.0:
            dt = 0.0
        elif dt > 0.5:
            dt = 0.0   # gros trou (pause) -> on n'avance pas la phase d'un coup

        period = self._tempo_period
        self._tempo_phase += dt / max(period, 1e-3)
        while self._tempo_phase >= 1.0:           # franchissement d'un temps prédit
            self._tempo_phase -= 1.0
            self._bar_count = (self._bar_count + 1) & 3   # 0..3 (mesure 4 temps)
            self._phrase_beat = (self._phrase_beat + 1) % (self._phrase_bars * 4)

        # 2) Correction sur un battement FRAIS (oscillateur adaptatif).
        if beat_now:
            # Erreur de phase e ∈ (-0.5, 0.5] : 0 = le beat tombe pile sur phase=0.
            e = self._tempo_phase
            if e > 0.5:
                e -= 1.0
            ae = abs(e)
            # --- PÉRIODE par intervalle inter-onset (robuste, repliement d'octave)
            # L'écart depuis le dernier battement EST une observation de période
            # (ou un multiple) : on le replie sur l'octave la plus proche de notre
            # estimation, puis EMA -> verrouillage en quelques battements.
            if self._tempo_last_beat_t >= 0.0:
                ioi = now - self._tempo_last_beat_t
                if ioi > 0.05:
                    if self._tempo_conf > 0.40:
                        # SUIVI : on a un verrou -> replie l'IOI sur l'octave la
                        # plus proche du tempo connu (absorbe beats manqués/en trop).
                        while ioi < 0.75 * period:
                            ioi *= 2.0
                        while ioi > 1.50 * period:
                            ioi *= 0.5
                        beta = 0.20
                    else:
                        # ACQUISITION : pas encore de verrou -> on FAIT CONFIANCE à
                        # l'IOI brut (juste replié dans la bande [60,200] BPM) et on
                        # converge vite, sans biais vers la période par défaut.
                        while ioi < 0.30:
                            ioi *= 2.0
                        while ioi > 1.00:
                            ioi *= 0.5
                        beta = 0.50
                    if 0.28 <= ioi <= 1.05:
                        period = (1.0 - beta) * period + beta * ioi
                        if period < 0.30:
                            period = 0.30
                        elif period > 1.00:
                            period = 1.00
                        self._tempo_period = period
            # --- PHASE : tire fermement la phase vers le battement (alignement).
            self._tempo_phase -= 0.40 * e
            if self._tempo_phase < 0.0:
                self._tempo_phase += 1.0
            elif self._tempo_phase >= 1.0:
                self._tempo_phase -= 1.0
            # --- CONFIANCE : forte quand les battements tombent régulièrement où
            # on les prédit (petite erreur de phase). EMA -> entraînement progressif.
            target = 1.0 - 2.5 * ae
            if target < 0.0:
                target = 0.0
            self._tempo_conf += 0.12 * (target - self._tempo_conf)
            self._tempo_last_beat_t = now
        else:
            # Pas de battement : si le pouls s'est tu, on PERD le verrou peu à peu.
            if (self._tempo_last_beat_t >= 0.0
                    and (now - self._tempo_last_beat_t) > 2.0 * period):
                self._tempo_conf -= dt * 0.35   # ~3 s pour tout relâcher
                if self._tempo_conf < 0.0:
                    self._tempo_conf = 0.0

        # 3) Anticipation : tension qui MONTE sur le dernier tiers avant le temps
        #    fort prédit (l'inspir), nulle ailleurs, portée par la confiance.
        ph = self._tempo_phase
        w = 0.33
        if ph > (1.0 - w):
            tension = (ph - (1.0 - w)) / w
            tension *= tension                 # ease-in (douceur croissante)
        else:
            tension = 0.0

        # 4) Publication dans le snapshot (lu par particules + caméra).
        conf = self._tempo_conf
        feats.tempo_bpm = float(60.0 / max(self._tempo_period, 1e-3)) if conf > 0.05 else 0.0
        feats.pulse_phase = float(self._tempo_phase)
        feats.bar_phase = float((self._bar_count + self._tempo_phase) * 0.25)
        feats.groove_conf = float(conf)
        feats.anticipation = float(tension * conf)

        # 5) PHRASE (échelle macro). La charge a une attaque LENTE (~2 s : "ça se
        #    charge sur des mesures") et un release plus vif ; le drop s'épanouit
        #    puis retombe. phrase_phase recalée au drop (cf. _phrase_detect).
        if self._build_target > self._build:
            self._build += (1.0 - math.exp(-dt / 1.2)) * (self._build_target - self._build)
        else:
            self._build += (1.0 - math.exp(-dt / 0.5)) * (self._build_target - self._build)
        self._drop_env *= math.exp(-dt / 1.2)
        feats.build = float(self._build)
        feats.drop = float(min(self._drop_env, 1.0))
        feats.phrase_phase = float((self._phrase_beat + self._tempo_phase)
                                   / max(1, self._phrase_bars * 4))

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

        # Onsets percussifs : simple décroissance (aucune attaque en silence).
        kick_s = self._onset_kick.decay_only()
        snare_s = self._onset_snare.decay_only()
        hat_s = self._onset_hat.decay_only()
        self._build_target = 0.0     # plus de tension en silence -> la charge se relâche
        self._tonal_warmth *= 0.97   # plus d'harmonie -> retour au neutre (gris)

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
            kick=float(kick_s), snare=float(snare_s), hat=float(hat_s),
            kick_hit=False, snare_hit=False, hat_hit=False,
            tonal_warmth=float(self._tonal_warmth), key_hue=float(self._key_root / 12.0),
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
