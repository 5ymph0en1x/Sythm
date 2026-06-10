# -*- coding: utf-8 -*-
"""
audio_features.py
=================
Le CONTRAT de données partagé du visualiseur : la dataclasse `AudioFeatures`,
photo instantanée des caractéristiques audio pour UNE frame de rendu.

Extrait d'audio_engine.py pour DÉCOUPLER la donnée de son producteur : le moteur
de bas niveau (capture + FFT GPU, `audio_engine.AudioEngine`) ET l'analyseur
musical de haut niveau (`musical_analyzer.MusicalAnalyzer`) le partagent sans
dépendance circulaire. `audio_engine` le RÉ-EXPORTE -> `from audio_engine import
AudioFeatures` reste valide (contrat inchangé pour particles/renderer).

Aucune dépendance (pas de cupy/numpy) : les annotations `cp.ndarray` sont des
CHAÎNES (jamais évaluées au runtime), donc ce module reste importable partout.
"""
from __future__ import annotations

from dataclasses import dataclass


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

    Voir le contrat partagé en tête d'audio_engine.py : noms et unités sont figés.
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
    # --- EXTENSIONS (non-cassantes) : SONIE PERÇUE (psychoacoustique).
    # Le préset « Cognitive » ne pilote la nuée QU'AVEC ce que l'oreille ENTEND, pas
    # avec l'énergie physique. On pondère le spectre par une courbe d'ISOSONIE
    # (pondération A ≈ contour de Fletcher-Munson : l'oreille est sourde au sub-grave
    # et aux très aigus, ultra-sensible vers 2–5 kHz) puis on COMPRIME en loudness
    # (loi de puissance de Stevens : doubler la sonie perçue ≈ ×10 l'énergie). Tous
    # ces champs valent 0 si non calculés -> consommés via getattr (compat. totale).
    loudness: float = 0.0        # [0,1] : sonie GLOBALE perçue (A-pondérée, compressée)
    p_bass: float = 0.0          # [0,1] : équilibre de bande PERÇU (grave A-pondéré)
    p_low_mid: float = 0.0       # [0,1] : bas-médium perçu
    p_mid: float = 0.0           # [0,1] : médium perçu (zone de présence, ~la + audible)
    p_high: float = 0.0          # [0,1] : aigu perçu
    # --- EXTENSION (non-cassante) : BASSE PROFONDE « ressentie » (force viscérale).
    # Énergie du SUB (~20–60 Hz) : ce que le corps RESSENT plutôt que ce que l'oreille
    # entend. VOLONTAIREMENT non pondéré par l'isosonie (qui l'écraserait) -> pilote les
    # « ondes gravitationnelles » : plus le grave est deep, plus l'impact est GÉNÉRAL.
    sub: float = 0.0             # [0,1] : sous-grave profond (20–60 Hz), lissé
