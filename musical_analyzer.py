# -*- coding: utf-8 -*-
"""
musical_analyzer.py
===================
ANALYSE MUSICALE DE HAUT NIVEAU — découplée de l'acquisition du signal.

`audio_engine.AudioEngine` fait l'ACQUISITION (capture loopback + FFT GPU +
réductions pondérées) et produit une `RawFrame` (quelques scalaires CPU + le
spectre GPU). Ce module en extrait la SÉMANTIQUE MUSICALE — entièrement du
scalaire CPU, sans GPU ni micro — ce qui le rend UNITAIREMENT TESTABLE : on
peut lui injecter des `RawFrame` synthétiques et un temps `now` déterministe,
puis vérifier le verrouillage de tempo, le build/drop, la tonalité, etc., sans
lancer de carte son.

Frontière (respecte le threading + le contrat figés) :
  * l'AudioEngine garde TOUT ce qui doit vivre sur le thread de rendu / contexte
    CUDA (FFT, réductions, lissage du spectre GPU) ;
  * le MusicalAnalyzer ne voit que des scalaires déjà réduits + le handle GPU du
    spectre (qu'il transmet tel quel dans AudioFeatures).

Le `now` (secondes) est INJECTÉ par l'appelant (l'AudioEngine le calcule une
fois par frame) au lieu d'être lu en interne via time.perf_counter() : timing
unifié sur la frame ET déterminisme total pour les tests.

Décomposition :
  RhythmTracker    : onsets (kick/snare/hat) + beat + oscillateur de tempo + phrase
  HarmonyAnalyzer  : corrélation chroma -> profils de tonalité (Krumhansl)
  PerceptualModel  : sonie d'isosonie + équilibre de bandes perçu + sub ressenti
  MusicalAnalyzer  : façade — suivi de bandes/amplitude + compose les trois
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from audio_features import AudioFeatures

# Constantes de LISSAGE des features (suiveur d'enveloppe : attaque rapide /
# release lent) + décroissance du max glissant (AGC). SOURCE UNIQUE : l'AudioEngine
# les ré-importe pour le lissage du spectre GPU, afin que CPU et GPU restent calés.
ATTACK = 0.6        # 0..1 : plus grand = montée plus rapide
RELEASE = 0.10      # 0..1 : plus petit = descente plus lente
MAX_DECAY = 0.9995  # le max glissant (AGC) redescend lentement


# ===========================================================================
#  RawFrame : le pont AudioEngine (acquisition GPU) -> MusicalAnalyzer (CPU).
# ===========================================================================
@dataclass
class RawFrame:
    """Sortie BRUTE d'une frame d'acquisition : scalaires déjà réduits (CPU) + le
    spectre GPU (transmis tel quel). Tout est physique/non lissé : c'est
    l'analyseur qui normalise, lisse et interprète."""
    band_rms: np.ndarray          # (4,) RMS par bande, échelle physique COMMUNE
    block_rms: float              # amplitude RMS temporelle brute
    centroid_hz: float            # centroïde spectral (Hz)
    loud_lin: float               # niveau large bande A-pondéré (isosonie)
    sub_lin: float                # niveau sub-grave (20–60 Hz, non pondéré)
    chroma: np.ndarray            # (12,) vecteur chroma (post-diapason)
    spectrum_gpu: object = None   # cupy.ndarray (N_SPECTRUM,) — transmis à AudioFeatures


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
#  RhythmTracker : onsets + beat + oscillateur de tempo/phase + phrase.
# ===========================================================================
class RhythmTracker:
    """Tout le RYTHME : la détection de beat (flux des graves), les trois onsets
    percussifs (kick/snare/hat), l'oscillateur de tempo/phase qui s'entraîne au
    groove (anticipation), et le traqueur de phrase macro (build/drop). Cohésif :
    le tempo, la phrase et les onsets se nourrissent mutuellement EN INTERNE."""

    def __init__(self):
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

    # ------------------------------------------------------------------ #
    def update(self, band_rms, sm_bass, sm_high, centroid_norm, now):
        """Beat + onsets + phrase pour une frame FRAÎCHE. `band_rms` = énergies de
        bande BRUTES (pour le flux/onsets) ; `sm_bass`/`sm_high` = bandes LISSÉES
        (pour la tension de phrase). Renvoie un dict des champs rythmiques."""
        beat_strength, is_beat = self._detect_beat(band_rms[:2], now)
        kick_s, kick_h = self._onset_kick.update(float(band_rms[0]), now)
        snare_s, snare_h = self._onset_snare.update(
            float(band_rms[1] + band_rms[2]), now)
        hat_s, hat_h = self._onset_hat.update(float(band_rms[3]), now)
        # Traqueur de PHRASE (build/drop) sur les features lissées/normalisées.
        self._phrase_detect(high=sm_high, centroid=centroid_norm,
                            snare=snare_s, hat=hat_s, bass=sm_bass, now=now)
        return dict(beat=beat_strength, is_beat=is_beat,
                    kick=kick_s, snare=snare_s, hat=hat_s,
                    kick_hit=kick_h, snare_hit=snare_h, hat_hit=hat_h)

    def decay_silent(self):
        """Cas silence : impulsion de beat + onsets décroissent, la tension de
        phrase se relâche. Renvoie (beat, kick, snare, hat)."""
        self._beat_env *= self._beat_decay
        kick_s = self._onset_kick.decay_only()
        snare_s = self._onset_snare.decay_only()
        hat_s = self._onset_hat.decay_only()
        self._build_target = 0.0     # plus de tension en silence -> la charge se relâche
        return float(min(self._beat_env, 1.0)), kick_s, snare_s, hat_s

    def _detect_beat(self, lowband_rms, now):
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

        is_beat = False
        if flux > threshold and (now - self._last_beat_t) > self._beat_refractory:
            self._beat_env = 1.0          # attaque immédiate
            self._last_beat_t = now
            is_beat = True
        else:
            self._beat_env *= self._beat_decay   # décroissance de l'impulsion

        return float(min(self._beat_env, 1.0)), is_beat

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

    def apply_tempo(self, feats: AudioFeatures, now: float, beat_now: bool) -> None:
        """Oscillateur de TEMPO/PHASE : entraînement au groove + anticipation.
        Appelé à CHAQUE frame (même cache) pour que la phase avance en continu ;
        ne corrige période/phase QUE sur un battement frais (`beat_now`). Écrit
        aussi les champs de PHRASE (lissage de build/drop)."""
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


# ===========================================================================
#  HarmonyAnalyzer : corrélation chroma -> profils de tonalité (Krumhansl).
# ===========================================================================
class HarmonyAnalyzer:
    """Prend le vecteur CHROMA (12 classes, déjà recentré sur le diapason estimé
    par l'AudioEngine) et le corrèle aux 24 profils Krumhansl-Schmuckler ->
    (tonal_warmth, key_hue). Lent (l'harmonie change sur des mesures)."""

    def __init__(self):
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

    def update(self, chroma):
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

    def decay(self):
        """Cas silence : plus d'harmonie -> retour au neutre (gris)."""
        self._tonal_warmth *= 0.97
        return self._tonal_warmth, self._key_root / 12.0


# ===========================================================================
#  PerceptualModel : sonie d'isosonie + équilibre de bandes perçu + sub ressenti.
# ===========================================================================
class PerceptualModel:
    """Transforme les niveaux PHYSIQUES (band_rms, niveau A-pondéré, sub) en
    grandeurs PERÇUES. `perc_band_w` (poids d'isosonie moyens par bande) est
    pré-calculé par l'AudioEngine (qui a l'axe fréquentiel) et injecté ici."""

    def __init__(self, perc_band_w):
        self._perc_band_w = perc_band_w        # (4,) poids d'isosonie par bande
        # RÉFÉRENCE COMMUNE (un seul max glissant pour la sonie ET les 4 bandes ->
        # loudness et équilibre partagent une même échelle perçue) + enveloppes.
        self._perc_running_max = 1e-3
        self._perc_env_bands = np.zeros(4, dtype=np.float32)
        self._loud_env = 0.0
        # BASSE PROFONDE ressentie (sub) : AGC + enveloppe dédiés.
        self._running_max_sub = 1e-3
        self._env_sub = 0.0

    def update(self, band_rms, loud_lin, sub_lin):
        """Sonie PERÇUE -> (loudness 0..1, p_bands[4] 0..1, sub 0..1). `band_rms`
        (RMS par bande) et `loud_lin` (niveau large bande A-pondéré) sont dans la
        MÊME échelle physique. On pondère les bandes par l'isosonie, on les
        normalise avec la sonie globale par une RÉFÉRENCE COMMUNE (un seul AGC ->
        un grave que l'oreille n'entend guère reste FAIBLE dans un mix, au lieu
        d'être renormalisé à fond), on COMPRIME en loudness (loi de puissance,
        exposant 0.6 sur l'amplitude ≈ 0.3 sur la puissance — Stevens), puis on
        lisse (attaque rapide / release lent). Le SUB a sa propre dynamique."""
        p_lin = band_rms.astype(np.float32) * self._perc_band_w          # (4,) A-pondéré
        # RÉFÉRENCE COMMUNE = max(sonie large bande, bande perçue la + forte), AGC lent.
        ref_now = max(float(loud_lin), float(p_lin.max()))
        self._perc_running_max = max(self._perc_running_max * MAX_DECAY, ref_now, 1e-3)
        ref = self._perc_running_max
        # Sonie globale (pilote la luminosité/le flux) : compression + enveloppe.
        ln = (min(float(loud_lin) / ref, 1.0)) ** 0.6
        self._loud_env += (ATTACK if ln > self._loud_env else RELEASE) * (ln - self._loud_env)
        # Équilibre de bandes perçu (pilote la couleur), MÊME référence -> balance réelle.
        p_comp = np.clip(p_lin / ref, 0.0, 1.0) ** 0.6
        up = p_comp > self._perc_env_bands
        coeff = np.where(up, ATTACK, RELEASE).astype(np.float32)
        self._perc_env_bands = self._perc_env_bands + coeff * (p_comp - self._perc_env_bands)
        # BASSE PROFONDE ressentie (sub) : AGC + enveloppe (sa propre dynamique).
        self._running_max_sub = max(self._running_max_sub * MAX_DECAY, float(sub_lin), 1e-3)
        sub_n = min(float(sub_lin) / self._running_max_sub, 1.0)
        self._env_sub += (ATTACK if sub_n > self._env_sub else RELEASE) * (sub_n - self._env_sub)
        return float(self._loud_env), self._perc_env_bands, float(self._env_sub)

    def decay(self):
        """Cas silence : décroissance douce vers le silence."""
        self._perc_env_bands *= np.float32(0.85)
        self._loud_env *= 0.85
        self._perc_running_max *= MAX_DECAY
        self._env_sub *= 0.85
        self._running_max_sub *= MAX_DECAY
        return float(self._loud_env), self._perc_env_bands, float(self._env_sub)


# ===========================================================================
#  MusicalAnalyzer : façade — suivi de bandes/amplitude + compose les trois.
# ===========================================================================
class MusicalAnalyzer:
    """Sémantique musicale complète d'une frame. Prend une `RawFrame` (acquisition)
    + le temps `now` (injecté) et produit un `AudioFeatures`. Le tempo s'applique
    SÉPARÉMENT (apply_tempo), à CHAQUE frame, y compris sur instantané recyclé."""

    def __init__(self, perc_band_w, spectrum_fmin, spectrum_fmax):
        self._fmin = float(spectrum_fmin)
        self._fmax = float(spectrum_fmax)
        self._log_fmin = math.log10(self._fmin)
        self._log_span = math.log10(self._fmax) - self._log_fmin
        # Enveloppes lissées des 4 bandes + amplitude (attaque rapide / release lent).
        self._env_bands = np.zeros(4, dtype=np.float32)
        self._env_amp = 0.0
        # Normalisation adaptative (AGC doux) : max glissant par bande + ampl.
        self._running_max_bands = np.full(4, 1e-3, dtype=np.float32)
        self._running_max_amp = 1e-3
        # Traqueurs spécialisés.
        self.rhythm = RhythmTracker()
        self.harmony = HarmonyAnalyzer()
        self.perceptual = PerceptualModel(perc_band_w)

    # ------------------------------------------------------------------ #
    def analyze(self, raw: RawFrame, now: float) -> AudioFeatures:
        """Frame FRAÎCHE -> AudioFeatures (hors tempo, appliqué ensuite)."""
        # --- Normalisation adaptative (AGC doux) sur bandes + amplitude ---
        norm_bands = self._adaptive_normalize_bands(raw.band_rms)
        norm_amp = self._adaptive_normalize_amp(raw.block_rms)
        # --- Lissage temporel (attaque rapide / release lent) ---
        sm_bands = self._envelope_follow_bands(norm_bands)
        sm_amp = self._envelope_follow_amp(norm_amp)
        # --- Centroïde normalisé 0..1 (échelle log entre FMIN et FMAX) ---
        c = (math.log10(max(raw.centroid_hz, self._fmin)) - self._log_fmin) / self._log_span
        centroid_norm = float(min(max(c, 0.0), 1.0))
        # --- Rythme (beat + onsets + phrase) sur signaux frais ---
        rhy = self.rhythm.update(raw.band_rms, float(sm_bands[0]), float(sm_bands[3]),
                                 centroid_norm, now)
        # --- Sonie PERÇUE (loudness + équilibre de bandes + sub ressenti) ---
        loudness, p_bands, sub = self.perceptual.update(raw.band_rms, raw.loud_lin, raw.sub_lin)
        # --- Harmonie (modalité + tonalité) ---
        warmth_now, key_hue_now = self.harmony.update(raw.chroma)

        return AudioFeatures(
            bass=float(sm_bands[0]),
            low_mid=float(sm_bands[1]),
            mid=float(sm_bands[2]),
            high=float(sm_bands[3]),
            amplitude=float(sm_amp),
            beat=float(rhy["beat"]),
            is_beat=bool(rhy["is_beat"]),
            centroid=centroid_norm,
            spectrum_gpu=raw.spectrum_gpu,
            t=0.0,   # rempli par l'AudioEngine
            kick=float(rhy["kick"]), snare=float(rhy["snare"]), hat=float(rhy["hat"]),
            kick_hit=bool(rhy["kick_hit"]), snare_hit=bool(rhy["snare_hit"]),
            hat_hit=bool(rhy["hat_hit"]),
            tonal_warmth=float(warmth_now), key_hue=float(key_hue_now),
            loudness=float(loudness),
            p_bass=float(p_bands[0]), p_low_mid=float(p_bands[1]),
            p_mid=float(p_bands[2]), p_high=float(p_bands[3]),
            sub=float(sub),
        )

    def apply_tempo(self, feats: AudioFeatures, now: float, beat_now: bool) -> None:
        """Avance l'oscillateur de tempo (chaque frame) et écrit les champs
        tempo/phrase dans `feats`. Délègue au RhythmTracker."""
        self.rhythm.apply_tempo(feats, now, beat_now)

    def silent(self, spectrum_gpu) -> AudioFeatures:
        """Cas « rien ne joue » : décroissance douce de toutes les enveloppes.
        Le spectre GPU (déjà décru, valide) est fourni par l'AudioEngine."""
        self._env_bands *= 0.85
        self._env_amp *= 0.85
        self._running_max_bands *= MAX_DECAY
        self._running_max_amp *= MAX_DECAY
        beat, kick_s, snare_s, hat_s = self.rhythm.decay_silent()
        loudness, p_bands, sub = self.perceptual.decay()
        warmth, key_hue = self.harmony.decay()
        return AudioFeatures(
            bass=float(self._env_bands[0]),
            low_mid=float(self._env_bands[1]),
            mid=float(self._env_bands[2]),
            high=float(self._env_bands[3]),
            amplitude=float(self._env_amp),
            beat=float(beat),
            is_beat=False,
            centroid=0.0,
            spectrum_gpu=spectrum_gpu,
            t=0.0,   # rempli par l'AudioEngine
            kick=float(kick_s), snare=float(snare_s), hat=float(hat_s),
            kick_hit=False, snare_hit=False, hat_hit=False,
            tonal_warmth=float(warmth), key_hue=float(key_hue),
            loudness=float(loudness),
            p_bass=float(p_bands[0]), p_low_mid=float(p_bands[1]),
            p_mid=float(p_bands[2]), p_high=float(p_bands[3]),
            sub=float(sub),
        )

    # --- Suivi de bandes / amplitude (AGC doux + enveloppe) -------------------
    def _adaptive_normalize_bands(self, raw: np.ndarray) -> np.ndarray:
        """AGC doux par bande : on divise par un max glissant (suit les pics,
        redescend lentement). Garde les features ~0..1 quel que soit le volume
        système, sans pompage brutal."""
        self._running_max_bands *= MAX_DECAY
        self._running_max_bands = np.maximum(self._running_max_bands, raw)
        self._running_max_bands = np.maximum(self._running_max_bands, 1e-3)
        return raw / self._running_max_bands

    def _adaptive_normalize_amp(self, raw: float) -> float:
        self._running_max_amp *= MAX_DECAY
        self._running_max_amp = max(self._running_max_amp, raw, 1e-3)
        return raw / self._running_max_amp

    def _envelope_follow_bands(self, target: np.ndarray) -> np.ndarray:
        """Suiveur d'enveloppe : attaque rapide à la montée, release lent à la
        descente -> bandes fluides mais réactives."""
        up = target > self._env_bands
        coeff = np.where(up, ATTACK, RELEASE).astype(np.float32)
        self._env_bands = self._env_bands + coeff * (target - self._env_bands)
        return self._env_bands

    def _envelope_follow_amp(self, target: float) -> float:
        coeff = ATTACK if target > self._env_amp else RELEASE
        self._env_amp = self._env_amp + coeff * (target - self._env_amp)
        return self._env_amp
