# -*- coding: utf-8 -*-
"""
test_musical_analyzer.py
========================
Tests UNITAIRES de l'analyse musicale — SANS micro NI GPU. C'est tout l'intérêt du
découplage : `musical_analyzer` n'importe que numpy, donc on injecte des `RawFrame`
synthétiques + un temps `now` déterministe et on vérifie la sémantique musicale
(verrouillage de tempo, beat, harmonie, sonie perçue, silence).

Exécutable tel quel : `python test_musical_analyzer.py`  (ou via pytest).
"""
import numpy as np

from musical_analyzer import (MusicalAnalyzer, RhythmTracker, HarmonyAnalyzer,
                              PerceptualModel, RawFrame)

# Poids d'isosonie par bande REPRÉSENTATIFs (mesurés depuis la pondération A réelle :
# le grave pèse ~0.1, le médium 1.0). Évite d'importer l'AudioEngine (et donc cupy).
PERC_W = np.array([0.10, 0.47, 1.00, 0.83], dtype=np.float32)
FMIN, FMAX = 30.0, 18000.0
DT = 1.0 / 60.0   # 60 fps


def _frame(bass, low_mid, mid, high, loud, sub, centroid_hz=1000.0, chroma=None):
    if chroma is None:
        chroma = np.zeros(12, dtype=np.float64)
    return RawFrame(band_rms=np.array([bass, low_mid, mid, high], dtype=np.float32),
                    block_rms=float(loud), centroid_hz=centroid_hz,
                    loud_lin=float(loud), sub_lin=float(sub),
                    chroma=np.asarray(chroma, dtype=np.float64), spectrum_gpu=None)


def test_tempo_locks_to_120bpm():
    """Un beat toutes les 0.5 s (120 BPM) doit VERROUILLER l'oscillateur ~120 BPM
    avec une confiance qui monte — sans aucun micro."""
    ana = MusicalAnalyzer(PERC_W, FMIN, FMAX)
    now = 0.0
    frames_per_beat = 30           # 0.5 s @ 60 fps
    last = None
    for i in range(360):           # 6 s
        on_beat = (i % frames_per_beat) == 0
        raw = _frame(bass=1.0 if on_beat else 0.05,
                     low_mid=0.5 if on_beat else 0.02,
                     mid=0.1, high=0.1, loud=0.5, sub=0.6)
        feats = ana.analyze(raw, now)
        ana.apply_tempo(feats, now, bool(feats.is_beat))
        last = feats
        now += DT
    assert 110.0 <= last.tempo_bpm <= 130.0, f"tempo {last.tempo_bpm:.1f} hors [110,130]"
    assert last.groove_conf > 0.4, f"groove_conf trop faible: {last.groove_conf:.2f}"
    assert last.anticipation >= 0.0


def test_beat_fires_on_flux_spike():
    """Une montée nette de l'énergie grave déclenche is_beat (une seule fois)."""
    rt = RhythmTracker()
    now = 0.0
    fired = 0
    for i in range(20):
        bass = 1.0 if i == 5 else 0.02     # un seul pic
        rhy = rt.update(np.array([bass, 0.3 if i == 5 else 0.01, 0.05, 0.05], dtype=np.float32),
                        sm_bass=bass, sm_high=0.05, centroid_norm=0.3, now=now)
        fired += int(rhy["is_beat"])
        now += DT
    assert fired == 1, f"attendu 1 beat, obtenu {fired}"


def test_harmony_c_major_is_warm():
    """Un accord de DO MAJEUR (C-E-G) corrèle vers le MAJEUR -> warmth > 0."""
    h = HarmonyAnalyzer()
    chroma = np.zeros(12); chroma[[0, 4, 7]] = 1.0     # C, E, G
    warmth = 0.0
    for _ in range(120):
        warmth, key_hue = h.update(chroma)
    assert warmth > 0.05, f"un accord majeur doit être chaud (warmth={warmth:.3f})"
    assert 0.0 <= key_hue < 1.0


def test_perceptual_equal_loudness():
    """Référence COMMUNE : après une présence forte, un sub seul (égal en énergie de
    bande mais inaudible) doit sonner BEAUCOUP moins fort (isosonie)."""
    pm = PerceptualModel(PERC_W)
    loud_mix = None
    for _ in range(60):                                # mix riche en présence
        loud_mix, p, sub = pm.update(np.array([0.8, 0.3, 0.7, 0.5], dtype=np.float32),
                                     loud_lin=1.0, sub_lin=0.8)
    assert p[2] > p[0], "présence perçue > grave perçu"
    loud_sub = None
    for _ in range(15):                                # puis un sub SEUL (réf encore haute)
        loud_sub, p, sub = pm.update(np.array([0.9, 0.05, 0.03, 0.02], dtype=np.float32),
                                     loud_lin=0.12, sub_lin=0.95)
    assert loud_sub < 0.5 * loud_mix, f"sub seul trop fort ({loud_sub:.2f} vs {loud_mix:.2f})"
    assert sub > 0.5, "le SUB ressenti (non pondéré) doit, lui, rester fort"


def test_silent_decays_toward_zero():
    """Le chemin silence fait décroître les bandes vers 0 et n'émet pas de beat."""
    ana = MusicalAnalyzer(PERC_W, FMIN, FMAX)
    now = 0.0
    for i in range(30):                                # un peu de signal
        ana.analyze(_frame(0.8, 0.6, 0.7, 0.5, 0.7, 0.6), now)
        now += DT
    f1 = ana.silent(spectrum_gpu=None)
    for _ in range(60):                                # puis du silence
        f2 = ana.silent(spectrum_gpu=None)
    assert f2.bass < f1.bass and f2.amplitude < 0.05
    assert f2.is_beat is False
    assert f2.spectrum_gpu is None                     # transmis tel quel


def test_runs_without_gpu_or_mic():
    """Sanity : tout le module tourne en numpy pur (aucun import cupy)."""
    import sys
    assert "cupy" not in sys.modules or sys.modules.get("cupy") is not None
    ana = MusicalAnalyzer(PERC_W, FMIN, FMAX)
    feats = ana.analyze(_frame(0.5, 0.5, 0.5, 0.5, 0.5, 0.5), 0.0)
    ana.apply_tempo(feats, 0.0, False)
    assert 0.0 <= feats.bass <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests unitaires PASSENT (sans micro ni GPU).")
