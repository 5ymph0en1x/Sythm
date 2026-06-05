# -*- coding: utf-8 -*-
"""
config_window.py — Fenêtre de CONFIGURATION de Sythm (au lancement), INTERNATIONALE.
====================================================================================
Ouvre une fenêtre THÉMÉE (TKinterModernThemes, thème « park », mode « dark ») qui
expose les réglages clés de Sythm AVANT que le moteur GPU ne démarre. L'interface
est traduite en 5 langues (EN/DE/FR/IT/ES) via un menu déroulant ; le changement
de langue est INSTANTANÉ (le corps de la fenêtre est reconstruit à la volée).

Les valeurs choisies sont appliquées par main.py et persistées dans
`sythm_config.json` (à côté de l'exécutable), avec la langue retenue.

`run_config()` renvoie : dict (Lancer) / None (annulé) / False (UI indisponible).
"""
from __future__ import annotations

import os
import sys
import json

# ---------------------------------------------------------------------------
#  DÉFAUTS « usine » — SOURCE DE VÉRITÉ UNIQUE du paramétrage de Sythm.
#  main.py et particles.py NE codent plus ces valeurs en dur : ils les LISENT
#  d'ici (`from config_window import DEFAULTS`). C'est donc le SEUL endroit où
#  éditer les valeurs par défaut. La fenêtre les présente/ajuste au lancement, et
#  sythm_config.json (réglages utilisateur) les surcharge à l'exécution.
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Nuée
    "N_PARTICLES": 2_500_000, "CLOUD_RADIUS": 3.5, "PARTICLE_SIZE": 0.5,
    "EXPOSURE": 0.15, "EMIT_RATE": 5.0, "EMITTED_LIFETIME": 10.0,
    # Fenêtre & rendu
    "FULLSCREEN": False, "WINDOW_W": 1280, "WINDOW_H": 720, "VSYNC": True,
    "SUPERSAMPLE_FACTOR": 1.0, "ENABLE_DENOISE": True,
    # Post-FX
    "ENABLE_BLOOM": False, "BLOOM_INTENSITY": 0.3,
    "ENABLE_MOTION_BLUR": False, "MOTION_BLUR_STRENGTH": 0.15, "DENOISE_SIGMA": 0.01,
    # Rythme & flux (particles.py, préfixe _)
    "_FIELD_STRENGTH": 0.9, "_TURB_BASE": 0.25, "_BREATH_OUT": 1.5,
    "_BUILD_CONVERGE": 1.6, "_DROP_BLOOM": 2.5, "_ACCEL_GAIN": 0.4,
    # Couleur & harmonie (particles.py)
    "_TONAL_STRENGTH": 0.7, "_TONAL_GLOW": 0.25, "_WARMTH_HUE": 0.15, "_KEY_HUE_SPAN": 0.12,
    # Caméra & capture
    "CAMERA_MODE": "beat_reactive", "CAMERA_ROTATE_SPEED": 0.15,
    "RECORD_ENCODER": "nvenc", "RECORD_FPS": 30, "RECORD_QUALITY": 21,
    "RECORD_AUDIO_BITRATE": "192k",
}

# GROUPS : (titre, (grid_row, grid_col), [ (clé, libellé, kind, params) ])
#   kind "int"/"float" -> params (lo, hi, step) ; "bool" ; "choice" -> [options]
#   Les titres/libellés sont en FRANÇAIS : ils servent de CLÉS de traduction (cf. TR).
_AUDIO_MUTE = "muet"   # libellé (FR) affiché pour RECORD_AUDIO_BITRATE == "" (pas de son)
_BITRATES = ("192k", "256k", "320k")
GROUPS = [
    ("✦  Nuée", (0, 0), [
        ("N_PARTICLES", "Particules", "int", (100_000, 8_000_000, 100_000)),
        ("CLOUD_RADIUS", "Rayon de la boîte", "float", (0.5, 6.0, 0.1)),
        ("PARTICLE_SIZE", "Taille du point (px)", "float", (0.5, 6.0, 0.5)),
        ("EXPOSURE", "Exposition HDR", "float", (0.02, 1.5, 0.01)),
        ("EMIT_RATE", "Taux d'émission", "float", (0.0, 20.0, 1.0)),
        ("EMITTED_LIFETIME", "Vie des traînées (s)", "float", (1.0, 20.0, 1.0)),
    ]),
    ("🪟  Fenêtre & rendu", (0, 1), [
        ("FULLSCREEN", "Plein écran", "bool", None),
        ("WINDOW_W", "Largeur (px)", "int", (640, 7680, 160)),
        ("WINDOW_H", "Hauteur (px)", "int", (360, 4320, 90)),
        ("VSYNC", "VSync", "bool", None),
        ("SUPERSAMPLE_FACTOR", "Supersampling", "float", (0.5, 2.0, 0.25)),
        ("ENABLE_DENOISE", "Débruitage", "bool", None),
    ]),
    ("🌌  Post-FX", (0, 2), [
        ("ENABLE_BLOOM", "Bloom", "bool", None),
        ("BLOOM_INTENSITY", "Intensité bloom", "float", (0.0, 2.0, 0.05)),
        ("ENABLE_MOTION_BLUR", "Motion blur", "bool", None),
        ("MOTION_BLUR_STRENGTH", "Force motion blur", "float", (0.0, 1.0, 0.05)),
        ("DENOISE_SIGMA", "Sigma débruitage", "float", (0.001, 0.1, 0.001)),
    ]),
    ("🥁  Rythme & flux", (1, 0), [
        ("_FIELD_STRENGTH", "Force du champ", "float", (0.0, 3.0, 0.1)),
        ("_TURB_BASE", "Turbulence", "float", (0.0, 1.0, 0.05)),
        ("_BREATH_OUT", "Respiration (expir)", "float", (0.0, 4.0, 0.1)),
        ("_BUILD_CONVERGE", "Build : charge", "float", (0.0, 4.0, 0.1)),
        ("_DROP_BLOOM", "Drop : explosion", "float", (0.0, 6.0, 0.25)),
        ("_ACCEL_GAIN", "Étincelle de cisaillement", "float", (0.0, 1.5, 0.05)),
    ]),
    ("🎨  Couleur & harmonie", (1, 1), [
        ("_TONAL_STRENGTH", "Relief tonal", "float", (0.0, 2.0, 0.05)),
        ("_TONAL_GLOW", "Lueur des strates", "float", (0.0, 1.0, 0.05)),
        ("_WARMTH_HUE", "Chaud/froid (modalité)", "float", (0.0, 0.4, 0.01)),
        ("_KEY_HUE_SPAN", "Teinte par tonalité", "float", (0.0, 0.4, 0.01)),
    ]),
    ("🎥  Caméra & capture", (1, 2), [
        ("CAMERA_MODE", "Mode caméra", "choice", ["fixed", "auto_rotate", "beat_reactive"]),
        ("CAMERA_ROTATE_SPEED", "Vitesse rotation", "float", (0.0, 1.0, 0.05)),
        ("RECORD_ENCODER", "Encodeur", "choice", ["nvenc", "x265"]),
        ("RECORD_FPS", "FPS capture", "int", (24, 120, 1)),
        ("RECORD_QUALITY", "Qualité (CQ/CRF)", "int", (10, 30, 1)),
        ("RECORD_AUDIO_BITRATE", "Audio", "choice", ["192k", "256k", "320k", _AUDIO_MUTE]),
    ]),
]

MAIN_KEYS = [k for k in DEFAULTS if not k.startswith("_")]
PARTICLE_KEYS = [k for k in DEFAULTS if k.startswith("_")]
_KIND = {key: kind for _t0, _p, items in GROUPS for (key, _l, kind, _pp) in items}

# ---------------------------------------------------------------------------
#  PRESETS VISUELS
#  Chaque preset définit un look visuel cohérent en ajustant densité, flux,
#  post-traitement et caméra. Ils n'incluent PAS les paramètres techniques
#  (RECORD_*).
# ---------------------------------------------------------------------------
PRESETS = {
    "Ambiant": {
        "N_PARTICLES": 1_200_000,
        "CLOUD_RADIUS": 4.5,
        "PARTICLE_SIZE": 1.2,
        "EXPOSURE": 0.28,
        "EMIT_RATE": 1.8,
        "EMITTED_LIFETIME": 18.0,
        "ENABLE_BLOOM": True,
        "BLOOM_INTENSITY": 0.55,
        "ENABLE_MOTION_BLUR": True,
        "MOTION_BLUR_STRENGTH": 0.40,
        "_FIELD_STRENGTH": 0.55,
        "_TURB_BASE": 0.12,
        "_BREATH_OUT": 0.7,
        "_BUILD_CONVERGE": 1.2,
        "_DROP_BLOOM": 1.8,
        "_TONAL_STRENGTH": 1.4,
        "CAMERA_MODE": "auto_rotate",
        "CAMERA_ROTATE_SPEED": 0.08,
    },
    "Minimal": {
        "N_PARTICLES": 700_000,
        "CLOUD_RADIUS": 2.6,
        "PARTICLE_SIZE": 0.55,
        "EXPOSURE": 0.38,
        "EMIT_RATE": 0.6,
        "EMITTED_LIFETIME": 7.0,
        "ENABLE_BLOOM": False,
        "ENABLE_MOTION_BLUR": False,
        "_FIELD_STRENGTH": 0.65,
        "_TURB_BASE": 0.06,
        "_TONAL_STRENGTH": 0.35,
        "CAMERA_MODE": "fixed",
    },
    "Énergétique": {
        "N_PARTICLES": 1_800_000,
        "CLOUD_RADIUS": 3.0,
        "PARTICLE_SIZE": 0.42,
        "EXPOSURE": 0.13,
        "EMIT_RATE": 4.5,
        "EMITTED_LIFETIME": 4.0,
        "ENABLE_BLOOM": True,
        "BLOOM_INTENSITY": 0.85,
        "ENABLE_MOTION_BLUR": False,
        "_FIELD_STRENGTH": 1.25,
        "_TURB_BASE": 0.38,
        "_BREATH_OUT": 2.2,
        "_BUILD_CONVERGE": 1.8,
        "_DROP_BLOOM": 4.2,
        "_ACCEL_GAIN": 0.65,
        "_TONAL_STRENGTH": 0.6,
        "CAMERA_MODE": "beat_reactive",
    },
    "Cosmique": {
        "N_PARTICLES": 2_600_000,
        "CLOUD_RADIUS": 5.2,
        "PARTICLE_SIZE": 0.95,
        "EXPOSURE": 0.17,
        "EMIT_RATE": 3.2,
        "EMITTED_LIFETIME": 24.0,
        "ENABLE_BLOOM": True,
        "BLOOM_INTENSITY": 0.65,
        "ENABLE_MOTION_BLUR": True,
        "MOTION_BLUR_STRENGTH": 0.50,
        "_FIELD_STRENGTH": 0.72,
        "_TURB_BASE": 0.20,
        "_BREATH_OUT": 1.0,
        "_TONAL_STRENGTH": 1.9,
        "_TONAL_GLOW": 0.42,
        "CAMERA_MODE": "auto_rotate",
        "CAMERA_ROTATE_SPEED": 0.045,
    },
    "Percussif": {
        "N_PARTICLES": 2_400_000,
        "CLOUD_RADIUS": 2.9,
        "PARTICLE_SIZE": 0.48,
        "EXPOSURE": 0.20,
        "EMIT_RATE": 7.5,
        "EMITTED_LIFETIME": 4.8,
        "ENABLE_BLOOM": True,
        "BLOOM_INTENSITY": 1.05,
        "_FIELD_STRENGTH": 1.15,
        "_TURB_BASE": 0.38,
        "_BREATH_OUT": 2.4,
        "_BUILD_CONVERGE": 0.9,
        "_DROP_BLOOM": 6.0,
        "_ACCEL_GAIN": 0.92,
        "CAMERA_MODE": "beat_reactive",
    },
}

# Descriptions courtes des presets (FR = clé de traduction, cf. TR).
PRESET_DESC = {
    "Ambiant": "Nuage doux, traînées longues, caméra lente, tonalité présente.",
    "Minimal": "Faible densité, contraste net, pas d’effets, caméra fixe.",
    "Énergétique": "Dense et réactif, bloom fort, ondes de choc marquées.",
    "Cosmique": "Grand espace, longues traînées, relief tonal fort, rotation lente.",
    "Percussif": "Rythme visible (fronts d’onde), bloom intense, caméra beat-reactive.",
}

# ---------------------------------------------------------------------------
#  INTERNATIONALISATION. Le FRANÇAIS est la clé ; chaque langue mappe clé -> texte.
#  Les chaînes absentes d'une langue retombent sur le français (termes techniques
#  identiques : VSync, Bloom, Motion blur, Post-FX, Supersampling, Audio…).
# ---------------------------------------------------------------------------
DEFAULT_LANG = "en"
LANG_ORDER = ["en", "de", "fr", "it", "es"]            # ordre du menu déroulant
AUTONYMS = {"en": "English", "de": "Deutsch", "fr": "Français",
            "it": "Italiano", "es": "Español"}
_NAME2CODE = {v: k for k, v in AUTONYMS.items()}

TR = {
    "en": {
        "Sythm — Configuration": "Sythm — Settings",
        "Réglages — ajuste, puis lance. (sauvegardés pour la prochaine fois)":
            "Settings — adjust, then launch. (saved for next time)",
        "Langue": "Language", "Réinitialiser": "Reset", "Lancer Sythm  ▶": "Launch Sythm  ▶",
        "muet": "mute",
        "✦  Nuée": "✦  Cloud", "🪟  Fenêtre & rendu": "🪟  Window & rendering",
        "🥁  Rythme & flux": "🥁  Rhythm & flow", "🎨  Couleur & harmonie": "🎨  Colour & harmony",
        "🎥  Caméra & capture": "🎥  Camera & capture",
        "Particules": "Particles", "Rayon de la boîte": "Box radius",
        "Taille du point (px)": "Point size (px)", "Exposition HDR": "HDR exposure",
        "Taux d'émission": "Emission rate", "Vie des traînées (s)": "Trail lifetime (s)",
        "Plein écran": "Fullscreen", "Largeur (px)": "Width (px)", "Hauteur (px)": "Height (px)",
        "Débruitage": "Denoise", "Intensité bloom": "Bloom intensity",
        "Force motion blur": "Motion blur strength", "Sigma débruitage": "Denoise sigma",
        "Force du champ": "Field strength", "Respiration (expir)": "Breath (exhale)",
        "Build : charge": "Build: charge", "Drop : explosion": "Drop: blast",
        "Étincelle de cisaillement": "Shear sparkle", "Relief tonal": "Tonal relief",
        "Lueur des strates": "Strata glow", "Chaud/froid (modalité)": "Warm/cold (mode)",
        "Teinte par tonalité": "Hue by key", "Mode caméra": "Camera mode",
        "Vitesse rotation": "Rotation speed", "Encodeur": "Encoder",
        "FPS capture": "Capture FPS", "Qualité (CQ/CRF)": "Quality (CQ/CRF)",
        "Presets visuels": "Visual presets", "Appliquer ce preset": "Apply preset",
        "Ambiant": "Ambient", "Minimal": "Minimal", "Énergétique": "Energetic",
        "Cosmique": "Cosmic", "Percussif": "Percussive",
        "Nuage doux, traînées longues, caméra lente, tonalité présente.":
            "Soft cloud, long trails, slow camera, present tonality.",
        "Faible densité, contraste net, pas d’effets, caméra fixe.":
            "Low density, crisp contrast, no effects, fixed camera.",
        "Dense et réactif, bloom fort, ondes de choc marquées.":
            "Dense and reactive, strong bloom, pronounced shockwaves.",
        "Grand espace, longues traînées, relief tonal fort, rotation lente.":
            "Vast space, long trails, strong tonal relief, slow rotation.",
        "Rythme visible (fronts d’onde), bloom intense, caméra beat-reactive.":
            "Visible rhythm (wavefronts), intense bloom, beat-reactive camera.",
    },
    "de": {
        "Sythm — Configuration": "Sythm — Konfiguration",
        "Réglages — ajuste, puis lance. (sauvegardés pour la prochaine fois)":
            "Einstellungen — anpassen, dann starten. (für später gespeichert)",
        "Langue": "Sprache", "Réinitialiser": "Zurücksetzen", "Lancer Sythm  ▶": "Sythm starten  ▶",
        "muet": "stumm",
        "✦  Nuée": "✦  Wolke", "🪟  Fenêtre & rendu": "🪟  Fenster & Rendering",
        "🥁  Rythme & flux": "🥁  Rhythmus & Fluss", "🎨  Couleur & harmonie": "🎨  Farbe & Harmonie",
        "🎥  Caméra & capture": "🎥  Kamera & Aufnahme",
        "Particules": "Partikel", "Rayon de la boîte": "Box-Radius",
        "Taille du point (px)": "Punktgröße (px)", "Exposition HDR": "HDR-Belichtung",
        "Taux d'émission": "Emissionsrate", "Vie des traînées (s)": "Spur-Lebensdauer (s)",
        "Plein écran": "Vollbild", "Largeur (px)": "Breite (px)", "Hauteur (px)": "Höhe (px)",
        "VSync": "VSync", "Débruitage": "Entrauschen", "Intensité bloom": "Bloom-Intensität",
        "Force motion blur": "Motion-Blur-Stärke", "Sigma débruitage": "Entrausch-Sigma",
        "Force du champ": "Feldstärke", "Turbulence": "Turbulenz",
        "Respiration (expir)": "Atem (Ausatmen)", "Build : charge": "Build: Aufladung",
        "Drop : explosion": "Drop: Explosion", "Étincelle de cisaillement": "Scherungs-Funkeln",
        "Relief tonal": "Tonales Relief", "Lueur des strates": "Schichten-Leuchten",
        "Chaud/froid (modalité)": "Warm/kalt (Modus)", "Teinte par tonalité": "Farbton nach Tonart",
        "Mode caméra": "Kameramodus", "Vitesse rotation": "Drehgeschwindigkeit",
        "Encodeur": "Encoder", "FPS capture": "Aufnahme-FPS", "Qualité (CQ/CRF)": "Qualität (CQ/CRF)",
        "Presets visuels": "Visuelle Presets", "Appliquer ce preset": "Preset anwenden",
        "Ambiant": "Ambient", "Minimal": "Minimal", "Énergétique": "Energetisch",
        "Cosmique": "Kosmisch", "Percussif": "Perkussiv",
        "Nuage doux, traînées longues, caméra lente, tonalité présente.":
            "Weiche Wolke, lange Spuren, langsame Kamera, präsente Tonalität.",
        "Faible densité, contraste net, pas d’effets, caméra fixe.":
            "Geringe Dichte, klarer Kontrast, keine Effekte, feste Kamera.",
        "Dense et réactif, bloom fort, ondes de choc marquées.":
            "Dicht und reaktiv, starker Bloom, ausgeprägte Schockwellen.",
        "Grand espace, longues traînées, relief tonal fort, rotation lente.":
            "Weiter Raum, lange Spuren, starkes tonales Relief, langsame Drehung.",
        "Rythme visible (fronts d’onde), bloom intense, caméra beat-reactive.":
            "Sichtbarer Rhythmus (Wellenfronten), intensiver Bloom, beat-reaktive Kamera.",
    },
    "it": {
        "Sythm — Configuration": "Sythm — Configurazione",
        "Réglages — ajuste, puis lance. (sauvegardés pour la prochaine fois)":
            "Impostazioni — regola, poi avvia. (salvate per la prossima volta)",
        "Langue": "Lingua", "Réinitialiser": "Reimposta", "Lancer Sythm  ▶": "Avvia Sythm  ▶",
        "muet": "muto",
        "✦  Nuée": "✦  Nuvola", "🪟  Fenêtre & rendu": "🪟  Finestra & rendering",
        "🥁  Rythme & flux": "🥁  Ritmo & flusso", "🎨  Couleur & harmonie": "🎨  Colore & armonia",
        "🎥  Caméra & capture": "🎥  Camera & cattura",
        "Particules": "Particelle", "Rayon de la boîte": "Raggio del box",
        "Taille du point (px)": "Dim. punto (px)", "Exposition HDR": "Esposizione HDR",
        "Taux d'émission": "Tasso d'emissione", "Vie des traînées (s)": "Durata scie (s)",
        "Plein écran": "Schermo intero", "Largeur (px)": "Larghezza (px)", "Hauteur (px)": "Altezza (px)",
        "Débruitage": "Riduzione rumore", "Intensité bloom": "Intensità bloom",
        "Force motion blur": "Forza motion blur", "Sigma débruitage": "Sigma riduzione",
        "Force du champ": "Forza del campo", "Turbulence": "Turbolenza",
        "Respiration (expir)": "Respiro (espir.)", "Build : charge": "Build: carica",
        "Drop : explosion": "Drop: esplosione", "Étincelle de cisaillement": "Scintilla di taglio",
        "Relief tonal": "Rilievo tonale", "Lueur des strates": "Bagliore strati",
        "Chaud/froid (modalité)": "Caldo/freddo (modo)", "Teinte par tonalité": "Tinta per tonalità",
        "Mode caméra": "Modo camera", "Vitesse rotation": "Velocità rotazione",
        "Encodeur": "Encoder", "FPS capture": "FPS cattura", "Qualité (CQ/CRF)": "Qualità (CQ/CRF)",
        "Presets visuels": "Preset visivi", "Appliquer ce preset": "Applica preset",
        "Ambiant": "Ambient", "Minimal": "Minimale", "Énergétique": "Energico",
        "Cosmique": "Cosmico", "Percussif": "Percussivo",
        "Nuage doux, traînées longues, caméra lente, tonalité présente.":
            "Nuvola morbida, scie lunghe, camera lenta, tonalità presente.",
        "Faible densité, contraste net, pas d’effets, caméra fixe.":
            "Bassa densità, contrasto netto, nessun effetto, camera fissa.",
        "Dense et réactif, bloom fort, ondes de choc marquées.":
            "Denso e reattivo, bloom forte, onde d'urto marcate.",
        "Grand espace, longues traînées, relief tonal fort, rotation lente.":
            "Grande spazio, scie lunghe, forte rilievo tonale, rotazione lenta.",
        "Rythme visible (fronts d’onde), bloom intense, caméra beat-reactive.":
            "Ritmo visibile (fronti d'onda), bloom intenso, camera beat-reactive.",
    },
    "es": {
        "Sythm — Configuration": "Sythm — Configuración",
        "Réglages — ajuste, puis lance. (sauvegardés pour la prochaine fois)":
            "Ajustes — configura y lanza. (guardados para la próxima vez)",
        "Langue": "Idioma", "Réinitialiser": "Restablecer", "Lancer Sythm  ▶": "Iniciar Sythm  ▶",
        "muet": "silencio",
        "✦  Nuée": "✦  Nube", "🪟  Fenêtre & rendu": "🪟  Ventana y render",
        "🥁  Rythme & flux": "🥁  Ritmo y flujo", "🎨  Couleur & harmonie": "🎨  Color y armonía",
        "🎥  Caméra & capture": "🎥  Cámara y captura",
        "Particules": "Partículas", "Rayon de la boîte": "Radio de la caja",
        "Taille du point (px)": "Tamaño punto (px)", "Exposition HDR": "Exposición HDR",
        "Taux d'émission": "Tasa de emisión", "Vie des traînées (s)": "Vida de estelas (s)",
        "Plein écran": "Pantalla completa", "Largeur (px)": "Ancho (px)", "Hauteur (px)": "Alto (px)",
        "Débruitage": "Reducción de ruido", "Intensité bloom": "Intensidad bloom",
        "Force motion blur": "Fuerza motion blur", "Sigma débruitage": "Sigma de ruido",
        "Force du champ": "Fuerza del campo", "Turbulence": "Turbulencia",
        "Respiration (expir)": "Respiración (exhal.)", "Build : charge": "Build: carga",
        "Drop : explosion": "Drop: explosión", "Étincelle de cisaillement": "Destello de cizalla",
        "Relief tonal": "Relieve tonal", "Lueur des strates": "Brillo de estratos",
        "Chaud/froid (modalité)": "Cálido/frío (modo)", "Teinte par tonalité": "Tono por tonalidad",
        "Mode caméra": "Modo de cámara", "Vitesse rotation": "Velocidad de rotación",
        "Encodeur": "Codificador", "FPS capture": "FPS de captura", "Qualité (CQ/CRF)": "Calidad (CQ/CRF)",
        "Presets visuels": "Presets visuales", "Appliquer ce preset": "Aplicar preset",
        "Ambiant": "Ambiente", "Minimal": "Minimal", "Énergétique": "Enérgico",
        "Cosmique": "Cósmico", "Percussif": "Percusivo",
        "Nuage doux, traînées longues, caméra lente, tonalité présente.":
            "Nube suave, estelas largas, cámara lenta, tonalidad presente.",
        "Faible densité, contraste net, pas d’effets, caméra fixe.":
            "Baja densidad, contraste nítido, sin efectos, cámara fija.",
        "Dense et réactif, bloom fort, ondes de choc marquées.":
            "Denso y reactivo, bloom fuerte, ondas de choque marcadas.",
        "Grand espace, longues traînées, relief tonal fort, rotation lente.":
            "Gran espacio, estelas largas, fuerte relieve tonal, rotación lenta.",
        "Rythme visible (fronts d’onde), bloom intense, caméra beat-reactive.":
            "Ritmo visible (frentes de onda), bloom intenso, cámara beat-reactive.",
    },
}


# ---------------------------------------------------------------------------
#  Persistance (sythm_config.json à côté de l'exécutable / du script).
# ---------------------------------------------------------------------------
def _config_path() -> str:
    base = (os.path.dirname(os.path.abspath(sys.executable))
            if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "sythm_config.json")


def _read_json() -> dict:
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_saved() -> dict:
    """Réglages persistés (filtrés aux clés connues ; ignore __lang__)."""
    return {k: v for k, v in _read_json().items() if k in DEFAULTS}


def load_lang() -> str:
    c = _read_json().get("__lang__")
    return c if c in AUTONYMS else DEFAULT_LANG


def save_saved(values: dict, lang: str = None) -> None:
    try:
        d = dict(values)
        d["__lang__"] = lang if lang in AUTONYMS else _read_json().get("__lang__", DEFAULT_LANG)
        with open(_config_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Fenêtre de configuration (internationale, changement de langue à chaud).
# ---------------------------------------------------------------------------
def run_config(_test_build_only: bool = False):
    """Affiche la fenêtre de config. Voir le docstring du module pour le contrat
    de retour (dict / None / False)."""
    try:
        import tkinter as tk
        import TKinterModernThemes as TKMT
    except Exception as exc:
        print(f"[config] UI indisponible ({exc}) -> valeurs par défaut.", file=sys.stderr)
        return False

    values = dict(DEFAULTS)
    values.update(load_saved())

    class ConfigApp(TKMT.ThemedTKinterFrame):
        def __init__(self):
            # usecommandlineargs/useconfigfile=False : on FORCE park/dark.
            super().__init__("Sythm — Configuration", "park", "dark",
                             usecommandlineargs=False, useconfigfile=False)
            self.launched = False
            self.vars = {}
            self.lang = load_lang()
            self._lang_var = tk.StringVar(value=AUTONYMS[self.lang])
            self._body = None
            self.root.protocol("WM_DELETE_WINDOW", self._close)
            self._build_body()

        # ------ i18n ------
        def _t(self, s):
            return TR.get(self.lang, {}).get(s, s)

        def _make_var(self, key):
            v = values[key]
            kind = _KIND[key]
            if kind == "bool":
                return tk.BooleanVar(value=bool(v))
            if kind == "choice":
                disp = self._t(_AUDIO_MUTE) if (key == "RECORD_AUDIO_BITRATE" and v == "") else str(v)
                return tk.StringVar(value=disp)
            if kind == "int":
                return tk.IntVar(value=int(v))
            return tk.DoubleVar(value=float(v))

        # ------ construction / reconstruction du corps ------
        def _build_body(self):
            if self._body is not None:
                try:
                    self._body.master.destroy()    # détruit l'ancien corps (et ses enfants)
                    # retire l'ancienne entrée du suivi interne de TKMT, sinon le
                    # re-placement à (0,0) déclenche un warning « Overlapping widgets ».
                    self.widgets.widgetlist = [w for w in self.widgets.widgetlist
                                               if getattr(w, "widget", None) is not self._body]
                except Exception:
                    pass
            self._body = self.addFrame("body", row=0, col=0)
            b = self._body

            top = b.addFrame("top", row=0, col=0, colspan=3)
            top.Label("S Y T H M", size=22, row=0, col=1)          # CENTRÉ (col 1, entre 2 colonnes égales)
            top.OptionMenu([AUTONYMS[c] for c in LANG_ORDER], self._lang_var,
                           command=self._on_lang, default=AUTONYMS[self.lang],
                           row=0, col=2, sticky="e")                # sélecteur de langue, collé à DROITE
            try:
                # col 0 et col 2 forcées à largeurs ÉGALES (uniform) -> "S Y T H M"
                # (col 1) parfaitement centré dans la fenêtre, le menu (col 2) à droite.
                top.master.columnconfigure(0, weight=1, uniform="hdr")
                top.master.columnconfigure(2, weight=1, uniform="hdr")
            except Exception:
                pass

            # --- Sélecteur de presets visuels (traduit) ----------------------
            preset_fr = b.addLabelFrame(self._t("Presets visuels"), row=3, col=0, colspan=3)
            if not hasattr(self, "_preset_key"):
                self._preset_key = next(iter(PRESETS), "")
            # La var porte le nom AFFICHÉ (traduit) ; le nom CANONIQUE (clé FR de
            # PRESETS) reste self._preset_key -> survit au changement de langue.
            self._preset_var = tk.StringVar(value=self._t(self._preset_key))
            preset_fr.OptionMenu([self._t(n) for n in PRESETS], self._preset_var,
                                 command=self._on_preset, default=self._t(self._preset_key),
                                 row=0, col=0, sticky="w")
            preset_fr.Button(self._t("Appliquer ce preset"), self._apply_preset,
                             row=0, col=1, sticky="w")
            self.preset_desc = preset_fr.Label(self._t(PRESET_DESC.get(self._preset_key, "")),
                                               size=9, weight="normal", row=0, col=2, sticky="w")

            for title, (gr, gc), items in GROUPS:
                fr = b.addLabelFrame(self._t(title), row=gr + 1, col=gc)
                for i, (key, label, kind, params) in enumerate(items):
                    fr.Label(self._t(label), size=10, weight="normal", row=i, col=0, sticky="w")
                    var = self.vars.get(key) or self.vars.setdefault(key, self._make_var(key))
                    if kind == "bool":
                        fr.SlideSwitch("", var, row=i, col=1)
                    elif kind == "choice":
                        opts = [self._t(_AUDIO_MUTE) if o == _AUDIO_MUTE else o for o in params]
                        fr.OptionMenu(opts, var, command=lambda *a: None,
                                      default=var.get(), row=i, col=1)
                    elif kind == "int":
                        lo, hi, st = params
                        fr.NumericalSpinbox(lo, hi, st, var, row=i, col=1)
                    else:
                        lo, hi, st = params
                        fr.NumericalSpinbox(lo, hi, st, var, row=i, col=1)

            bar = b.addFrame("bar", row=4, col=0, colspan=3)
            bar.Button(self._t("Réinitialiser"), self._reset, row=0, col=0)
            bar.AccentButton(self._t("Lancer Sythm  ▶"), self._launch, row=0, col=1)
            try:
                self.root.update_idletasks()
            except Exception:
                pass

        def _on_lang(self, *a):
            code = _NAME2CODE.get(self._lang_var.get(), DEFAULT_LANG)
            if code == self.lang:
                return
            # Ré-affiche le « muet » audio dans la nouvelle langue (toute valeur
            # non-débit = muet).
            av = self.vars.get("RECORD_AUDIO_BITRATE")
            if av is not None and av.get() not in _BITRATES:
                av.set(TR.get(code, {}).get(_AUDIO_MUTE, _AUDIO_MUTE))
            self.lang = code
            # Reconstruction DIFFÉRÉE : on ne détruit pas le menu pendant son
            # propre callback (sinon erreur Tk).
            self.root.after(1, self._relang)

        def _relang(self):
            self._build_body()
            try:
                self.root.title(self._t("Sythm — Configuration"))
            except Exception:
                pass

        def _reset(self):
            for key, var in self.vars.items():
                d = DEFAULTS[key]
                if key == "RECORD_AUDIO_BITRATE" and d == "":
                    d = self._t(_AUDIO_MUTE)
                var.set(d)

        def _on_preset(self, *a):
            # nom AFFICHÉ (traduit) -> clé canonique FR ; met à jour la description.
            disp = self._preset_var.get()
            self._preset_key = {self._t(n): n for n in PRESETS}.get(disp, self._preset_key)
            try:
                self.preset_desc.config(text=self._t(PRESET_DESC.get(self._preset_key, "")))
            except Exception:
                pass

        def _apply_preset(self):
            """Applique le preset sélectionné (clé canonique) aux variables de l'UI."""
            preset = PRESETS.get(self._preset_key)
            if not preset:
                return
            for key, value in preset.items():
                if key in self.vars:
                    var = self.vars[key]
                    try:
                        if isinstance(var, tk.BooleanVar):
                            var.set(bool(value))
                        elif isinstance(var, (tk.IntVar, tk.DoubleVar)):
                            var.set(value)
                        else:
                            var.set(str(value))
                    except Exception:
                        pass
            print(f"[config] Preset « {self._preset_key} » appliqué.")

        def _launch(self):
            self.launched = True
            self.root.quit()

        def _close(self):
            self.launched = False
            self.root.quit()

        def read_values(self) -> dict:
            out = {}
            for key, var in self.vars.items():
                kind = _KIND[key]
                try:
                    if kind == "bool":
                        out[key] = bool(var.get())
                    elif kind == "int":
                        out[key] = int(round(float(var.get())))
                    elif kind == "float":
                        out[key] = float(var.get())
                    elif key == "RECORD_AUDIO_BITRATE":
                        s = str(var.get())
                        out[key] = s if s in _BITRATES else ""   # tout sauf un débit = muet
                    else:
                        out[key] = str(var.get())
                except Exception:
                    out[key] = DEFAULTS[key]
            return out

    app = ConfigApp()
    app.root.title(app._t("Sythm — Configuration"))

    if _test_build_only:                 # smoke : presets + changements de langue.
        app.root.update()
        app._preset_key = "Minimal"; app._apply_preset()   # applique un preset
        assert int(round(float(app.vars["N_PARTICLES"].get()))) == PRESETS["Minimal"]["N_PARTICLES"]
        for _lc in ("fr", "de", "es", "en"):
            app._lang_var.set(AUTONYMS[_lc]); app._on_lang()
            app.root.update(); app.root.update()  # traite le after(1) -> reconstruction
            assert app.lang == _lc, f"langue {_lc} non prise"
        app._preset_var.set(app._t("Cosmique")); app._on_preset()   # nom affiché -> clé
        assert app._preset_key == "Cosmique", "mapping nom de preset traduit KO"
        app.root.destroy()
        return True

    try:
        app.run()                        # boucle (quittée par _launch / _close)
    except Exception as exc:
        print(f"[config] erreur UI ({exc}) -> valeurs par défaut.", file=sys.stderr)
        try:
            app.root.destroy()
        except Exception:
            pass
        return False

    lang = app.lang
    if not app.launched:                 # fenêtre fermée -> annulation.
        try:
            app.root.destroy()
        except Exception:
            pass
        return None

    out = app.read_values()
    try:
        app.root.destroy()
    except Exception:
        pass
    save_saved(out, lang)
    return out


if __name__ == "__main__":
    cfg = run_config()
    print("Config retournée :", cfg)