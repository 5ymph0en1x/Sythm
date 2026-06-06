# -*- coding: utf-8 -*-
"""
stereo.py
=========
EMPAQUETAGE STÉRÉOSCOPIQUE « FRAME PACKING » 1080p (HDMI 1.4a) pour Sythm.

Vraie stéréoscopie, conçue DÈS LE RENDU (pas du faux-3D reprojeté) : la MÊME
simulation de particules (un seul ParticleSystem, un seul jeu de buffers GL) est
dessinée DEUX FOIS par le Renderer, depuis deux caméras OFF-AXIS (axes
parallèles + frustums asymétriques -> parallaxe nulle au plan de convergence ;
cf. renderer._eye_view_proj). Les deux images sont ensuite empaquetées dans le
format frame packing de la norme :

        +-----------------------------+  ligne 0     (haut de l'image)
        |        OEIL GAUCHE          |  1920 x 1080
        +-----------------------------+  ligne 1080
        |   active space (45 lignes)  |  bande de garde NOIRE
        +-----------------------------+  ligne 1125
        |        OEIL DROIT           |  1920 x 1080
        +-----------------------------+  ligne 2205   (bas de l'image)

   Total : 1920 x 2205, présenté à 24 Hz. Sur un afficheur/projecteur réglé en
   frame packing 1080p, c'est le signal 3D exact.

RÔLE DE CE MODULE :
  * Posséder DEUX chaînes PostProcessor indépendantes (une par sous-image) : leurs
    historiques de motion blur restent séparés -> aucun « fantôme » d'un oeil sur
    l'autre.
  * À chaque frame : effacer tout le cadre empaqueté (la bande de garde reste
    noire), puis demander au Renderer l'oeil du HAUT puis l'oeil du BAS et composer
    chacun, via PostProcessor.process(..., viewport=...), dans son sous-rectangle.

RAPPEL ORIGINE GL : le framebuffer a son origine en BAS-À-GAUCHE. L'oeil du HAUT
de l'image occupe donc les lignes GL [hauteur-1080 .. hauteur), l'oeil du BAS les
lignes GL [0 .. 1080). On ancre les deux yeux respectivement en haut et en bas du
framebuffer réel : si celui-ci fait exactement 2205, la bande de garde fait
exactement 45 lignes (norme) ; sinon elle absorbe l'écart (dégradation propre).

NOTE « sans GPU » : ce module n'importe NI moderngl NI cupy ; il ne manipule que
des objets fournis par l'Integrator (ctx, renderer, fabrique de PostProcessor).
"""

from __future__ import annotations


class StereoRig:
    """Rendu + empaquetage stéréo frame packing 1080p (cf. en-tête du module).

    Contrat (utilisé par main.py comme un PostProcessor « augmenté ») :
        def __init__(self, ctx, renderer, make_postfx, swap_eyes=False)
        def render_frame(self, screen, fb_w, fb_h)   # dessine + empaquette 1 frame
        def set_params(self, **kwargs)               # relaye aux deux chaînes
        def resize(self, *a)                         # no-op (cadre figé en stéréo)
        def release(self)                            # libère les deux chaînes
    """

    # Géométrie EXACTE du frame packing 1080p (HDMI 1.4a). NE PAS modifier.
    EYE_W = 1920
    EYE_H = 1080
    GAP = 45
    TOTAL_H = EYE_H + GAP + EYE_H   # 2205
    FPS = 24

    def __init__(self, ctx, renderer, make_postfx, swap_eyes=False):
        """:param ctx:         contexte moderngl (pour cibler/effacer l'écran).
        :param renderer:    Renderer ; doit exposer render(eye=-1/+1) (off-axis).
        :param make_postfx: fabrique SANS argument renvoyant un PostProcessor neuf
                            configuré pour UN oeil (render = screen = 1920x1080).
                            On en crée DEUX (haut/bas) -> historiques séparés.
        :param swap_eyes:   inverse quel oeil-caméra alimente le HAUT et le BAS
                            (confort, ou matériel à canaux G/D inversés)."""
        self.ctx = ctx
        self.renderer = renderer
        self.swap_eyes = bool(swap_eyes)
        # Une chaîne post-FX par SOUS-IMAGE (position fixe -> historique cohérent).
        self.postfx_top = make_postfx()
        self.postfx_bottom = make_postfx()

    # ------------------------------------------------------------------ #
    #  Réglages / cycle de vie (interface façon PostProcessor)           #
    # ------------------------------------------------------------------ #
    def set_params(self, **kwargs):
        """Relaye les réglages à chaud aux DEUX chaînes (bloom/expo/denoise...)."""
        for p in (self.postfx_top, self.postfx_bottom):
            if hasattr(p, "set_params"):
                try:
                    p.set_params(**kwargs)
                except Exception:
                    pass

    def resize(self, *args):
        """No-op : la géométrie frame packing est FIGÉE (la fenêtre stéréo ne se
        redimensionne pas). Présent pour rester interchangeable avec PostProcessor."""
        return None

    def release(self):
        """Libère les ressources GL des deux chaînes post-FX."""
        for p in (self.postfx_top, self.postfx_bottom):
            if hasattr(p, "release"):
                try:
                    p.release()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    #  Rendu d'une frame stéréo empaquetée                               #
    # ------------------------------------------------------------------ #
    def render_frame(self, screen, fb_w, fb_h):
        """Dessine les DEUX yeux et les empaquette dans `screen` (framebuffer par
        défaut, taille réelle fb_w x fb_h).

        Suppose que update_camera() a DÉJÀ été appelé sur le renderer cette frame
        (le rig central est commun aux deux yeux). Effacement complet d'abord ->
        la bande de garde (et tout débordement) reste NOIRE.
        """
        fb_w = int(fb_w)
        fb_h = int(fb_h)

        # 1) Efface TOUT le cadre empaqueté (bande de garde noire incluse).
        screen.use()
        try:
            screen.viewport = (0, 0, fb_w, fb_h)
        except Exception:
            pass
        screen.clear(0.0, 0.0, 0.0, 1.0)

        # 2) Sous-rectangles GL (origine bas-gauche), À L'ÉCHELLE du framebuffer réel :
        #    œil du HAUT ancré en haut, œil du BAS ancré en bas, bande de garde
        #    proportionnelle AU MILIEU. Quand fb_h == 2205 (afficheur 3D en frame
        #    packing, ou plein écran EXACT dessus via F), on retombe EXACTEMENT sur
        #    1080 / 45 / 1080 ; à toute autre taille (aperçu fenêtré, déplaçable), on
        #    obtient un over/under correct. Les yeux occupent toute la largeur.
        gap = max(1, round(fb_h * self.GAP / self.TOTAL_H))   # 45 si fb_h == 2205
        eye_h = max(1, (fb_h - gap) // 2)                     # 1080 si fb_h == 2205
        top_vp = (0, fb_h - eye_h, fb_w, eye_h)
        bottom_vp = (0, 0, fb_w, eye_h)

        # 3) Quel oeil-caméra (-1 gauche / +1 droit) en HAUT et en BAS. Convention
        #    frame packing : sous-image du HAUT = oeil GAUCHE (sauf swap).
        top_sign, bottom_sign = (-1, +1) if not self.swap_eyes else (+1, -1)

        # 4) Oeil du HAUT : on rend (off-axis) puis on compose dans son rectangle.
        #    Les commandes GL étant ordonnées, la chaîne post-FX échantillonne la
        #    texture de l'oeil AVANT que le rendu de l'oeil suivant ne l'écrase.
        hdr = self.renderer.render(eye=top_sign)
        self.postfx_top.process(hdr, screen, viewport=top_vp)

        # 5) Oeil du BAS.
        hdr = self.renderer.render(eye=bottom_sign)
        self.postfx_bottom.process(hdr, screen, viewport=bottom_vp)
