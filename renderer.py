# -*- coding: utf-8 -*-
"""
renderer.py
===========
PIPELINE DE RENDU DES PARTICULES (OpenGL 4.6 core via moderngl)
---------------------------------------------------------------
Tranche "rendu" du visualiseur audio-reactif temps reel (RTX 4090).

Role de ce module :
  * Creer les BUFFERS GL des particules (positions + couleurs) que le
    ParticleSystem remplit directement par interop CUDA -> on ne fait que les
    lier et les dessiner, jamais les ecrire cote CPU.
  * Dessiner les N particules en UN SEUL draw call (GL_POINTS), sous forme de
    petits disques gaussiens lumineux, en BLENDING ADDITIF, dans un framebuffer
    HDR flottant (RGBA16F), eventuellement MULTI-ECHANTILLONNE (MSAA) puis
    resolu vers une texture simple-echantillon que le post-traitement consomme.
  * Gerer la CAMERA perspective (fixe / rotation lente automatique / secousse &
    zoom subtils pilotes par le beat audio).

Contrat partage (respecte a la lettre) :
    class Renderer:
        def __init__(self, ctx, width, height, n_particles, msaa=4,
                     color_gradient=None): ...
        @property pos_buffer -> moderngl.Buffer      # N*vec4 (x,y,z,brightness)
        @property col_buffer -> moderngl.Buffer      # N*vec4 (r,g,b,a) HDR lineaire
        def update_camera(self, t, features): ...
        def enable_stereo(self, enabled=True, iod=None,
                          convergence_scale=None): ...   # STEREO off-axis (relief reel)
        def render(self, eye=0) -> moderngl.Texture       # eye 0=mono, -1/+1=oeil G/D off-axis
        def resize(self, width, height): ...

LAYOUT DES BUFFERS (verrou interop CUDA, cf. ParticleSystem) :
  * pos_buffer : N * vec4 float32 = (x, y, z, w=brightness), monde, contigu.
  * col_buffer : N * vec4 float32 = (r, g, b, a), couleur HDR LINEAIRE (>1 ok).
  Stride 16 octets, aucun padding. Le ParticleSystem enregistre les ids GL
  bruts (`pos_buffer.glo`, `col_buffer.glo`) pour cudaGraphicsGLRegisterBuffer
  et ecrit dedans depuis ses kernels. N est FIXE a la construction.

REMARQUE IMPORTANTE (py_compile sans GPU) :
  Les imports sont faits en tete (moderngl, numpy) mais AUCUN appel GL n'a lieu
  a l'import : tout le travail GL vit dans __init__/render/resize/update_camera,
  qui ne sont declenches qu'avec un vrai contexte fourni par l'Integrator.
"""

from __future__ import annotations

import os
import math

import numpy as np

# moderngl est importe en tete (autorise) mais n'est utilise qu'a l'execution,
# jamais a l'import : py_compile et l'import du module passent sans contexte GL.
import moderngl


# Repertoire des shaders, resolu relativement a ce fichier (robuste au CWD).
_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


def _load_shader(name: str) -> str:
    """Lit un fichier shader du dossier shaders/ et renvoie son source.

    Lecture paresseuse (appelee depuis __init__, pas a l'import) : aucun effet
    de bord au chargement du module.
    """
    path = os.path.join(_SHADER_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ===========================================================================
#  Maths camera : matrices en pur NumPy (pyrr NON requis).
#  Convention : main droite, Y vers le haut, colonnes = vecteurs de base.
#  On stocke en row-major NumPy puis on transmet a moderngl en bytes
#  ROW-MAJOR : moderngl interprete write() comme du column-major GLSL, donc on
#  envoie la TRANSPOSEE (cf. _mat_bytes) pour que mat * vec se comporte comme
#  attendu en GLSL.
# ===========================================================================
def _perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Matrice de projection perspective (style gluPerspective), main droite.

    fovy_deg : champ de vision vertical en degres.
    aspect   : largeur / hauteur du viewport.
    znear/zfar : plans de coupe (znear > 0).
    Resultat : matrice 4x4 float32 (row-major numpy).
    """
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Matrice de vue (style gluLookAt), main droite. Vecteurs np.float32 (3,)."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Axe avant (de la cible vers l'oeil), normalise. En main droite la camera
    # regarde vers -Z, donc f pointe oeil->cible et on construit la base avec -f.
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-8)
    side = np.cross(fwd, up)
    side /= (np.linalg.norm(side) + 1e-8)
    true_up = np.cross(side, fwd)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = true_up
    m[2, :3] = -fwd
    # Translation = -R * eye (projection de l'oeil sur la nouvelle base).
    m[0, 3] = -np.dot(side, eye)
    m[1, 3] = -np.dot(true_up, eye)
    m[2, 3] = np.dot(fwd, eye)
    return m


def _frustum(left, right, bottom, top, znear, zfar) -> np.ndarray:
    """Projection perspective OFF-AXIS (equiv. glFrustum), main droite.

    Autorise un frustum ASYMETRIQUE (left != -right) — requis par la stereo
    off-axis, qui decale horizontalement le frustum de chaque oeil pour annuler
    la parallaxe au plan de convergence. Resultat : 4x4 float32 row-major (cf.
    _mat_bytes pour la transmission a GLSL en column-major)."""
    rl = (right - left) or 1e-8
    tb = (top - bottom) or 1e-8
    fn = (zfar - znear) or 1e-8
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = (2.0 * znear) / rl
    m[0, 2] = (right + left) / rl
    m[1, 1] = (2.0 * znear) / tb
    m[1, 2] = (top + bottom) / tb
    m[2, 2] = -(zfar + znear) / fn
    m[2, 3] = -(2.0 * zfar * znear) / fn
    m[3, 2] = -1.0
    return m


def _mat_bytes(m: np.ndarray) -> bytes:
    """Serialise une matrice 4x4 row-major NumPy pour un uniform mat4 GLSL.

    GLSL/moderngl attendent du COLUMN-MAJOR ; nos matrices sont row-major, donc
    on transmet la transposee (ordre memoire column-major equivalent).
    """
    return m.T.astype("f4").tobytes()


class Renderer:
    """Moteur de rendu des particules + camera + framebuffer HDR.

    Cf. en-tete du module pour le contrat complet. L'Integrator possede le
    contexte/fenetre et fournit la resolution de RENDU (eventuellement
    supersamplee). Le ParticleSystem remplit pos_buffer / col_buffer par interop.
    """

    # Taille apparente FIXE des particules, en pixels ecran. Ne varie ni avec la
    # distance ni avec l'audio (cf. cahier des charges : seules position &
    # couleur changent). Exposee comme attribut pour reglage par l'Integrator.
    DEFAULT_POINT_SIZE = 6.0

    def __init__(self, ctx, width, height, n_particles, msaa=4,
                 color_gradient=None):
        """
        :param ctx:         contexte moderngl cree par l'Integrator (OpenGL 4.6).
        :param width/height:resolution de RENDU en pixels (deja supersamplee si besoin).
        :param n_particles: nombre N de particules (FIXE pour toute la session).
        :param msaa:        nombre d'echantillons MSAA (1 = desactive ; 4 ou 8 typiques).
        """
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self.n_particles = int(n_particles)
        # On borne MSAA aux valeurs raisonnables ; 1 = pas de multi-echantillonnage.
        self.msaa = max(1, int(msaa))

        # --- Parametres reglables (exposes a l'Integrator) -------------------
        self.point_size = float(self.DEFAULT_POINT_SIZE)
        # Mode camera : "fixed", "auto" (rotation lente), "beat" (auto + reaction
        # subtile au beat/amp). Defaut : "beat" (le plus vivant).
        self.camera_mode = "beat"
        # Vitesse angulaire d'orbite (rad/s) pour les modes "auto"/"beat". Reglee
        # par l'Integrator depuis CAMERA_ROTATE_SPEED ; defaut = ~1 tour / 40 s
        # (= l'ancienne constante codee en dur, valeur historique preservee). 0 =>
        # orbite immobile en azimut.
        self.rotate_speed = 2.0 * math.pi / 40.0
        # Palette HDR optionnelle fournie par l'Integrator (USER_COLOR_GRADIENT).
        # Acceptee pour respecter le contrat de construction de main.py et
        # conservee pour un usage futur (teinte globale du nuage). La couleur PAR
        # PARTICULE est actuellement produite par le kernel CUDA (HSV) cote
        # particles.py ; ce gradient n'est pas encore echantillonne par le rendu.
        self.color_gradient = color_gradient

        # --- Activation du gl_PointSize programmable -------------------------
        # Sous core profile, gl_PointSize n'est honore que si GL_PROGRAM_POINT_SIZE
        # est actif. moderngl expose ce flag ; on l'active des la construction.
        try:
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        except Exception:
            # Certains backends l'ont deja par defaut ; on n'echoue pas la-dessus.
            pass

        # --- Programme + VAO des particules ----------------------------------
        self.program = self.ctx.program(
            vertex_shader=_load_shader("particle.vert"),
            fragment_shader=_load_shader("particle.frag"),
        )

        # --- Buffers GL des particules (remplis par interop CUDA) ------------
        # N * vec4 float32 = N * 16 octets, pour pos et col. Allocation en mode
        # DYNAMIQUE (le contenu est reecrit chaque frame par les kernels CUDA).
        # On ne fournit pas de donnees initiales (reserve seul) : le contenu est
        # ecrit par CUDA avant le premier draw.
        nbytes = self.n_particles * 4 * 4  # N * vec4 * 4 octets
        self._pos_buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)
        self._col_buffer = self.ctx.buffer(reserve=nbytes, dynamic=True)

        # VAO : lie pos_buffer -> location 0 (in_pos vec4), col_buffer -> 1 (in_col).
        # Format "4f" = 4 floats par sommet (vec4), stride implicite 16 octets,
        # tightly packed -> exactement le layout verrouille avec ParticleSystem.
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self._pos_buffer, "4f", "in_pos"),
                (self._col_buffer, "4f", "in_col"),
            ],
        )

        # --- Framebuffers HDR (MSAA + resolution) ----------------------------
        # Crees ici, recrees a chaque resize().
        self._fbo_msaa = None        # FBO multi-echantillonne (cible du rendu)
        self._msaa_color = None      # renderbuffer couleur multi-echantillon
        self._msaa_depth = None      # renderbuffer depth multi-echantillon
        self._resolve_tex = None     # texture RGBA16F simple-echantillon (sortie)
        self._fbo_resolve = None     # FBO enveloppant resolve_tex (pour le blit)
        self._build_framebuffers()

        # --- Matrices camera (initialisees, mises a jour par update_camera) --
        # Parametres de projection PARTAGES par le rendu mono ET stereo.
        self._cam_fovy = 45.0
        self._cam_near = 0.05
        self._cam_far = 100.0
        self._view = np.eye(4, dtype=np.float32)
        self._proj = _perspective(self._cam_fovy, self._aspect(),
                                  self._cam_near, self._cam_far)
        self._view_proj = self._proj @ self._view

        # Distance de base de la camera : cadre le papillon de Lorenz (demi-extent
        # ~2.5..3 en unites monde) depuis l'EXTERIEUR, pour voir sa structure 3D.
        self._cam_base_dist = 8.0
        # Demi-cote de la BOITE de particules (= CLOUD_RADIUS), pose par l'Integrator
        # apres construction (cf. renderer.box_radius dans main.py). Sert au mode camera
        # "tunnel" pour placer l'oeil sur l'axe pres du bord proche.
        self.box_radius = 3.5
        # TUNNEL : fonction (z) -> (x, y) de l'AXE COURBE du tube, cablee par
        # l'Integrator (renderer.tunnel_axis_fn = particles.tunnel_axis_at). La
        # camera SUIT le serpent et s'INCLINE dans les virages. None = axe droit.
        self.tunnel_axis_fn = None
        self._tun_roll = 0.0       # roulis de virage lisse (banking)
        # Etat lisse du zoom reactif (evite les a-coups image a image).
        self._zoom_env = 0.0
        self._shake_env = 0.0

        # --- STEREOSCOPIE (off-axis, axes paralleles) — desactivee par defaut -
        # Rendu MONO normal sauf si enable_stereo() est appele (par l'Integrator
        # en mode 3D). IOD = ecart inter-oculaire en unites MONDE ; convergence =
        # distance oeil->cible * stereo_convergence_scale (suit donc le zoom). On
        # memorise l'etat camera (rig central) a chaque update_camera pour deriver
        # les deux frustums sans refaire le calcul d'orbite.
        self.stereo_enabled = False
        self.stereo_iod = 0.22
        self.stereo_convergence_scale = 1.0
        self._cam_eye = np.array([0.0, 0.0, self._cam_base_dist], dtype=np.float32)
        self._cam_target = np.zeros(3, dtype=np.float32)
        self._cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # ====================================================================== #
    #  Construction / reconstruction des framebuffers                        #
    # ====================================================================== #
    def _build_framebuffers(self):
        """Cree (ou recree) le FBO HDR de rendu et la texture de resolution.

        Schema :
          [particules] --(draw additif)--> FBO MSAA (RGBA16F multi-echantillon)
                        --(blit/resolve)--> texture RGBA16F simple-echantillon
                                            --> consommee par PostFX.

        Si msaa == 1, le "FBO MSAA" est en realite simple-echantillon et le
        rendu peut viser directement la texture de resolution ; on garde
        toutefois deux FBO pour un chemin de code uniforme (le blit est alors
        une simple copie 1:1, peu couteuse).
        """
        w, h = self.width, self.height

        # Liberation des anciennes ressources GL (resize).
        self._release_framebuffers()

        # --- Texture de sortie : RGBA16F simple-echantillon ------------------
        # dtype "f2" = float16 par composante -> RGBA16F. C'est l'HDR LINEAIRE
        # PRE-TONEMAP que PostFX echantillonne (filtrage lineaire, clamp bords).
        self._resolve_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self._resolve_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._resolve_tex.repeat_x = False  # clamp to edge
        self._resolve_tex.repeat_y = False
        self._fbo_resolve = self.ctx.framebuffer(color_attachments=[self._resolve_tex])

        if self.msaa > 1:
            # --- FBO multi-echantillonne : cible reelle du rendu particules ---
            # Renderbuffer couleur RGBA16F multi-echantillon. On joint un depth
            # multi-echantillon de meme nombre de samples (exige par le FBO MSAA
            # complet) meme si le test de profondeur est desactive : c'est juste
            # une piece d'attache valide, peu couteuse.
            self._msaa_color = self.ctx.renderbuffer(
                (w, h), 4, samples=self.msaa, dtype="f2"
            )
            self._msaa_depth = self.ctx.depth_renderbuffer((w, h), samples=self.msaa)
            self._fbo_msaa = self.ctx.framebuffer(
                color_attachments=[self._msaa_color],
                depth_attachment=self._msaa_depth,
            )
        else:
            # Pas de MSAA : on rend directement dans le FBO de resolution.
            self._fbo_msaa = self._fbo_resolve

    def _release_framebuffers(self):
        """Libere proprement les FBO/textures/renderbuffers existants."""
        # Ne pas liberer _fbo_msaa s'il est un alias de _fbo_resolve (msaa==1).
        for attr in ("_msaa_color", "_msaa_depth"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
                setattr(self, attr, None)

        fbo_msaa = getattr(self, "_fbo_msaa", None)
        fbo_resolve = getattr(self, "_fbo_resolve", None)
        if fbo_msaa is not None and fbo_msaa is not fbo_resolve:
            fbo_msaa.release()
        self._fbo_msaa = None

        if fbo_resolve is not None:
            fbo_resolve.release()
        self._fbo_resolve = None

        tex = getattr(self, "_resolve_tex", None)
        if tex is not None:
            tex.release()
        self._resolve_tex = None

    # ====================================================================== #
    #  Buffers exposes (interop CUDA)                                        #
    # ====================================================================== #
    @property
    def pos_buffer(self) -> "moderngl.Buffer":
        """Buffer GL des positions : N * vec4 (x, y, z, brightness), dynamique.

        Le ParticleSystem enregistre `self.pos_buffer.glo` pour l'interop CUDA et
        y ecrit chaque frame. Le renderer ne fait que le lier comme attribut.
        """
        return self._pos_buffer

    @property
    def col_buffer(self) -> "moderngl.Buffer":
        """Buffer GL des couleurs : N * vec4 (r, g, b, a) HDR lineaire, dynamique.

        Idem : enregistre via `self.col_buffer.glo` pour l'interop CUDA.
        """
        return self._col_buffer

    # ====================================================================== #
    #  Camera                                                                #
    # ====================================================================== #
    def _aspect(self) -> float:
        """Ratio largeur/hauteur courant (garde-fou contre la division par 0)."""
        return self.width / max(1, self.height)

    # Correspondance noms Integrator (main.py) -> noms internes du renderer.
    # main.py expose "fixed"/"auto_rotate"/"beat_reactive" ; en interne on
    # raisonne en "fixed"/"auto"/"beat".
    _CAMERA_MODE_ALIASES = {
        "fixed": "fixed",
        "auto_rotate": "auto",
        "auto": "auto",
        "beat_reactive": "beat",
        "beat": "beat",
        "tunnel": "tunnel",       # vol DANS le tunnel (regard le long de l'axe Z)
    }

    def set_camera_mode(self, mode):
        """Change le mode camera a chaud. Accepte les noms de l'Integrator
        ("fixed"/"auto_rotate"/"beat_reactive") comme les noms internes
        ("fixed"/"auto"/"beat") ; un nom inconnu retombe sur "beat"."""
        self.camera_mode = self._CAMERA_MODE_ALIASES.get(str(mode).lower(), "beat")

    def enable_stereo(self, enabled=True, iod=None, convergence_scale=None):
        """Active/regle le rendu STEREO off-axis (relief reel). Sans effet sur le
        chemin mono tant que l'Integrator n'appelle pas render(eye=-1/+1).

        :param enabled:           True -> l'Integrator dessinera les deux yeux.
        :param iod:               ecart inter-oculaire (unites MONDE) ; + grand = + de relief.
        :param convergence_scale: multiplie la distance oeil->cible pour placer le
                                  plan a PARALLAXE NULLE ( <1 = le nuage 'sort' de
                                  l'ecran, >1 = il s'enfonce derriere)."""
        self.stereo_enabled = bool(enabled)
        if iod is not None:
            self.stereo_iod = max(0.0, float(iod))
        if convergence_scale is not None:
            self.stereo_convergence_scale = max(1e-3, float(convergence_scale))

    def update_camera(self, t, features):
        """Met a jour les matrices vue/projection a partir du temps et de l'audio.

        :param t:        temps ecoule en secondes (float).
        :param features: AudioFeatures (bass/lowmid/mid/high/amp/beat/spectrum).
                         Peut etre None (mode silencieux / pas encore d'audio).

        Modes (self.camera_mode) :
          * "fixed" : camera immobile face au nuage.
          * "auto"  : rotation lente automatique en orbite.
          * "beat"  : rotation lente + ZOOM doux pilote par amp + SECOUSSE breve
                      sur le beat. Tout reste SUBTIL (le cahier des charges insiste
                      sur des variations discretes : la taille des particules, elle,
                      ne bouge jamais ; seule la camera respire).
        """
        # Recuperation defensive des features (None -> valeurs nulles).
        amp = float(getattr(features, "amplitude", 0.0)) if features is not None else 0.0
        beat = float(getattr(features, "beat", 0.0)) if features is not None else 0.0
        # Modele de tempo/phase (groove) : lus defensivement -> 0 si absent ou pas
        # encore verrouille. Ils transforment la reaction "tremblement par kick" en
        # HOULE calee sur la mesure (cf. audio_engine._apply_tempo).
        pulse_phase = float(getattr(features, "pulse_phase", 0.0)) if features is not None else 0.0
        bar_phase = float(getattr(features, "bar_phase", 0.0)) if features is not None else 0.0
        conf = float(getattr(features, "groove_conf", 0.0)) if features is not None else 0.0
        build = float(getattr(features, "build", 0.0)) if features is not None else 0.0
        drop = float(getattr(features, "drop", 0.0)) if features is not None else 0.0
        two_pi = 2.0 * math.pi
        houle = math.sin(two_pi * bar_phase)                    # -1..1 lent (4 temps)
        pulse_peak = 0.5 + 0.5 * math.cos(two_pi * pulse_phase)  # 1 au temps, 0 au milieu

        # --- Angle d'orbite ---
        if self.camera_mode == "fixed":
            angle = 0.0
            elev = 0.18  # leger surplomb fixe
        else:
            # Rotation lente, VITESSE ANGULAIRE REGLABLE (rad/s) : self.rotate_speed
            # branche depuis CAMERA_ROTATE_SPEED (defaut ~1 tour / 40 s). 0 => azimut
            # fige (orbite immobile, mais l'elevation respire encore).
            angle = t * self.rotate_speed
            # Oscillation verticale tres douce de l'elevation.
            elev = 0.18 + 0.10 * math.sin(t * 0.13)
            # HOULE verticale calee sur la MESURE (mode beat, portee par la confiance) :
            # la camera "respire" lentement sur 4 temps au lieu de subir chaque kick.
            if self.camera_mode == "beat":
                elev += conf * 0.07 * houle

        # --- Zoom (amp) + HOCHEMENT de groove + secousse RESIDUELLE -----------
        # Lissage exponentiel pour eviter tout saut brutal entre deux frames.
        groove_zoom = 0.0
        if self.camera_mode == "beat":
            # Zoom doux pilote par l'amplitude (comme avant).
            self._zoom_env += (0.25 * min(amp, 1.5) - self._zoom_env) * 0.08
            # HOCHEMENT : la camera "pousse" doucement sur le temps fort. Pilote par
            # la PHASE CONTINUE (pas un a-coup), d'autant plus marque que c'est cale.
            groove_zoom = conf * 0.45 * pulse_peak
            # Secousse : S'EFFACE a mesure que le groove se verrouille -> on remplace
            # le tremblement-par-kick par le hochement lisse ci-dessus.
            self._shake_env = max(self._shake_env * 0.85, beat * (1.0 - 0.8 * conf), 0.8 * drop)
        else:
            self._zoom_env += (0.0 - self._zoom_env) * 0.08
            self._shake_env *= 0.85

        # Distance camera : base - zoom - hochement (positif => on se rapproche).
        # PHRASE : le build RECULE la camera (on se ramasse), le drop PLONGE dedans.
        dist = (self._cam_base_dist - self._zoom_env - groove_zoom
                + 1.2 * build - 1.6 * drop)

        # Position de l'oeil en orbite autour de l'origine.
        cx = math.cos(angle) * dist
        cz = math.sin(angle) * dist
        cy = elev * dist
        eye = np.array([cx, cy, cz], dtype=np.float32)

        # Secousse : petit decalage pseudo-aleatoire mais deterministe (fonction
        # du temps), d'amplitude proportionnelle au beat. Reste minuscule.
        if self._shake_env > 1e-3:
            sh = 0.04 * self._shake_env
            eye[0] += sh * math.sin(t * 53.0)
            eye[1] += sh * math.sin(t * 61.0 + 1.7)
            eye[2] += sh * math.sin(t * 47.0 + 3.1)

        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # MODE TUNNEL : on REMPLACE l'orbite externe par un vol DANS le tube, regard le
        # long de l'axe (-Z), FOV elargi pour accentuer le point de fuite — et DILATE
        # au drop (le saut en lumiere etire la perspective). On ecrit le FOV dans
        # self._cam_fovy (et pas une variable locale) pour que les frustums STEREO
        # off-axis (_eye_view_proj) voient le MEME champ : sinon le mode 3D rendait
        # le tunnel avec le FOV standard de 45°.
        if self.camera_mode == "tunnel":
            eye, target, up = self._tunnel_camera(t, two_pi, bar_phase, conf,
                                                  build, drop, pulse_peak)
            self._cam_fovy = 78.0 + 12.0 * drop
        else:
            self._cam_fovy = 45.0

        # Memorise l'etat camera (rig central) pour deriver les frustums STEREO.
        self._cam_eye = eye
        self._cam_target = target
        self._cam_up = up
        self._view = _look_at(eye, target, up)
        self._proj = _perspective(self._cam_fovy, self._aspect(),
                                  self._cam_near, self._cam_far)
        # view_proj = projection * vue (applique a une position monde homogene).
        self._view_proj = self._proj @ self._view

    def _tunnel_camera(self, t, two_pi, bar_phase, conf, build, drop, pulse_peak):
        """Camera DANS le tunnel : l'oeil VOLE SUR L'AXE COURBE du tube (le serpent
        conduit par le Lorenz cache, via tunnel_axis_fn) pres du bord proche
        (z ~ +L), regard le long de l'axe vers -Z, VERS LE VIRAGE a venir (point de
        fuite au fond, enroule a l'infini). La camera S'INCLINE dans les virages
        (banking lisse, comme un cockpit), POMPE en avant sur la pulsation predite
        (lunge porte par le groove) ; le DROP plonge en avant (saut en lumiere), le
        BUILD recule un peu (on se ramasse). Renvoie (eye, target, up)."""
        L = float(getattr(self, "box_radius", 3.5))
        look = 2.6                                        # distance de visee (plan de convergence 3D)
        # Avance/recul : drop -> plonge ; build -> recule ; pulsation -> pompage.
        ez = L - 1.3 * drop + 0.6 * build - 0.30 * conf * pulse_peak
        # L'oeil sur l'axe COURBE (+ une respiration de main levee minuscule) ; la
        # cible sur l'axe PLUS LOIN dans le tube -> on regarde DANS le virage.
        axis = self.tunnel_axis_fn
        if callable(axis):
            ax_e, ay_e = axis(ez)
            ax_t, ay_t = axis(ez - look)
        else:
            ax_e = ay_e = ax_t = ay_t = 0.0
        sway = 0.025 * L
        ex = ax_e + sway * math.sin(t * 0.37)
        ey = ay_e + sway * math.sin(t * 0.29 + 1.1)
        eye = np.array([ex, ey, ez], dtype=np.float32)
        target = np.array([ax_t, ay_t, ez - look], dtype=np.float32)
        # BANKING : roulis proportionnel au deport lateral du virage a venir
        # (incline VERS l'interieur du virage), lisse pour rester organique ;
        # + vrille lente continue + houle calee sur la mesure (portee par le groove).
        bank_target = max(-0.55, min(0.55, 0.9 * (ax_t - ax_e)))
        self._tun_roll += (bank_target - self._tun_roll) * 0.06
        roll = t * 0.06 + 0.25 * conf * math.sin(two_pi * bar_phase) + self._tun_roll
        up = np.array([math.sin(roll), math.cos(roll), 0.0], dtype=np.float32)
        return eye, target, up

    def _eye_view_proj(self, eye_sign):
        """view_proj OFF-AXIS d'un oeil. eye_sign = -1 (gauche) / +1 (droit).

        STEREO geometriquement correcte (methode de Bourke) : les deux cameras
        gardent des axes PARALLELES (pas de toe-in, qui introduirait une parallaxe
        VERTICALE inconfortable) ; tout le relief vient d'un frustum ASYMETRIQUE
        decale horizontalement pour annuler la parallaxe au plan de CONVERGENCE.
        L'oeil est decale lateralement de +/- IOD/2 le long du vecteur 'droite'
        camera. Convergence = distance oeil->cible * stereo_convergence_scale.
        """
        eye = self._cam_eye
        target = self._cam_target
        up = self._cam_up

        fwd = target - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        cam_right = np.cross(fwd, up)
        cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-8)

        iod = float(self.stereo_iod)
        convergence = max(1e-3, float(np.linalg.norm(target - eye))
                          * self.stereo_convergence_scale)

        # Oeil decale lateralement ; regard PARALLELE (eye_pos -> eye_pos + fwd).
        eye_pos = eye + cam_right * (eye_sign * 0.5 * iod)
        view = _look_at(eye_pos, eye_pos + fwd, up)

        # Frustum asymetrique : demi-hauteur au near = near*tan(fovy/2) ; le
        # decalage horizontal compense le decalage d'oeil (parallaxe nulle a la
        # convergence). Oeil gauche (-1) -> frustum vers +x ; droit (+1) -> -x.
        near = self._cam_near
        far = self._cam_far
        wd2 = near * math.tan(math.radians(self._cam_fovy) * 0.5)
        aspect = self._aspect()
        ndfl = near / convergence
        shift = -eye_sign * 0.5 * iod * ndfl
        left = -aspect * wd2 + shift
        right = aspect * wd2 + shift
        proj = _frustum(left, right, -wd2, wd2, near, far)
        return proj @ view

    # ====================================================================== #
    #  Rendu                                                                 #
    # ====================================================================== #
    def render(self, eye=0) -> "moderngl.Texture":
        """Dessine les N particules dans le FBO HDR, resout le MSAA, et renvoie
        la texture couleur RGBA16F simple-echantillon (consommee par PostFX).

        :param eye: 0 = MONO (frustum symetrique) ; -1 / +1 = oeil GAUCHE / DROIT
                    en projection OFF-AXIS (stereo). Les buffers de particules sont
                    IDENTIQUES pour les deux yeux (une seule simulation) : seule la
                    matrice camera change -> on dessine le meme nuage deux fois.

        Etat GL pose ici (et restaure en sortie pour ne rien casser en aval) :
          * Viewport = resolution de rendu.
          * Effacement : NOIR profond opaque (fond sombre, alpha=1).
          * Test de profondeur DESACTIVE et ecriture depth coupee : en additif,
            l'ordre n'importe pas et on veut que toutes les particules cumulent
            leur lumiere sans s'occulter (nuage volumetrique translucide).
          * Blending ADDITIF : glBlendFunc(GL_ONE, GL_ONE) -> somme des
            contributions HDR (les zones denses saturent en blanc chaud).
        """
        ctx = self.ctx

        # --- Transmission de la matrice camera et de la taille de point ------
        # (uniforms du programme particule). MONO -> view_proj symetrique ;
        # STEREO -> view_proj off-axis de l'oeil. Serialise en column-major.
        view_proj = self._view_proj if eye == 0 else self._eye_view_proj(eye)
        if "u_view_proj" in self.program:
            self.program["u_view_proj"].write(_mat_bytes(view_proj))
        if "u_point_size" in self.program:
            self.program["u_point_size"].value = self.point_size

        # --- Cible : FBO HDR (MSAA si actif) ---------------------------------
        self._fbo_msaa.use()
        ctx.viewport = (0, 0, self.width, self.height)

        # Fond NOIR PROFOND, opaque (alpha=1 pour la sortie PostFX).
        self._fbo_msaa.clear(0.0, 0.0, 0.0, 1.0)

        # --- Etat GL pour l'additif volumetrique -----------------------------
        # Depth test off : aucune occlusion entre particules (cumul total).
        ctx.disable(moderngl.DEPTH_TEST)
        # Blending additif : on enregistre puis on force (GL_ONE, GL_ONE).
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        # --- UN SEUL draw call pour les N particules (GL_POINTS) -------------
        self.vao.render(mode=moderngl.POINTS, vertices=self.n_particles)

        # --- Resolution MSAA -> texture simple-echantillon -------------------
        if self.msaa > 1:
            # Blit du FBO multi-echantillon vers le FBO de resolution : OpenGL
            # moyenne les samples -> anti-aliasing. moderngl expose copy_framebuffer.
            ctx.copy_framebuffer(self._fbo_resolve, self._fbo_msaa)
        # (si msaa == 1, _fbo_msaa EST _fbo_resolve : rien a resoudre.)

        # --- Restauration d'un etat GL neutre pour les passes suivantes ------
        # On laisse le blending tel quel pourrait surprendre PostFX ; on remet
        # un etat sobre (blend desactive, depth desactive reste sans danger pour
        # des passes plein-ecran). PostFX repose de toute facon son propre etat.
        ctx.disable(moderngl.BLEND)

        return self._resolve_tex

    # ====================================================================== #
    #  Redimensionnement                                                     #
    # ====================================================================== #
    def resize(self, width, height):
        """Reconstruit les FBO/texture a la nouvelle resolution de rendu et
        met a jour l'aspect de la camera. Les buffers particules (pos/col) ne
        sont PAS recrees : N est fixe et leur taille ne depend pas de l'ecran.
        """
        w, h = int(width), int(height)
        if w <= 0 or h <= 0:
            return  # fenetre minimisee : on ignore
        if w == self.width and h == self.height:
            return  # rien a faire
        self.width = w
        self.height = h
        self._build_framebuffers()
        # Met a jour la matrice de projection (aspect) immediatement ; la vue
        # sera rafraichie au prochain update_camera.
        self._proj = _perspective(self._cam_fovy, self._aspect(),
                                  self._cam_near, self._cam_far)
        self._view_proj = self._proj @ self._view

    # ====================================================================== #
    #  Nettoyage                                                             #
    # ====================================================================== #
    def release(self):
        """Libere toutes les ressources GL detenues par le renderer.

        L'Integrator peut l'appeler a la fermeture. Le contexte lui-meme
        appartient a l'Integrator et n'est PAS detruit ici.
        """
        self._release_framebuffers()
        for attr in ("vao", "_pos_buffer", "_col_buffer", "program"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, attr, None)
