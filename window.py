"""
window.py
=========
Création de la fenêtre GLFW et du contexte OpenGL 4.6 core, puis
récupération d'un contexte moderngl par-dessus.

Responsabilités de ce module (tranche « fenêtre / contexte ») :
  - Initialiser GLFW et poser les hints OpenGL (4.6 core, MSAA via SAMPLES).
  - Créer une fenêtre plein écran (moniteur primaire) OU fenêtrée
    redimensionnable selon `config.fullscreen`.
  - Régler la synchronisation verticale (vsync) : swap_interval(1) par défaut,
    swap_interval(0) pour viser le maximum de FPS (mode « adaptatif »).
  - Installer les callbacks clavier (ESC -> fermeture) et de redimensionnement
    (met à jour la taille courante + lève un drapeau `resized`).
  - Exposer le contexte moderngl (`ctx`), la taille courante, la résolution
    native du moniteur (utile pour le 4K/8K) et l'état (doit_fermer / resized).

NOTE IMPORTANTE sur le MSAA :
  Le hint SAMPLES active le multi-échantillonnage sur le *framebuffer par
  défaut* (l'écran). Or, dans notre pipeline, les particules sont rendues
  dans un framebuffer HDR offscreen (RGBA16F, géré par renderer.py), puis
  ce FBO est ré-affiché à l'écran (blit / post-traitement). Le MSAA du
  framebuffer par défaut ne s'applique donc PAS au rendu des particules
  lui-même ; il n'aide qu'à l'affichage final / aux éléments dessinés
  directement à l'écran. On le laisse configurable car c'est sans coût si
  l'étape de présentation finale en profite, mais l'anti-aliasing « utile »
  pour les particules vient du supersampling (SUPERSAMPLE dans renderer.py)
  et de la gaussienne douce du fragment shader.
"""

import glfw
import moderngl


class Window:
    """Fenêtre GLFW + contexte OpenGL/moderngl.

    Paramètres attendus dans `config` (objet ou namespace) :
      - config.width, config.height : taille fenêtrée souhaitée (px).
      - config.fullscreen : bool, plein écran sur le moniteur primaire.
      - config.msaa : int, nombre d'échantillons MSAA (0/1 = désactivé).
      - config.vsync : bool, True -> swap_interval(1), False -> 0 (FPS max).
      - config.title : str (optionnel), titre de la fenêtre.
    """

    def __init__(self, config):
        self.config = config

        # --- État exposé au reste de l'application -----------------------
        # Taille courante du framebuffer (en pixels réels, DPI compris).
        self._fb_size = (int(getattr(config, "width", 1280)),
                         int(getattr(config, "height", 720)))
        # Drapeau levé par le callback de resize, consommé via was_resized().
        self._resized = False
        # Résolution native du moniteur primaire (pour 4K/8K).
        self.monitor_size = self._fb_size
        # Callback clavier additionnel optionnel (toggles B/M/C de main.py).
        # Signature attendue : fn(key, action). Installé via set_extra_key_callback.
        self._extra_key_callback = None
        # Plein écran BORDERLESS (sans bordure, couvrant le moniteur ; bascule à
        # chaud via toggle_fullscreen). On évite l'EXCLUSIF car, sous Windows,
        # une appli OpenGL exclusive sans surface HDR force l'écran en SDR
        # (« HDR désactivé ») ; le borderless laisse le compositeur DWM actif et
        # conserve donc le mode HDR de l'écran.
        self._is_fullscreen = bool(getattr(config, "fullscreen", False))
        # Géométrie fenêtrée mémorisée pour revenir du plein écran.
        self._windowed_pos = (80, 80)
        self._windowed_size = (int(getattr(config, "width", 1280)),
                               int(getattr(config, "height", 720)))

        # --- Initialisation GLFW -----------------------------------------
        if not glfw.init():
            raise RuntimeError("Échec de l'initialisation de GLFW.")

        # Hints du contexte : OpenGL 4.6 core profile, forward compatible.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        # MSAA sur le framebuffer par défaut (voir note d'en-tête).
        samples = int(getattr(config, "msaa", 0) or 0)
        glfw.window_hint(glfw.SAMPLES, samples)

        # On veut un framebuffer par défaut classique sRGB désactivé : le HDR
        # et le tonemapping sont gérés en aval, on évite une correction gamma
        # surprise sur le framebuffer par défaut.
        glfw.window_hint(glfw.SRGB_CAPABLE, glfw.FALSE)

        title = getattr(config, "title", "Sythm - Visualiseur audio GPU")

        # --- Récupération du moniteur primaire et de sa résolution native -
        primary = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(primary)
        if video_mode is not None:
            self.monitor_size = (video_mode.size.width, video_mode.size.height)

        # --- Création de la fenêtre --------------------------------------
        if getattr(config, "fullscreen", False):
            # Plein écran BORDERLESS (et non exclusif), DÉBORDANT de 1px hors écran
            # sur chaque bord (taille +2, position -1) : la fenêtre n'est pas un
            # calque exact du moniteur -> Windows ne la promeut pas en
            # independent-flip (qui ferait retomber l'écran en SDR) -> DWM compose
            # -> le HDR de l'écran est conservé. monitor=None -> pas d'exclusif.
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            w, h = self.monitor_size
            self.handle = glfw.create_window(w + 2, h + 2, title, None, None)
            if self.handle:
                try:
                    mx, my = glfw.get_monitor_pos(primary)
                    glfw.set_window_pos(self.handle, mx - 1, my - 1)
                except Exception:
                    pass
        else:
            # Fenêtré et redimensionnable (RESIZABLE par défaut sous GLFW).
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            w, h = self._fb_size
            self.handle = glfw.create_window(w, h, title, None, None)

        if not self.handle:
            glfw.terminate()
            raise RuntimeError("Échec de la création de la fenêtre GLFW.")

        # On rend le contexte courant AVANT de créer le contexte moderngl.
        glfw.make_context_current(self.handle)

        # Synchronisation verticale :
        #   1 -> vsync activé (limité à la fréquence du moniteur, pas de tearing)
        #   0 -> désactivé (FPS maximum, « adaptatif » / sans limite)
        glfw.swap_interval(1 if getattr(config, "vsync", True) else 0)

        # Taille réelle du framebuffer (peut différer de la taille fenêtre en
        # cas de mise à l'échelle DPI). C'est cette taille qu'utilise le rendu.
        fb_w, fb_h = glfw.get_framebuffer_size(self.handle)
        self._fb_size = (fb_w, fb_h)

        # --- Contexte moderngl par-dessus le contexte GLFW courant -------
        # require=460 -> OpenGL 4.6 ; moderngl récupère le contexte existant.
        self.ctx = moderngl.create_context(require=460)
        # Activation du point sprite programmable : nécessaire pour pouvoir
        # piloter gl_PointSize depuis le vertex shader (cf. renderer.py).
        # Sous core profile 4.6 c'est implicite, mais on garde l'intention claire.

        # --- Installation des callbacks ----------------------------------
        glfw.set_key_callback(self.handle, self._on_key)
        glfw.set_framebuffer_size_callback(self.handle, self._on_framebuffer_size)

    # ------------------------------------------------------------------ #
    #  Callbacks GLFW                                                     #
    # ------------------------------------------------------------------ #
    def _on_key(self, window, key, scancode, action, mods):
        """ESC -> demande de fermeture de la fenêtre.

        Toute autre touche est relayée au callback additionnel éventuel
        (enregistré par main.py via set_extra_key_callback) pour les bascules
        d'effets/caméra (B/M/C...). On garde la gestion d'ESC ici car c'est une
        responsabilité « fenêtre » (fermeture propre), indépendante du reste.
        """
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            return
        # Relais vers le callback applicatif (toggles). On l'isole dans un
        # try pour qu'une exception côté appli ne casse jamais la boucle GLFW.
        if self._extra_key_callback is not None:
            try:
                self._extra_key_callback(key, action)
            except Exception as exc:  # pragma: no cover - robustesse runtime
                print(f"[window] callback clavier applicatif a levé : {exc}")

    def _on_framebuffer_size(self, window, width, height):
        """Redimensionnement : on mémorise la nouvelle taille du framebuffer
        et on lève le drapeau `_resized` pour que renderer/camera réagissent.
        On ignore les tailles nulles (fenêtre minimisée)."""
        if width > 0 and height > 0:
            self._fb_size = (width, height)
            self._resized = True

    # ------------------------------------------------------------------ #
    #  Extension clavier applicative                                      #
    # ------------------------------------------------------------------ #
    def set_extra_key_callback(self, callback):
        """Enregistre un callback clavier additionnel `fn(key, action)`.

        Utilisé par main.py pour les bascules (B=bloom, M=motion blur,
        C=mode caméra). ESC reste géré en interne par la fenêtre. Passer None
        pour désinstaller.
        """
        self._extra_key_callback = callback

    # ------------------------------------------------------------------ #
    #  Plein écran BORDERLESS (bascule à chaud)                          #
    # ------------------------------------------------------------------ #
    def toggle_fullscreen(self):
        """Bascule entre fenêtré et PLEIN ÉCRAN BORDERLESS (fenêtre sans bordure
        couvrant le moniteur primaire à sa résolution native). On n'utilise PAS
        le plein écran exclusif : sous Windows, une appli OpenGL exclusive sans
        surface HDR force l'écran en SDR (« HDR désactivé »). Le borderless
        laisse le compositeur DWM actif -> le mode HDR de l'écran est conservé.
        Mémorise la géométrie fenêtrée pour y revenir. Le changement de
        framebuffer lève `resized` -> renderer/postfx se réajustent ensuite."""
        if self.handle is None:
            return
        if self._is_fullscreen:
            # Retour fenêtré : on redécore et on restaure la géométrie mémorisée.
            glfw.set_window_attrib(self.handle, glfw.DECORATED, glfw.TRUE)
            x, y = self._windowed_pos
            w, h = self._windowed_size
            glfw.set_window_monitor(self.handle, None, x, y, w, h, 0)
            self._is_fullscreen = False
        else:
            # Mémorise la fenêtre courante, puis passe en borderless plein moniteur.
            try:
                self._windowed_pos = glfw.get_window_pos(self.handle)
                self._windowed_size = glfw.get_window_size(self.handle)
            except Exception:
                pass
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            try:
                mx, my = glfw.get_monitor_pos(monitor)
            except Exception:
                mx, my = 0, 0
            # DECORATED=FALSE + fenêtre (monitor=None) = borderless. MAIS on la
            # fait DÉBORDER de 1px hors écran sur chaque bord (pos -1, taille +2) :
            # ainsi elle n'est PAS un calque EXACT du moniteur, donc Windows ne la
            # promeut pas en "fullscreen optimization"/independent-flip (qui ferait
            # retomber l'écran en SDR comme l'exclusif). DWM continue de composer
            # -> le mode HDR de l'écran est CONSERVÉ. Le débordement (+2 par
            # dimension) garde aussi des tailles PAIRES (requis par l'encodage 4:2:0).
            glfw.set_window_attrib(self.handle, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(self.handle, None, mx - 1, my - 1,
                                    mode.size.width + 2, mode.size.height + 2, 0)
            self._is_fullscreen = True
        # Le changement peut réinitialiser le swap interval : on le réapplique.
        glfw.swap_interval(1 if getattr(self.config, "vsync", True) else 0)
        # Met à jour la taille du framebuffer + lève le drapeau de resize.
        fb_w, fb_h = glfw.get_framebuffer_size(self.handle)
        self._fb_size = (fb_w, fb_h)
        self._resized = True

    @property
    def is_fullscreen(self):
        """True si la fenêtre est actuellement en plein écran borderless."""
        return self._is_fullscreen

    # ------------------------------------------------------------------ #
    #  Boucle / présentation                                              #
    # ------------------------------------------------------------------ #
    def poll_events(self):
        """Traite la file d'événements GLFW (clavier, resize, fermeture)."""
        glfw.poll_events()

    def swap_buffers(self):
        """Présente le framebuffer (échange avant/arrière)."""
        glfw.swap_buffers(self.handle)

    # ------------------------------------------------------------------ #
    #  État exposé                                                        #
    # ------------------------------------------------------------------ #
    @property
    def size(self):
        """Taille courante du framebuffer en pixels (width, height)."""
        return self._fb_size

    def doit_fermer(self):
        """True si l'utilisateur a demandé la fermeture (ESC, croix...)."""
        return glfw.window_should_close(self.handle)

    @property
    def resized(self):
        """True si un redimensionnement a eu lieu depuis la dernière lecture.
        Lecture « consommatrice » : le drapeau retombe à False après lecture,
        ce qui permet à la boucle principale de n'appeler resize() qu'une fois."""
        r = self._resized
        self._resized = False
        return r

    # ------------------------------------------------------------------ #
    #  Nettoyage                                                          #
    # ------------------------------------------------------------------ #
    def close(self):
        """Libère la fenêtre et termine GLFW proprement."""
        if self.handle is not None:
            glfw.destroy_window(self.handle)
            self.handle = None
        glfw.terminate()
