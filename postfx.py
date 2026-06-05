"""
postfx.py
=========
Post-traitement cinematographique plein ecran en `moderngl` pour le
visualiseur audio-reactif (cible RTX 4090, Python + GLSL).

ROLE DE CE MODULE (tranche « post-FX ») :
  Prend en entree une texture HDR RGBA16F (lineaire, AVANT tonemap, a la
  resolution de rendu, produite par renderer.py) et produit l'image SDR finale
  affichee a l'ecran, en appliquant dans cet ordre :

    1. BLOOM  : bright-pass (extraction des hautes lumieres avec soft knee),
                puis flou gaussien SEPARABLE (passes horizontale + verticale,
                FBO ping-pong) sur une petite PYRAMIDE DE MIPS pour un halo
                large et doux, enfin recomposition ADDITIVE avec la scene HDR.
                Seuil et intensite parametrables.

    2. MOTION BLUR : accumulation par BUFFER D'HISTORIQUE (melange de la frame
                courante avec la precedente via un alpha -> trainee temporelle).
                Force parametrable. (L'alternative « velocity buffer » est
                documentee dans motionblur.frag.) Le buffer d'historique est
                conserve EN INTERNE ici (decouplage avec renderer.py).

    3. TONE MAPPING HDR : ACES filmic (+ Uncharted2 en variante) avec controle
                d'EXPOSITION, conversion HDR lineaire -> SDR, puis encodage sRGB.
                Pense pour rester eclatant sur fond noir profond.

    4. SUPERSAMPLING / DOWNSCALE : si la resolution de rendu depasse celle de
                l'ecran, on redimensionne avec un filtre de qualite (Lanczos-2,
                a defaut bilineaire). C'est aussi le point d'insertion d'un
                upscaler neuronal (DLSS / TensorRT / ONNX) ou d'une passe TAA
                (voir lanczos_upscale.frag).

    5. BONUS (optionnel, peu couteux) : vignette douce (et emplacement prevu
                pour un depth-of-field).

CONTRAT PARTAGE (renderer.py et l'integrateur en dependent) :
    class PostProcessor:
        def __init__(self, ctx, render_width, render_height,
                     screen_width, screen_height,
                     enable_bloom=True, enable_motion_blur=True, exposure=1.0)
        def process(self, hdr_texture, target_framebuffer)
        def resize(self, render_width, render_height, screen_width, screen_height)
        def set_params(self, **kwargs)

NOTE « sans GPU » :
    Ce module doit passer `py_compile` (et meme s'importer) SANS contexte
    OpenGL. Toute la creation de ressources GL est confinee a __init__/_build_*
    et n'est touchee qu'a l'execution reelle ; l'import seul ne cree rien.
"""

import os

import numpy as np

try:
    import moderngl
except Exception:  # pragma: no cover - moderngl absent en CI pure
    # On tolere l'absence de moderngl pour que py_compile / l'import reste vert.
    moderngl = None


# =============================================================================
#  BLOC DE PARAMETRES « REGLABLES » (tunable header)
#  -------------------------------------------------------------------------
#  Valeurs par defaut exposees ici pour reglage rapide. L'integrateur peut les
#  surcharger a la volee via PostProcessor.set_params(**kwargs).
# =============================================================================
DEFAULT_EXPOSURE = 1.0            # exposition lineaire avant tonemap (1 = neutre)
# Résolution de RÉFÉRENCE pour calibrer l'exposition. En blending additif avec une
# taille de point CONSTANTE en pixels, la luminosité d'un pixel ∝ recouvrement de
# sprites ∝ 1/(pixels de rendu) : la MÊME scène est plus sombre à plus haute
# résolution. On met donc l'exposition à l'échelle ∝ pixels de rendu, calibrée sur
# 1280×720 (fenêtre par défaut) -> un réglage EXPOSURE donné rend la MÊME luminosité
# quelle que soit la résolution / le supersampling. cf. _effective_exposure().
_EXPOSURE_REF_PIXELS = 1280 * 720

DEFAULT_BLOOM_THRESHOLD = 1.0     # seuil de luminance du bright-pass (HDR)
DEFAULT_BLOOM_KNEE = 0.5          # largeur du genou doux (soft knee) du seuil
DEFAULT_BLOOM_INTENSITY = 0.6     # gain du bloom lors de la recomposition

DEFAULT_MOTION_BLUR_STRENGTH = 0.7  # force de la trainee temporelle [0,1[

DEFAULT_VIGNETTE_STRENGTH = 0.35  # intensite de la vignette (0 = desactivee)

# Debruitage a-trous (lisse le speckle des particules eparses).
DEFAULT_DENOISE_SIGMA = 1.0       # force (edge-stopping luminance) ; + grand = + lisse
DEFAULT_DENOISE_ITERS = 4         # nb de passes a-trous (pas 1,2,4,8 -> rayon large)

# Nombre de niveaux de la pyramide de bloom. Plus de niveaux = halo plus large
# et plus doux (chaque niveau est 2x plus petit -> couvre une zone 2x plus large
# en espace ecran pour un cout en baisse geometrique).
BLOOM_MIP_COUNT = 5

# Operateurs de tone mapping disponibles (mappes vers un entier cote shader).
TONEMAP_MODES = {"aces": 0, "uncharted2": 1}
DEFAULT_TONEMAP = "aces"

# Dossier des shaders, relatif a ce fichier.
_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


def _load_shader(name):
    """Lit le source d'un shader depuis le dossier shaders/.

    Garde volontairement simple : pas de cache, pas de #include (moderngl ne
    gere pas #include ; nos passes post-FX n'en ont pas besoin)."""
    path = os.path.join(_SHADER_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class PostProcessor:
    """Chaine de post-traitement plein ecran.

    Toutes les passes partagent UN seul VAO « triangle plein ecran » (sans VBO,
    cf. fullscreen.vert) : on dessine 3 sommets generes depuis gl_VertexID.

    Les FBO intermediaires sont en RGBA16F (HDR lineaire) jusqu'a la passe de
    composition/tonemap qui sort en SDR/sRGB.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, ctx, render_width, render_height,
                 screen_width, screen_height,
                 enable_bloom=True, enable_motion_blur=True, exposure=1.0):
        self.ctx = ctx

        # --- Dimensions ---------------------------------------------------
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        # --- Parametres reglables (etat courant) --------------------------
        self.enable_bloom = bool(enable_bloom)
        self.enable_motion_blur = bool(enable_motion_blur)
        self.exposure = float(exposure)

        self.bloom_threshold = DEFAULT_BLOOM_THRESHOLD
        self.bloom_knee = DEFAULT_BLOOM_KNEE
        self.bloom_intensity = DEFAULT_BLOOM_INTENSITY
        self.motion_blur_strength = DEFAULT_MOTION_BLUR_STRENGTH
        self.vignette_strength = DEFAULT_VIGNETTE_STRENGTH
        self.enable_vignette = DEFAULT_VIGNETTE_STRENGTH > 0.0
        self.enable_dof = False  # bonus, no-op pour l'instant
        self.tonemap_mode = TONEMAP_MODES.get(DEFAULT_TONEMAP, 0)

        # Debruitage (off par defaut ; active par l'Integrator via set_params).
        self.enable_denoise = False
        self.denoise_sigma = DEFAULT_DENOISE_SIGMA
        self.denoise_iters = DEFAULT_DENOISE_ITERS

        # --- Etat du buffer d'historique (motion blur) --------------------
        # On bascule (ping-pong) entre deux FBO d'historique ; True a la
        # premiere frame pour ne pas melanger un contenu indefini.
        self._first_frame = True
        self._history_index = 0

        # Conteneurs de ressources GL (remplis par _build_gl_resources).
        self._programs = {}
        self._vao_cache = {}
        self.vao = None

        # Sans contexte (py_compile / import « a sec »), on s'arrete la :
        # aucune ressource GL n'est creee tant qu'un vrai ctx n'est pas fourni.
        if self.ctx is None or moderngl is None:
            return

        self._build_programs()
        self._build_fullscreen_vao()
        self._build_framebuffers()

    # ------------------------------------------------------------------ #
    #  Construction des ressources GL                                     #
    # ------------------------------------------------------------------ #
    def _build_programs(self):
        """Compile tous les programmes (vertex commun + un fragment par passe)."""
        vert = _load_shader("fullscreen.vert")

        def prog(frag_name):
            return self.ctx.program(vertex_shader=vert,
                                    fragment_shader=_load_shader(frag_name))

        self._programs = {
            "brightpass": prog("brightpass.frag"),
            "blur": prog("blur.frag"),
            "motionblur": prog("motionblur.frag"),
            "composite": prog("composite_tonemap.frag"),
            "upscale": prog("lanczos_upscale.frag"),
            "denoise": prog("denoise.frag"),
            # Passe de copie/blit triviale via le meme fragment de composition
            # n'est pas necessaire : on reutilise blur/upscale au besoin.
        }

    def _build_fullscreen_vao(self):
        """Cree le VAO « triangle plein ecran » partage par toutes les passes.

        Aucun VBO : fullscreen.vert genere la geometrie depuis gl_VertexID.
        moderngl impose qu'un VAO soit lie a UN programme ; on en garde donc un
        par programme dans un petit cache, tous sans buffer d'attributs."""
        self._vao_cache = {}
        for key, program in self._programs.items():
            self._vao_cache[key] = self.ctx.vertex_array(program, [])
        # Alias pratique vers un VAO quelconque (non utilise directement).
        self.vao = next(iter(self._vao_cache.values()), None)

    def _make_color_fbo(self, width, height, dtype="f2", components=4):
        """Cree un FBO couleur (RGBA16F par defaut) avec filtrage lineaire.

        dtype 'f2' = 16 bits flottant (HDR). Le filtrage lineaire est essentiel
        pour les downsamples/blur (echantillonnage bilineaire materiel)."""
        tex = self.ctx.texture((max(1, int(width)), max(1, int(height))),
                               components, dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Clamp aux bords : evite que le bloom/blur « enroule » d'un cote a l'autre.
        tex.repeat_x = False
        tex.repeat_y = False
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        return fbo, tex

    def _build_framebuffers(self):
        """Alloue tous les FBO intermediaires en fonction des resolutions."""
        rw, rh = self.render_width, self.render_height

        # --- Motion blur : 2 buffers d'historique (ping-pong) a la resolution
        # de rendu, + un buffer pour la sortie melangee de la frame courante.
        self._history_fbos = [self._make_color_fbo(rw, rh) for _ in range(2)]
        # FBO recevant le resultat du motion blur (= scene « courante effective »).
        self._motion_fbo, self._motion_tex = self._make_color_fbo(rw, rh)

        # --- Debruitage : 2 FBO ping-pong a la resolution de rendu (HDR f2). ---
        self._denoise_fbos = [self._make_color_fbo(rw, rh) for _ in range(2)]

        # --- Bloom : pyramide de mips. Le niveau 0 a la moitie de la resolution
        # de rendu (le bright-pass downsample deja x2 -> moins de fetchs, halo
        # plus large « gratuitement »). Chaque niveau suivant est encore /2.
        self._bloom_levels = []   # (fbo, tex, (w,h)) pour chaque niveau
        self._bloom_pingpong = [] # (fbo, tex) temporaire pour la passe H du blur
        w, h = max(1, rw // 2), max(1, rh // 2)
        for _ in range(BLOOM_MIP_COUNT):
            fbo, tex = self._make_color_fbo(w, h)
            pp_fbo, pp_tex = self._make_color_fbo(w, h)
            self._bloom_levels.append((fbo, tex, (w, h)))
            self._bloom_pingpong.append((pp_fbo, pp_tex))
            w = max(1, w // 2)
            h = max(1, h // 2)

        # --- Composition : on ecrit dans un FBO HDR->SDR intermediaire SI un
        # downscale supersampling est ensuite necessaire (rendu > ecran).
        self._needs_downscale = (self.render_width > self.screen_width or
                                 self.render_height > self.screen_height)
        if self._needs_downscale:
            # La composition sort en SDR a la resolution de RENDU, puis la passe
            # de downscale Lanczos ramene a la resolution ecran.
            self._composite_fbo, self._composite_tex = self._make_color_fbo(
                rw, rh, dtype="f1")  # f1 = 8 bits suffisant apres tonemap
        else:
            self._composite_fbo = None
            self._composite_tex = None

    def _release_framebuffers(self):
        """Libere les FBO/textures intermediaires (avant un resize)."""
        def _rel(pair):
            fbo, tex = pair[0], pair[1]
            try:
                fbo.release()
                tex.release()
            except Exception:
                pass

        for hf in getattr(self, "_history_fbos", []):
            _rel(hf)
        if getattr(self, "_motion_fbo", None) is not None:
            _rel((self._motion_fbo, self._motion_tex))
        for df in getattr(self, "_denoise_fbos", []):
            _rel(df)
        for lvl in getattr(self, "_bloom_levels", []):
            _rel(lvl)
        for pp in getattr(self, "_bloom_pingpong", []):
            _rel(pp)
        if getattr(self, "_composite_fbo", None) is not None:
            _rel((self._composite_fbo, self._composite_tex))

    # ------------------------------------------------------------------ #
    #  Reglages a la volee                                                #
    # ------------------------------------------------------------------ #
    def set_params(self, **kwargs):
        """Met a jour les parametres reglables a chaud (toggles live).

        Cles reconnues : enable_bloom, bloom_threshold, bloom_knee,
        bloom_intensity, enable_motion_blur, motion_blur_strength, exposure,
        tonemap ('aces'|'uncharted2'), enable_vignette, vignette_strength,
        enable_dof. Les cles inconnues sont ignorees silencieusement."""
        if "enable_bloom" in kwargs:
            self.enable_bloom = bool(kwargs["enable_bloom"])
        if "bloom_threshold" in kwargs:
            self.bloom_threshold = float(kwargs["bloom_threshold"])
        if "bloom_knee" in kwargs:
            self.bloom_knee = float(kwargs["bloom_knee"])
        if "bloom_intensity" in kwargs:
            self.bloom_intensity = float(kwargs["bloom_intensity"])
        if "enable_motion_blur" in kwargs:
            self.enable_motion_blur = bool(kwargs["enable_motion_blur"])
        if "motion_blur_strength" in kwargs:
            # On borne dans [0, 0.999] : 1.0 figerait l'historique a jamais.
            self.motion_blur_strength = float(
                np.clip(kwargs["motion_blur_strength"], 0.0, 0.999))
        if "exposure" in kwargs:
            self.exposure = float(kwargs["exposure"])
        if "tonemap" in kwargs:
            self.tonemap_mode = TONEMAP_MODES.get(
                str(kwargs["tonemap"]).lower(), self.tonemap_mode)
        if "enable_vignette" in kwargs:
            self.enable_vignette = bool(kwargs["enable_vignette"])
        if "vignette_strength" in kwargs:
            self.vignette_strength = float(kwargs["vignette_strength"])
        if "enable_dof" in kwargs:
            self.enable_dof = bool(kwargs["enable_dof"])
        if "enable_denoise" in kwargs:
            self.enable_denoise = bool(kwargs["enable_denoise"])
        if "denoise_sigma" in kwargs:
            self.denoise_sigma = max(1e-3, float(kwargs["denoise_sigma"]))
        if "denoise_iters" in kwargs:
            self.denoise_iters = max(1, int(kwargs["denoise_iters"]))

    # ------------------------------------------------------------------ #
    #  Exposition corrigee de la resolution                               #
    # ------------------------------------------------------------------ #
    def _effective_exposure(self):
        """Exposition CORRIGEE DE LA RESOLUTION. En blending additif avec une taille
        de point CONSTANTE (en pixels), la luminosite d'un pixel vient du RECOUVREMENT
        de sprites et vaut ~ N*S^2/(pixels de rendu) -> elle decroit en 1/(pixels de
        rendu) : la MEME scene est plus sombre a plus haute resolution. Pour qu'un
        reglage EXPOSURE donne rende la MEME luminosite a toute resolution (et tout
        supersampling, le downscale moyennant et preservant la moyenne), on met
        l'exposition a l'echelle proportionnellement aux pixels de rendu, calibree sur
        _EXPOSURE_REF_PIXELS (1280x720). A la resolution de reference -> facteur 1."""
        px = max(1, self.render_width * self.render_height)
        return self.exposure * (px / _EXPOSURE_REF_PIXELS)

    # ------------------------------------------------------------------ #
    #  Redimensionnement                                                  #
    # ------------------------------------------------------------------ #
    def resize(self, render_width, render_height, screen_width, screen_height):
        """Re-alloue les FBO aux nouvelles resolutions (rendu et/ou ecran).

        Appele par l'integrateur quand la fenetre change de taille. On force le
        prochain frame a etre « premiere frame » pour ne pas trainer un
        historique de mauvaise dimension."""
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        if self.ctx is None or moderngl is None:
            return

        self._release_framebuffers()
        self._build_framebuffers()
        self._first_frame = True
        self._history_index = 0

    # ------------------------------------------------------------------ #
    #  Helpers de rendu                                                   #
    # ------------------------------------------------------------------ #
    def _draw(self, vao_key, fbo):
        """Rend une passe plein ecran (3 sommets) dans le FBO cible.

        On desactive depth/blend : les passes post-FX ecrivent franchement leur
        couleur (pas de melange materiel ; les additions se font dans le shader)."""
        fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self._vao_cache[vao_key].render(mode=moderngl.TRIANGLES, vertices=3)

    # ------------------------------------------------------------------ #
    #  Passe BLOOM                                                        #
    # ------------------------------------------------------------------ #
    def _render_bloom(self, source_tex):
        """Construit la pyramide de bloom a partir de la scene HDR (source_tex)
        et retourne la texture du niveau 0 (bloom accumule pret a composer).

        Etapes :
          a) bright-pass : extrait les hautes lumieres dans le niveau 0.
          b) downsample : remplit chaque niveau a partir du precedent (le blur
             gaussien sur une cible 2x plus petite agit comme un downsample).
          c) blur separable H+V a chaque niveau (FBO ping-pong).
          d) upsample additif : on remonte la pyramide en ajoutant chaque niveau
             flou au niveau superieur -> halo large et continu.
        """
        prog_bp = self._programs["brightpass"]
        prog_blur = self._programs["blur"]

        # --- a) Bright-pass : scene HDR -> niveau 0 de la pyramide ----------
        lvl0_fbo, lvl0_tex, lvl0_size = self._bloom_levels[0]
        source_tex.use(location=0)
        prog_bp["u_scene"].value = 0
        prog_bp["u_threshold"].value = self.bloom_threshold
        prog_bp["u_knee"].value = self.bloom_knee
        self._draw("brightpass", lvl0_fbo)

        # --- b) + c) Downsample en cascade + blur a chaque niveau ----------
        # Chaque niveau N>0 est rempli en floutant le niveau N-1 (filtrage
        # bilineaire sur cible plus petite = downsample doux), puis on floute
        # encore le niveau lui-meme pour elargir le noyau effectif.
        for i in range(BLOOM_MIP_COUNT):
            fbo, tex, (w, h) = self._bloom_levels[i]
            pp_fbo, pp_tex = self._bloom_pingpong[i]
            texel = (1.0 / w, 1.0 / h)

            if i > 0:
                # Downsample : source = niveau precedent, cible = niveau i.
                src_fbo, src_tex, _ = self._bloom_levels[i - 1]
                src_tex.use(location=0)
                prog_blur["u_image"].value = 0
                prog_blur["u_texel_size"].value = texel
                prog_blur["u_direction"].value = (1.0, 0.0)
                self._draw("blur", fbo)  # ecrit le downsample floute horizontal

            # Passe H : niveau i -> ping-pong (flou horizontal).
            tex.use(location=0)
            prog_blur["u_image"].value = 0
            prog_blur["u_texel_size"].value = texel
            prog_blur["u_direction"].value = (1.0, 0.0)
            self._draw("blur", pp_fbo)

            # Passe V : ping-pong -> niveau i (flou vertical). Resultat : niveau
            # i contient sa version gaussienne complete (separable H puis V).
            pp_tex.use(location=0)
            prog_blur["u_image"].value = 0
            prog_blur["u_texel_size"].value = texel
            prog_blur["u_direction"].value = (0.0, 1.0)
            self._draw("blur", fbo)

        # --- d) Upsample additif : on remonte du plus petit au plus grand ---
        # On utilise le BLEND materiel additif pour cumuler chaque niveau flou
        # dans le niveau superieur (filtrage lineaire = upscale doux).
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # additif pur
        for i in range(BLOOM_MIP_COUNT - 1, 0, -1):
            src_fbo, src_tex, _ = self._bloom_levels[i]
            dst_fbo, dst_tex, (dw, dh) = self._bloom_levels[i - 1]
            dst_fbo.use()
            src_tex.use(location=0)
            prog_blur["u_image"].value = 0
            prog_blur["u_texel_size"].value = (1.0 / dw, 1.0 / dh)
            prog_blur["u_direction"].value = (0.0, 0.0)  # 0 = simple upscale
            # direction nulle -> step nul -> les 7 fetchs lisent le meme texel,
            # ce qui revient a un simple echantillonnage bilineaire (upscale).
            self._vao_cache["blur"].render(mode=moderngl.TRIANGLES, vertices=3)
        self.ctx.disable(moderngl.BLEND)

        return lvl0_tex

    # ------------------------------------------------------------------ #
    #  Passe MOTION BLUR (buffer d'historique)                            #
    # ------------------------------------------------------------------ #
    def _render_motion_blur(self, scene_tex):
        """Melange la scene courante avec l'historique accumule et retourne la
        texture resultante. Met a jour le ping-pong d'historique.

        accum = lerp(courante, historique, strength)  -> trainee qui decroit.
        La sortie sert a la fois de « scene effective » pour la suite ET devient
        le nouvel historique pour la frame suivante.
        """
        prog = self._programs["motionblur"]

        # Le FBO d'historique « ecriture » est celui qu'on va remplir cette frame.
        write_idx = self._history_index
        read_idx = 1 - write_idx
        write_fbo, write_tex = self._history_fbos[write_idx]
        read_fbo, read_tex = self._history_fbos[read_idx]

        scene_tex.use(location=0)
        read_tex.use(location=1)
        prog["u_scene"].value = 0
        prog["u_history"].value = 1
        prog["u_strength"].value = self.motion_blur_strength
        prog["u_first_frame"].value = 1 if self._first_frame else 0

        self._draw("motionblur", write_fbo)

        # On bascule le ping-pong : l'ecriture de cette frame est l'historique
        # de la suivante.
        self._history_index = read_idx
        self._first_frame = False
        return write_tex

    # ------------------------------------------------------------------ #
    #  Passe DÉBRUITAGE (à-trous edge-aware)                              #
    # ------------------------------------------------------------------ #
    def _render_denoise(self, src_tex):
        """À-trous edge-aware : `denoise_iters` passes en doublant le pas (1,2,
        4,8...) -> rayon large à faible coût. Lisse le bruit de speckle des
        particules éparses tout en préservant les filaments lumineux. Renvoie la
        texture débruitée (ping-pong entre 2 FBO)."""
        prog = self._programs["denoise"]
        texel = (1.0 / self.render_width, 1.0 / self.render_height)
        cur = src_tex
        step = 1.0
        dst_is_a = True
        for _ in range(self.denoise_iters):
            dst_fbo, dst_tex = self._denoise_fbos[0 if dst_is_a else 1]
            cur.use(location=0)
            prog["u_image"].value = 0
            prog["u_texel"].value = texel
            prog["u_step"].value = step
            prog["u_sigma_l"].value = self.denoise_sigma
            self._draw("denoise", dst_fbo)
            cur = dst_tex
            dst_is_a = not dst_is_a
            step *= 2.0
        return cur

    # ------------------------------------------------------------------ #
    #  Boucle principale de post-traitement                               #
    # ------------------------------------------------------------------ #
    def process(self, hdr_texture, target_framebuffer):
        """Applique toute la chaine post-FX.

        hdr_texture       : moderngl.Texture RGBA16F (HDR lineaire, pre-tonemap)
                            a la resolution de RENDU (fournie par renderer.py).
        target_framebuffer: framebuffer ecran/par-defaut ou dessiner l'image
                            finale SDR (fourni par l'integrateur).
        """
        if self.ctx is None or moderngl is None:
            return  # garde « sans GPU »

        # ----- 1. Motion blur (buffer d'historique) ------------------------
        # On le fait EN PREMIER pour que le bloom et le tonemap operent sur la
        # frame deja « tracee » (la trainee herite donc aussi du glow).
        if self.enable_motion_blur and self.motion_blur_strength > 0.0:
            scene_tex = self._render_motion_blur(hdr_texture)
        else:
            scene_tex = hdr_texture
            # On garde l'historique « propre » pour eviter un saut si on
            # reactive le motion blur : la prochaine activation repartira frais.
            self._first_frame = True

        # ----- 1b. Débruitage à-trous (lisse le speckle, garde les filaments) --
        if self.enable_denoise:
            scene_tex = self._render_denoise(scene_tex)

        # ----- 2. Bloom ----------------------------------------------------
        bloom_tex = None
        if self.enable_bloom:
            bloom_tex = self._render_bloom(scene_tex)

        # ----- 3. Composition + exposition + tonemap + sRGB ---------------
        prog = self._programs["composite"]
        scene_tex.use(location=0)
        prog["u_scene"].value = 0
        prog["u_exposure"].value = self._effective_exposure()   # corrigee de la resolution
        prog["u_tonemap_mode"].value = self.tonemap_mode
        prog["u_enable_bloom"].value = 1 if (self.enable_bloom and
                                             bloom_tex is not None) else 0
        prog["u_bloom_intensity"].value = self.bloom_intensity
        if bloom_tex is not None:
            bloom_tex.use(location=1)
            prog["u_bloom"].value = 1
        else:
            # Aucun bloom : on lie quand meme la scene a l'unite 1 pour ne pas
            # laisser un sampler non initialise (lecture ignoree cote shader).
            scene_tex.use(location=1)
            prog["u_bloom"].value = 1
        prog["u_enable_vignette"].value = 1 if self.enable_vignette else 0
        prog["u_vignette_strength"].value = self.vignette_strength

        # Cale le viewport de la cible FINALE (écran / framebuffer par défaut) sur
        # la résolution écran courante. Sans ça, après un redimensionnement de
        # fenêtre (passage plein écran / maximisation), moderngl conserve le
        # viewport périmé du framebuffer par défaut et l'image resterait dessinée
        # dans l'ancien coin, aux anciennes dimensions.
        try:
            target_framebuffer.viewport = (0, 0, self.screen_width, self.screen_height)
        except Exception:
            pass

        if self._needs_downscale:
            # Compose en SDR a la resolution de rendu, puis downscale Lanczos.
            self._draw("composite", self._composite_fbo)
            self._render_downscale(self._composite_tex, target_framebuffer)
        else:
            # Pas de supersampling : on compose directement a l'ecran.
            self._draw("composite", target_framebuffer)

    # ------------------------------------------------------------------ #
    #  Passe SUPERSAMPLING / DOWNSCALE                                    #
    # ------------------------------------------------------------------ #
    def _render_downscale(self, src_tex, target_framebuffer):
        """Ramene l'image (resolution de rendu) a la resolution ecran via
        Lanczos-2. C'est ici qu'on brancherait DLSS/TensorRT/ONNX ou une TAA
        (cf. lanczos_upscale.frag)."""
        prog = self._programs["upscale"]
        src_tex.use(location=0)
        prog["u_image"].value = 0
        prog["u_src_texel"].value = (1.0 / self.render_width,
                                     1.0 / self.render_height)
        self._draw("upscale", target_framebuffer)

    # ------------------------------------------------------------------ #
    #  Nettoyage                                                          #
    # ------------------------------------------------------------------ #
    def release(self):
        """Libere toutes les ressources GL (programmes, VAO, FBO)."""
        if self.ctx is None or moderngl is None:
            return
        self._release_framebuffers()
        for vao in self._vao_cache.values():
            try:
                vao.release()
            except Exception:
                pass
        for prog in self._programs.values():
            try:
                prog.release()
            except Exception:
                pass
