#version 460 core
// =============================================================================
//  brightpass.frag
//  -------------------------------------------------------------------------
//  Extraction des hautes lumieres (bright pass) pour le bloom, avec SOFT KNEE.
//
//  PRINCIPE :
//    Le bloom ne doit eclore que sur les pixels DEPASSANT un seuil de
//    luminance (bloom_threshold). Un seuil "dur" (step) cree une transition
//    brutale et un scintillement (aliasing temporel) tres visible sur un nuage
//    de particules en mouvement. On utilise donc un SOFT KNEE : une courbe
//    quadratique douce autour du seuil qui fait monter le bloom progressivement.
//    Formule inspiree de l'implementation de Jorge Jimenez / Unity (COD AW).
//
//  ENTREE  : texture HDR (apres motion blur) en RGBA16F, valeurs > 1 frequentes.
//  SORTIE  : seulement la portion "lumiere a faire saigner", en RGBA16F.
//
//  Cette passe ecrit en general dans le premier mip (le plus grand) de la
//  pyramide de bloom ; les niveaux suivants sont des downsamples successifs.
// =============================================================================

in  vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;       // source HDR
uniform float     u_threshold;   // bloom_threshold (seuil de luminance)
uniform float     u_knee;        // largeur du "genou" doux (>0). ~0.5*threshold

void main() {
    vec3 c = texture(u_scene, v_uv).rgb;

    // Luminance percue (coefficients Rec.709). On pourrait aussi prendre
    // max(r,g,b) pour un bloom plus agressif ; la luminance est plus naturelle.
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));

    // --- Soft knee (courbe quadratique de raccord autour du seuil) ---
    // knee = rayon de transition ; on evite la division par zero.
    float knee = max(u_knee, 1e-4);
    // 'soft' monte de 0 a knee*2 sur la zone [threshold-knee, threshold+knee].
    float soft = clamp(lum - u_threshold + knee, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1e-5);
    // Contribution finale : max entre la partie franchement au-dessus du seuil
    // et la rampe douce. Divise par lum pour obtenir un facteur multiplicatif.
    float contribution = max(soft, lum - u_threshold);
    contribution /= max(lum, 1e-5);

    // On conserve la teinte HDR d'origine, juste attenuee par 'contribution'.
    f_color = vec4(c * contribution, 1.0);
}
