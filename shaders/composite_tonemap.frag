#version 460 core
// =============================================================================
//  composite_tonemap.frag
//  -------------------------------------------------------------------------
//  Passe FINALE de composition : recombinaison du bloom + exposition +
//  tone mapping HDR -> SDR + correction sRGB + vignette optionnelle.
//
//  ORDRE DES OPERATIONS (volontaire, pour garder du "punch") :
//    1. Recomposition ADDITIVE : couleur HDR de la scene + somme ponderee des
//       niveaux flous de la pyramide de bloom. L'addition (et non un mix) est
//       ce qui fait "saigner" la lumiere au-dela des bords -> glow lumineux.
//    2. Exposition : multiplication lineaire avant le tonemap (simule
//       l'ouverture/temps de pose d'une camera). Se fait en lineaire HDR.
//    3. Tone mapping : compression de la plage HDR (0..inf) vers [0,1] de
//       maniere filmique, en preservant les hautes lumieres sans les ecreter
//       brutalement. Deux operateurs proposes (ACES / Uncharted2).
//    4. Vignette (optionnelle, bonus) : assombrissement doux des bords, cadre
//       l'image et renforce la profondeur sur fond noir.
//    5. Correction sRGB (gamma ~2.2) : l'ecran attend du sRGB, on encode la
//       sortie lineaire. On le fait NOUS-MEMES (framebuffer par defaut non-sRGB,
//       cf. window.py qui desactive SRGB_CAPABLE) pour maitriser la courbe.
//
//  Cette passe ecrit directement dans le framebuffer SDR final (ou dans un
//  buffer intermediaire si une passe de downscale supersampling suit).
// =============================================================================

in  vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;        // scene HDR lineaire (apres motion blur)
uniform sampler2D u_bloom;        // bloom deja accumule/flou (pyramide fusionnee)

uniform float u_exposure;         // exposition lineaire (1.0 = neutre)
uniform float u_bloom_intensity;  // gain applique au bloom avant addition
uniform int   u_enable_bloom;     // 1 = ajoute le bloom, 0 = scene pure
uniform int   u_tonemap_mode;     // 0 = ACES filmic, 1 = Uncharted2

uniform int   u_enable_vignette;  // 1 = applique la vignette
uniform float u_vignette_strength;// intensite de l'assombrissement des bords

// -----------------------------------------------------------------------------
//  ACES Filmic (approximation "RRT+ODT fit" de Stephen Hill / Krzysztof Narkowicz)
//  -------------------------------------------------------------------------
//  Courbe filmique standard de l'industrie : rend des hautes lumieres douces
//  et des couleurs satures sans virer au blanc cassant. Le "fit" polynomial
//  ci-dessous evite le passage couteux par les matrices ACES completes tout en
//  restant tres proche visuellement -> ideal temps reel.
// -----------------------------------------------------------------------------
vec3 tonemap_aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// -----------------------------------------------------------------------------
//  Uncharted2 (operateur "filmic" de John Hable, Naughty Dog)
//  -------------------------------------------------------------------------
//  Alternative au rendu plus contraste / plus "cinema sombre". On applique la
//  courbe a la couleur PUIS a un point blanc de reference (W) pour normaliser,
//  ce qui garantit que la valeur W se mappe sur 1.0 en sortie.
// -----------------------------------------------------------------------------
vec3 hable_partial(vec3 x) {
    const float A = 0.15;  // shoulder strength
    const float B = 0.50;  // linear strength
    const float C = 0.10;  // linear angle
    const float D = 0.20;  // toe strength
    const float E = 0.02;  // toe numerator
    const float F = 0.30;  // toe denominator
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemap_uncharted2(vec3 color) {
    const float W = 11.2;  // point blanc lineaire de reference
    const float exposure_bias = 2.0;
    vec3 curr = hable_partial(color * exposure_bias);
    vec3 white_scale = vec3(1.0) / hable_partial(vec3(W));
    return clamp(curr * white_scale, 0.0, 1.0);
}

// -----------------------------------------------------------------------------
//  Encodage sRGB precis (courbe a deux segments : lineaire pres du noir,
//  puissance ~2.4 ailleurs). Plus correct qu'un simple pow(x, 1/2.2).
// -----------------------------------------------------------------------------
vec3 linear_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, step(c, vec3(0.0031308)));
}

void main() {
    // --- 1. Recomposition additive du bloom -----------------------------
    vec3 hdr = texture(u_scene, v_uv).rgb;
    if (u_enable_bloom == 1) {
        vec3 bloom = texture(u_bloom, v_uv).rgb;
        hdr += bloom * u_bloom_intensity;
    }

    // --- 2. Exposition (en lineaire, avant tonemap) ----------------------
    hdr *= u_exposure;

    // --- 3. Tone mapping HDR -> [0,1] ------------------------------------
    vec3 mapped = (u_tonemap_mode == 1) ? tonemap_uncharted2(hdr)
                                        : tonemap_aces(hdr);

    // --- 4. Vignette optionnelle -----------------------------------------
    if (u_enable_vignette == 1) {
        // GEOMETRIE fixe (INDEPENDANTE de la force) : attenuation radiale douce,
        // 1.0 au centre -> ~0.1 dans les coins. dist^2 evite un sqrt.
        vec2 d = v_uv - 0.5;
        float dist2 = dot(d, d);                      // rayon^2 (coin ~ 0.5)
        float falloff = smoothstep(0.6, 0.1, dist2);  // 1 centre -> ~0 coins
        // u_vignette_strength = QUANTITE d'assombrissement aux bords (0..1), et NON
        // le rayon. Plancher (1 - s) -> les coins ne tombent jamais a zero, meme a
        // s=1 (on clampe s pour eviter le noir pur a strength>1).
        float s = clamp(u_vignette_strength, 0.0, 1.0);
        float vig = 1.0 - s * (1.0 - falloff);
        mapped *= vig;
    }

    // --- 5. Encodage sRGB pour l'affichage -------------------------------
    f_color = vec4(linear_to_srgb(mapped), 1.0);
}
