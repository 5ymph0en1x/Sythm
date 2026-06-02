#version 460 core
// =============================================================================
//  denoise.frag
//  -------------------------------------------------------------------------
//  Débruitage à-trous PRÉSERVANT LES CONTOURS (style SVGF sans variance).
//
//  La nuée de particules éparses produit un bruit de "speckle" haute fréquence
//  (chaque pixel reçoit un nombre aléatoire de particules -> bruit de Poisson).
//  Le signal sous-jacent (le champ de flux) étant LISSE, on peut lisser ce bruit
//  sans perdre la structure. Ce filtre fait une moyenne pondérée 5x5 où le poids
//  combine un noyau spatial (B3-spline) et un terme d'EDGE-STOPPING par
//  luminance : les voisins de luminance proche sont moyennés (bruit lissé), les
//  bords francs (filaments lumineux) sont préservés.
//
//  À-TROUS : on appelle ce shader plusieurs fois en doublant `u_step` (1,2,4,8).
//  Chaque passe élargit le support sans plus de taps -> grand rayon à coût bas.
//  EDGE-STOPPING RELATIF : la tolérance est proportionnelle à la luminance locale
//  -> robuste en HDR (lisse aussi bien les zones sombres que brillantes).
// =============================================================================

in  vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_image;    // image à débruiter (HDR linéaire)
uniform vec2      u_texel;    // 1.0 / résolution
uniform float     u_step;     // pas à-trous en pixels (1, 2, 4, 8, ...)
uniform float     u_sigma_l;  // force du débruitage (edge-stopping luminance)

const float K[5] = float[](0.0625, 0.25, 0.375, 0.25, 0.0625);  // B3-spline

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec3 c = texture(u_image, v_uv).rgb;
    float lc = luma(c);

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            vec2 off = vec2(float(i), float(j)) * u_step * u_texel;
            vec3 s = texture(u_image, v_uv + off).rgb;
            float ls = luma(s);
            float wk = K[i + 2] * K[j + 2];                  // noyau spatial
            // Edge-stopping relatif (HDR) : écart toléré ~ luminance locale.
            float denom = u_sigma_l * (lc + 0.3) + 1e-4;
            float wl = exp(-abs(lc - ls) / denom);
            float w = wk * wl;
            sum += s * w;
            wsum += w;
        }
    }
    f_color = vec4(sum / max(wsum, 1e-5), 1.0);
}
