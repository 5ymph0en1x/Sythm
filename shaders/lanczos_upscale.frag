#version 460 core
// =============================================================================
//  lanczos_upscale.frag
//  -------------------------------------------------------------------------
//  Redimensionnement de qualite (supersampling DOWNSCALE principalement, mais
//  fonctionne aussi en upscale) par filtre de LANCZOS-2.
//
//  CONTEXTE :
//    Le rendu se fait en SUPERSAMPLE (resolution interne > resolution ecran,
//    ex. 1.5x..2x) pour un anti-aliasing "brute force" tres propre sur le nuage
//    de particules. Il faut ensuite ramener cette image haute-res a la taille
//    de l'ecran. Un simple bilineaire flouterait les details ; Lanczos
//    reconstruit le signal avec un noyau sinc fenetre -> nettete superieure,
//    proche de l'ideal theorique, avec un leger "ringing" maitrise.
//
//  NOYAU LANCZOS-2 :
//    L(x) = sinc(x) * sinc(x/2)   pour |x| < 2, 0 sinon
//    avec sinc(x) = sin(pi x)/(pi x).  On echantillonne une fenetre 4x4 texels
//    autour de la position source et on pondere chaque texel par le produit des
//    poids Lanczos horizontaux et verticaux (noyau separable applique en 2D).
//
//  COUT : 16 fetchs/pixel. Sur RTX 4090 a 4K c'est negligeable. Si besoin de
//  reduire, voir le fallback "bilineaire + sharpen" documente plus bas.
//
//  -------------------------------------------------------------------------
//  OU BRANCHER DLSS / TensorRT / ONNX / TAA :
//    Cette passe de mise a l'echelle spatiale "classique" est exactement
//    l'endroit ou inserer un upscaler NEURONAL (DLSS, FSR, ou un modele ONNX
//    custom tournant sur les Tensor Cores via TensorRT). Le principe :
//      - rendre a BASSE resolution (perf),
//      - fournir au reseau : couleur basse-res + vecteurs de mouvement
//        (velocity buffer, cf. motionblur.frag) + jitter sub-pixel par frame,
//      - le reseau reconstruit une image HAUTE-res temporellement stable.
//    Cote moderngl on n'a pas d'acces direct a DLSS (API NVNGX/DX/Vulkan) ;
//    l'integration passerait par un module externe (CUDA/TensorRT) ecrivant
//    dans une texture interop, OU par une passe TAA "maison" : accumuler
//    plusieurs frames jittered avec reprojection par velocity buffer. La passe
//    TAA prendrait la place de ce shader, en amont du tonemap idealement.
// =============================================================================

in  vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_image;       // image source (haute resolution, RGBA16F)
uniform vec2      u_src_texel;   // 1.0 / taille_source (en pixels)

const float PI = 3.14159265358979323846;

// sinc(x) = sin(pi x) / (pi x), avec la limite sinc(0) = 1.
float sinc(float x) {
    if (abs(x) < 1e-4) return 1.0;
    float px = PI * x;
    return sin(px) / px;
}

// Noyau Lanczos-2 : produit de deux sinc, support |x| < 2.
float lanczos2(float x) {
    if (abs(x) >= 2.0) return 0.0;
    return sinc(x) * sinc(x * 0.5);
}

void main() {
    // Position en coordonnees de texels source.
    vec2 src_pos = v_uv / u_src_texel;
    // Centre du texel le plus proche (en coordonnees texel, demi-pixel).
    vec2 center = floor(src_pos - 0.5) + 0.5;

    vec3 accum = vec3(0.0);
    float wsum = 0.0;

    // Fenetre 4x4 : decalages -1,0,1,2 autour du texel central.
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            vec2 tap = center + vec2(float(i), float(j));
            // Distance (en texels) du tap a la position source reelle.
            vec2 dxy = src_pos - tap;
            float w = lanczos2(dxy.x) * lanczos2(dxy.y);
            // Re-echantillonnage au centre du texel (coords UV normalisees).
            vec3 c = texture(u_image, (tap) * u_src_texel).rgb;
            accum += c * w;
            wsum += w;
        }
    }

    // Normalisation (la somme des poids Lanczos n'est pas exactement 1).
    f_color = vec4(accum / max(wsum, 1e-5), 1.0);
}
