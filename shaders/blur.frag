#version 460 core
// =============================================================================
//  blur.frag
//  -------------------------------------------------------------------------
//  Flou gaussien SEPARABLE (une dimension a la fois) pour le bloom.
//
//  PRINCIPE DE LA SEPARABILITE :
//    Un flou gaussien 2D de noyau (2N+1)x(2N+1) coute O(N^2) echantillons par
//    pixel. Mais la gaussienne 2D se factorise en produit de deux gaussiennes
//    1D : G2D(x,y) = G1D(x) * G1D(y). On peut donc l'appliquer en DEUX passes
//    1D successives (horizontale puis verticale), ramenant le cout a O(2N).
//    Cote Python on appelle ce meme shader deux fois en changeant u_direction.
//
//  OPTIMISATION "LINEAR SAMPLING" (taps a poids fusionnes) :
//    Au lieu d'echantillonner chaque texel a poids discret, on exploite le
//    filtrage bilineaire materiel : en placant le point d'echantillonnage
//    ENTRE deux texels a la bonne position fractionnaire, un seul fetch lit la
//    moyenne ponderee des deux texels. On halve ainsi le nombre de fetchs.
//    Les offsets/poids ci-dessous sont precalcules pour un noyau gaussien de
//    sigma ~2 (9 texels effectifs couverts par 3 fetchs de chaque cote + centre).
//    Reference : Rastergrid "Efficient Gaussian blur with linear sampling".
//
//  ENTREE/SORTIE : RGBA16F (on reste en HDR lineaire pendant tout le bloom).
//
//  NOTE : ce shader sert pour CHAQUE niveau de la pyramide de mips du bloom.
//  u_texel_size doit valoir 1.0/taille_du_mip_courant pour rester correct
//  quelle que soit la resolution du niveau flou.
// =============================================================================

in  vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_image;       // image a flouter (mip courant du bloom)
uniform vec2      u_texel_size;  // 1.0 / taille_du_mip (en pixels)
uniform vec2      u_direction;   // (1,0) passe horizontale, (0,1) verticale

// Poids et offsets fusionnes (linear sampling) pour une gaussienne douce.
// 1 tap central + 3 taps de chaque cote = 7 fetchs couvrant 13 texels.
const float OFFSETS[3] = float[](1.411764705882353,
                                 3.294117647058823,
                                 5.176470588235294);
const float WEIGHTS[3] = float[](0.2969069646728344,
                                 0.09447039785044732,
                                 0.010381362401148057);
const float WEIGHT_CENTER = 0.1964825501511404;

void main() {
    // Pas d'avancee en pixels selon la direction de la passe.
    vec2 step = u_direction * u_texel_size;

    // Contribution du texel central.
    vec3 result = texture(u_image, v_uv).rgb * WEIGHT_CENTER;

    // Taps symetriques de part et d'autre ; chaque fetch (grace au bilineaire)
    // agrege deja deux texels voisins -> noyau large pour peu de fetchs.
    for (int i = 0; i < 3; ++i) {
        vec2 off = step * OFFSETS[i];
        result += texture(u_image, v_uv + off).rgb * WEIGHTS[i];
        result += texture(u_image, v_uv - off).rgb * WEIGHTS[i];
    }

    f_color = vec4(result, 1.0);
}
