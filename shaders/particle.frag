#version 460 core
// =============================================================================
//  particle.frag
//  -------------------------------------------------------------------------
//  Fragment shader des PARTICULES : transforme chaque point GL_POINTS en un
//  petit DISQUE LUMINEUX GAUSSIEN (sprite doux), additionne en HDR.
//
//  SPRITE GAUSSIEN (disque doux) :
//    gl_PointCoord donne la coordonnee [0,1]^2 a l'interieur du point rasterise.
//    On la recentre en [-1,1] (centre = 0). r2 = distance au carre au centre.
//    L'intensite suit une gaussienne exp(-k * r2) : maximum au centre, fondu
//    doux vers les bords -> aucune arete dure, aspect "halo" volumetrique.
//    On coupe au-dela du rayon 1 (cercle inscrit) pour rester rond.
//
//  COULEUR HDR + BRIGHTNESS :
//    La couleur finale = couleur HDR de la particule (v_color.rgb, lineaire,
//    peut depasser 1) * brightness (eclat par particule, v_brightness) *
//    profil gaussien. On reste en LINEAIRE et PRE-TONEMAP : le post-traitement
//    (bloom, tonemap) consomme ce HDR en aval. Pas de gamma ici.
//
//  BLENDING ADDITIF :
//    Le renderer active glBlendFunc(GL_ONE, GL_ONE) : chaque particule AJOUTE
//    sa lumiere a ce qui est deja dans le framebuffer. C'est ce qui cree le
//    nuage lumineux volumetrique (les zones denses cumulent l'energie et
//    saturent en blanc chaud). En additif, le canal alpha de sortie n'est pas
//    utilise comme opacite (pas de mix) ; on module plutot la COULEUR par
//    l'alpha et la gaussienne, puis on sort alpha=intensite a titre indicatif.
//    On n'ecrit donc PAS de "trou noir" sur les bords : un fragment hors du
//    disque a une contribution nulle (vec3(0)) -> neutre pour l'addition.
// =============================================================================

in vec4  v_color;       // couleur HDR lineaire (r,g,b,a)
in float v_brightness;  // multiplicateur d'eclat par particule (in_pos.w)

out vec4 f_color;

// Raideur de la gaussienne : plus c'est grand, plus le coeur est concentre et
// le halo serre. ~5.0 donne un joli point lumineux a coeur net et bord fondu.
const float GAUSS_K = 5.0;

void main() {
    // --- Coordonnee locale dans le sprite, recentree en [-1, 1] ---
    vec2 uv = gl_PointCoord * 2.0 - 1.0;   // centre du point -> (0,0)
    float r2 = dot(uv, uv);                // distance^2 au centre

    // --- Decoupe circulaire : hors du cercle inscrit, on ne contribue rien ---
    // (discard evite d'ecrire des fragments inutiles ; en additif un alpha 0
    //  suffirait, mais discard est plus propre et un poil moins couteux.)
    if (r2 > 1.0) {
        discard;
    }

    // --- Profil gaussien doux : 1 au centre, fondu vers 0 au bord ---
    float gauss = exp(-GAUSS_K * r2);

    // --- Couleur HDR finale ---
    // couleur lineaire * eclat par particule * alpha par particule * gaussienne.
    // brightness peut etre > 1 (coeur tres lumineux) -> on laisse deborder en HDR.
    vec3 rgb = v_color.rgb * v_brightness * v_color.a * gauss;

    // En additif (GL_ONE, GL_ONE), seule la composante RGB compte vraiment ;
    // on renseigne alpha = gauss a titre informatif (utile si une passe en aval
    // voulait lire une couverture). La luminosite vit dans le RGB HDR.
    f_color = vec4(rgb, gauss);
}
