#version 460 core
// =============================================================================
//  particle.vert
//  -------------------------------------------------------------------------
//  Vertex shader des PARTICULES (rendu en GL_POINTS, un seul draw call pour les
//  N particules). Chaque sommet = une particule.
//
//  ATTRIBUTS (cf. contrat partage avec le ParticleSystem / interop CUDA) :
//    - in_pos  : vec4 = (x, y, z, brightness) en coordonnees MONDE.
//                xyz = position monde de la particule.
//                w   = "brightness", multiplicateur scalaire d'intensite que le
//                      fragment shader applique a la couleur (eclat par particule).
//    - in_col  : vec4 = (r, g, b, a) couleur HDR LINEAIRE (les composantes
//                peuvent depasser 1.0 ; a = opacite par particule).
//
//  Ces deux buffers sont REMPLIS par le ParticleSystem via interop CUDA : le
//  renderer ne fait que les lier comme attributs et dessiner. Layout strictement
//  contigu : N * vec4 pour chaque buffer, stride 16 octets, aucun padding.
//
//  TAILLE DE POINT CONSTANTE (point important du cahier des charges) :
//    Les particules ne changent JAMAIS de taille apparente — ni avec la
//    distance, ni avec l'audio : elles ne varient qu'en POSITION et COULEUR.
//    On force donc gl_PointSize a une constante en PIXELS (uniform u_point_size),
//    transmise telle quelle quelle que soit la profondeur. C'est volontairement
//    une taille ECRAN fixe (et non un diametre monde projete) : c'est ce qui
//    donne l'aspect "champ d'etoiles" homogene, et evite tout scintillement de
//    taille. On garde la PROJECTION PERSPECTIVE pour la position (parallaxe,
//    profondeur du nuage), seule la taille reste accrochee a l'ecran.
//
//    NB : sous core profile, gl_PointSize n'est pris en compte que si
//    GL_PROGRAM_POINT_SIZE est active cote contexte (fait dans renderer.py).
// =============================================================================

layout(location = 0) in vec4 in_pos;   // (x, y, z, brightness)
layout(location = 1) in vec4 in_col;    // (r, g, b, a) HDR lineaire

uniform mat4  u_view_proj;    // matrice view-projection (camera)
uniform float u_point_size;   // taille du point en PIXELS (constante)

// Donnees interpolees vers le fragment shader.
out vec4  v_color;       // couleur HDR de la particule
out float v_brightness;  // multiplicateur d'eclat (in_pos.w)

void main() {
    // Position monde -> espace clip via la matrice view-projection de la camera.
    // On n'utilise QUE xyz : le w du buffer porte la brightness, pas une
    // coordonnee homogene. On force donc w_clip = 1 en construisant vec4(xyz,1).
    gl_Position = u_view_proj * vec4(in_pos.xyz, 1.0);

    // Taille ECRAN fixe en pixels : indeplendante de la profondeur et de l'audio.
    // (cf. en-tete : seules position & couleur varient, jamais la taille.)
    gl_PointSize = u_point_size;

    // Transmission de la couleur HDR et de l'eclat par particule au fragment.
    v_color      = in_col;
    v_brightness = in_pos.w;
}
