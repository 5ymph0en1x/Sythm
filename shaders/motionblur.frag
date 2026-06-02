#version 460 core
// =============================================================================
//  motionblur.frag
//  -------------------------------------------------------------------------
//  Motion blur par BUFFER D'HISTORIQUE (temporal trail / persistence).
//
//  PRINCIPE :
//    On melange lineairement (lerp) la frame HDR courante avec la frame
//    accumulee a l'image precedente. Le resultat est re-stocke et devient
//    l'historique de la frame suivante (ping-pong de FBO cote Python).
//
//        accum = lerp(courante, historique, strength)
//              = courante * (1 - strength) + historique * strength
//
//    - strength = 0.0  -> aucune trainee (sortie = frame courante).
//    - strength -> 1.0  -> trainees tres longues et persistantes.
//    Comme le nuage de particules est rendu en ADDITIF (valeurs > 1 frequentes),
//    ce simple feedback produit de superbes trainees lumineuses qui s'estompent
//    exponentiellement : chaque frame conserve 'strength' de l'energie passee.
//
//  PREMIERE FRAME :
//    L'historique n'existe pas encore -> on force strength=0 cote Python via
//    l'uniform u_first_frame (la sortie = pur courant, pas de pollution).
//
//  ALTERNATIVE (commentee, plus couteuse) : VELOCITY BUFFER.
//    Approche "physiquement correcte" : on rend un G-buffer de VITESSE ecran
//    (deplacement de chaque pixel entre frame N-1 et N, obtenu via les matrices
//    MVP precedente/courante ou la vitesse des particules projetee). Puis ce
//    shader echantillonnerait la couleur le long du vecteur vitesse :
//        vec2 vel = texture(u_velocity, uv).xy * u_strength;
//        for (int i=0;i<NB;i++) {
//            color += texture(u_scene, uv - vel * (float(i)/NB - 0.5));
//        }
//        color /= NB;
//    Avantage : flou directionnel exact par objet, pas de "fantome" global.
//    Inconvenient : exige un attribut/buffer de vitesse depuis le particle
//    system, une passe G-buffer supplementaire et NB taps par pixel -> plus
//    cher. Pour un nuage purement additif, le buffer d'historique donne un
//    rendu cinematographique tout aussi convaincant pour une fraction du cout.
// =============================================================================

in  vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;       // frame HDR courante (RGBA16F)
uniform sampler2D u_history;     // frame accumulee precedente (RGBA16F)
uniform float     u_strength;    // motion_blur_strength dans [0,1[
uniform int       u_first_frame; // 1 a la toute premiere image, 0 ensuite

void main() {
    vec4 courante = texture(u_scene, v_uv);

    // Premiere frame : pas d'historique valide -> on renvoie directement la
    // scene pour ne pas melanger du contenu indefini (noir ou poubelle GPU).
    if (u_first_frame == 1) {
        f_color = courante;
        return;
    }

    vec4 historique = texture(u_history, v_uv);

    // Melange exponentiel : la composante 'historique' s'attenue de strength
    // a chaque frame -> trainee qui decroit en douceur.
    f_color = mix(courante, historique, u_strength);
}
