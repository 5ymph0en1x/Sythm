#version 460 core
// =============================================================================
//  fullscreen.vert
//  -------------------------------------------------------------------------
//  Vertex shader "triangle plein ecran" SANS VBO ni attribut.
//
//  PRINCIPE (technique du "fullscreen triangle") :
//    On dessine UN SEUL triangle dont les 3 sommets debordent largement du
//    cube de clip [-1,1]. L'intersection de ce grand triangle avec le viewport
//    couvre exactement tout l'ecran. C'est plus efficace qu'un quad (2 tris)
//    car il n'y a aucun bord diagonal interne ou les deux triangles se
//    chevauchent (pas de "double shading" sur la diagonale), et surtout aucun
//    VBO/VAO de geometrie n'est necessaire : on genere tout depuis gl_VertexID.
//
//  APPEL COTE PYTHON :
//    vao = ctx.vertex_array(program, [])   # VAO vide, aucun buffer
//    vao.render(mode=moderngl.TRIANGLES, vertices=3)
//
//  Les 3 sommets generes (en coordonnees de clip) :
//    id=0 -> (-1,-1)   uv (0,0)
//    id=1 -> ( 3,-1)   uv (2,0)
//    id=2 -> (-1, 3)   uv (0,2)
//  Le triangle (-1,-1)/(3,-1)/(-1,3) recouvre tout [-1,1]^2.
//
//  On exporte 'v_uv' dans [0,1] sur la zone ecran (peut depasser 1 hors ecran,
//  mais ces fragments sont clippes et jamais rasterises).
// =============================================================================

out vec2 v_uv;

void main() {
    // Genere 0,2 puis 0 sur x ; 0 puis 2 sur y selon l'identifiant du sommet.
    // (gl_VertexID & 2) -> 0 ou 2  ;  ((gl_VertexID << 1) & 2) -> 0 ou 2.
    v_uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    // Passage de [0,2] (uv) a [-1,3] (clip) : uv*2-1.
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
