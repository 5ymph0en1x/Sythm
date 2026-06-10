
extern "C" {

__device__ __forceinline__ unsigned int hash_u(unsigned int x){
    x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x;
}
__device__ __forceinline__ float hash_f(unsigned int x){
    return (hash_u(x)&0x00FFFFFFu)*(1.0f/16777216.0f);
}
__device__ __forceinline__ float hash3i(int xi,int yi,int zi){
    unsigned int h=(unsigned int)(xi*73856093)^(unsigned int)(yi*19349663)^(unsigned int)(zi*83492791);
    return hash_f(h);
}
__device__ __forceinline__ float fadef(float t){ return t*t*t*(t*(t*6.0f-15.0f)+10.0f); }
__device__ __forceinline__ float lerpf(float a,float b,float t){ return a+t*(b-a); }
__device__ float vnoise3(float x,float y,float z){
    int xi=(int)floorf(x),yi=(int)floorf(y),zi=(int)floorf(z);
    float xf=x-xi,yf=y-yi,zf=z-zi; float u=fadef(xf),v=fadef(yf),w=fadef(zf);
    float c000=hash3i(xi,yi,zi),c100=hash3i(xi+1,yi,zi),c010=hash3i(xi,yi+1,zi),c110=hash3i(xi+1,yi+1,zi);
    float c001=hash3i(xi,yi,zi+1),c101=hash3i(xi+1,yi,zi+1),c011=hash3i(xi,yi+1,zi+1),c111=hash3i(xi+1,yi+1,zi+1);
    float x00=lerpf(c000,c100,u),x10=lerpf(c010,c110,u),x01=lerpf(c001,c101,u),x11=lerpf(c011,c111,u);
    return lerpf(lerpf(x00,x10,v),lerpf(x01,x11,v),w);
}
__device__ void curl_noise(float x,float y,float z,float* out){
    const float e=0.4f, inv2e=1.0f/(2.0f*e);
    float p_y1=vnoise3(x,y+e,z)-vnoise3(x,y-e,z);
    float p_z1=vnoise3(x,y,z+e)-vnoise3(x,y,z-e);
    float ox=31.4f,oy=17.7f,oz=47.1f;
    float q_x1=vnoise3(x+e+ox,y+oy,z+oz)-vnoise3(x-e+ox,y+oy,z+oz);
    float q_z1=vnoise3(x+ox,y+oy,z+e+oz)-vnoise3(x+ox,y+oy,z-e+oz);
    float rx=-59.2f,ry=11.3f,rz=23.8f;
    float r_x1=vnoise3(x+e+rx,y+ry,z+rz)-vnoise3(x-e+rx,y+ry,z+rz);
    float r_y1=vnoise3(x+rx,y+e+ry,z+rz)-vnoise3(x+rx,y-e+ry,z+rz);
    out[0]=(r_y1-q_z1)*inv2e; out[1]=(p_z1-r_x1)*inv2e; out[2]=(q_x1-p_y1)*inv2e;
}
__device__ void hsv2rgb(float h,float s,float v,float* rgb){
    h=h-floorf(h); float i=floorf(h*6.0f); float f=h*6.0f-i;
    float p=v*(1.0f-s),q=v*(1.0f-f*s),t=v*(1.0f-(1.0f-f)*s); int ii=((int)i)%6; float r,g,b;
    switch(ii){case 0:r=v;g=t;b=p;break;case 1:r=q;g=v;b=p;break;case 2:r=p;g=v;b=t;break;
        case 3:r=p;g=q;b=v;break;case 4:r=t;g=p;b=v;break;default:r=v;g=p;b=q;break;}
    rgb[0]=r;rgb[1]=g;rgb[2]=b;
}

// ---------------------------------------------------------------------------
// MANDELBULB — formule « triplex » de White/Nylander, puissance n VARIABLE.
// Estimateur de distance (DE) classique : on itère z -> z^n + c en coordonnées
// sphériques en suivant |dz| (dr), puis DE = 0.5·ln(r)·r/dr. On en tire AUSSI :
//   * trap   : ORBIT TRAP (min |z| sur l'orbite) -> la COULEUR suit la structure
//              interne de la fractale (chaque lobe a sa teinte) ;
//   * inside : l'orbite n'a JAMAIS divergé -> point INTÉRIEUR à l'ensemble.
// OPTIMISATIONS : une SEULE __powf par itération (r^(n-1) = r^n / r),
// __sincosf fusionnés, sortie anticipée dès l'évasion (r>2), cos(theta) clampé
// (sécurité acosf). 9 itérations suffisent largement pour ATTIRER des particules
// (on ne raymarch pas au pixel près) tout en donnant un trap riche.
// ---------------------------------------------------------------------------
__device__ float bulb_de(const float x, const float y, const float z,
                         const float power, float* trap, int* inside)
{
    float zx=x, zy=y, zz=z;
    float dr=1.0f, r=0.0f, tr=1e9f;
    int esc=0;
    for(int i=0;i<9;i++){
        r=sqrtf(zx*zx+zy*zy+zz*zz);
        if(r>2.0f){ esc=1; break; }
        tr=fminf(tr,r);
        float rs=fmaxf(r,1e-9f);
        float ct=zz/rs; ct=fminf(1.0f,fmaxf(-1.0f,ct));
        float theta=acosf(ct)*power;
        float phi=atan2f(zy,zx)*power;
        float zr=__powf(rs,power);
        dr=(zr/rs)*power*dr+1.0f;          // r^(n-1)·n·dr + 1, sans 2e powf
        float st,ctn,sp,cp;
        __sincosf(theta,&st,&ctn);
        __sincosf(phi,&sp,&cp);
        zx=zr*st*cp+x; zy=zr*st*sp+y; zz=zr*ctn+z;
    }
    *inside=(esc==0);
    *trap=fminf(tr,2.0f);
    return 0.5f*logf(fmaxf(r,1e-9f))*r/dr;
}

// Échantillonnage interpolé du RELIEF TONAL (tableau 1D, u in [0,1] = rayon).
__device__ __forceinline__ float relief_at(const float* relief, int nrel, float u){
    if(u<0.0f)u=0.0f; if(u>1.0f)u=1.0f;
    float fi=u*(float)(nrel-1);
    int i0=(int)fi; if(i0<0)i0=0; if(i0>nrel-2)i0=nrel-2;
    float fr=fi-(float)i0;
    return relief[i0]*(1.0f-fr)+relief[i0+1]*fr;
}

// INIT origines : remplissage uniforme de la boîte [-L,L]^3.
__global__ void init_field(float* pos, const int n, const float L, const unsigned int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    unsigned int b=seed+(unsigned int)i*2654435761u;
    pos[i*3+0]=(hash_f(b+0u)*2.0f-1.0f)*L;
    pos[i*3+1]=(hash_f(b+1u)*2.0f-1.0f)*L;
    pos[i*3+2]=(hash_f(b+2u)*2.0f-1.0f)*L;
}

// Couleur RICHE pilotée par les BANDES DE FRÉQUENCES (sort du vert-jaune) :
//   grave = rouge (0), médium = vert (0.33), aigu = bleu (0.66) -> teinte spectrale
//   centrée sur la fréquence dominante, + étalement par particule/vitesse pour la
//   richesse spatiale. Saturation dopée par le beat et les aigus.
__device__ void spectral_color(
    float hue_id, float local, float t,
    float bass, float mid, float high, float beat, float centroid, float harm_hue, float val, float* rgb)
{
    float wsum = bass + mid + high + 0.02f;
    float spec = (mid * 0.33f + high * 0.66f) / wsum;   // 0 rouge, 0.33 vert, 0.66 bleu
    float hue = spec + (hue_id - 0.5f) * 0.30f + (local - 0.5f) * 0.20f + t * 0.01f;
    // TEMPÉRATURE DE TEINTE par le CENTROÏDE spectral (brillance timbrale) : timbre
    // brillant (centroid haut) -> teinte plus froide ; sourd (bas) -> plus chaude.
    hue += (centroid - 0.5f) * 0.12f;
    hue += harm_hue;            // HARMONIE : modalité (chaud/froid) + teinte-maison de tonalité
    float sat = fminf(0.97f, 0.75f + 0.20f * beat + 0.12f * high);
    hsv2rgb(hue, sat, val, rgb);
}

// UPDATE origines : advection champ ABC(Lorenz) + ONDES DE CHOC + PAYSAGE TONAL
//                   + MÉMORISE la vitesse (les traînées héritent du geste) + GL.
__global__ void update_origin(
    float* pos, float* vel, float* gl_pos, float* gl_col,
    const int n, const float t, const float dt, const float L,
    const float lx, const float ly, const float lz,
    const float field_strength, const float k, const float turb_base,
    const float amp, const float beat, const float centroid,
    const float bass, const float mid, const float high,
    const float* wpos, const float* wpar, const int n_waves,
    const float* relief, const int nrel,
    const float tonal_strength, const float tonal_cap, const float tonal_glow,
    const float breath, const float accel_gain, const float accel_inv_scale,
    const float build, const float drop, const float harm_hue,
    const float grav_amp, const float grav_k, const float grav_phase,
    const float tunnel, const float tunnel_speed, const float tunnel_radius,
    const float tunnel_wall, const float tunnel_ax, const float tunnel_phx,
    const float tunnel_ay, const float tunnel_phy, const float tunnel_swirl,
    const float bulb, const float bulb_scale, const float bulb_power,
    const float bulb_reseed)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float px=pos[i*3+0],py=pos[i*3+1],pz=pos[i*3+2];
    float ovx=vel[i*3+0], ovy=vel[i*3+1], ovz=vel[i*3+2];   // vitesse frame n-1 (avant écrasement)

    // ----- MANDELBULB : RÉ-ENSEMENCEMENT continu (pluie d'accrétion). Le flot
    // tangentiel projeté sur la coquille n'est pas à divergence nulle : laissée à
    // elle-même, la matière s'AGGLUTINE lentement en nœuds. Plutôt que de lutter,
    // on RENOUVELLE : chaque frame, une petite fraction (bulb_reseed = dt/τ, τ≈10 s)
    // renaît sur une coquille sphérique au large du bulbe et RETOMBE dessus — la
    // couverture reste uniforme par construction (un nœud n'a jamais le temps de
    // croître), et la pluie de comètes vers la fractale fait partie du tableau.
    if(bulb>0.0f && bulb_reseed>0.0f){
        unsigned int sd=(unsigned int)(t*997.0f);
        unsigned int b2=(unsigned int)i*2654435761u ^ (sd*2246822519u);
        if(hash_f(b2)<bulb_reseed){
            float u1=hash_f(b2+1u), u2=hash_f(b2+2u), u3=hash_f(b2+3u);
            float ct=u1*2.0f-1.0f;                       // direction uniforme (sphère)
            float st=sqrtf(fmaxf(0.0f,1.0f-ct*ct));
            float ph=u2*6.2831853f;
            float rad=(1.25f+0.45f*u3)*1.2f*bulb_scale;  // au large de la surface (~1.2×scale)
            float sp,cp; __sincosf(ph,&sp,&cp);
            px=st*cp*rad; py=st*sp*rad; pz=ct*rad;
            ovx=0.0f; ovy=0.0f; ovz=0.0f;                // vitesse n-1 neutre (pas de faux éclair)
        }
    }
    float kx=px*k,ky=py*k,kz=pz*k;
    float fx=lx*sinf(kz)+lz*cosf(ky);
    float fy=ly*sinf(kx)+lx*cosf(kz);
    float fz=lz*sinf(ky)+ly*cosf(kx);
    // MOUVEMENT calé sur les BASSES : les graves accélèrent le flux + la turbulence.
    float spd=field_strength*(1.0f+0.5f*amp+2.0f*bass)*(1.0f+0.8f*build);  // le build accélère le flux
    float c[3]; float ns=0.6f;
    curl_noise(px*ns+t*0.2f, py*ns, pz*ns-t*0.2f, c);
    float turb=turb_base*(1.0f+4.0f*beat+2.5f*bass);
    // Vitesse de déplacement immédiat (sert AUSSI à lancer les particules émises).
    float tvx=fx*spd + c[0]*turb;
    float tvy=fy*spd + c[1]*turb;
    float tvz=fz*spd + c[2]*turb;

    // ----- TUNNEL : AXE COURBE du tube (le serpent). Le tunnel n'est pas un
    // cylindre droit : son axe SERPENTE — sinusoïdes en Z à périodes ENTIÈRES de
    // la boîte (kc = π/L -> enroulement périodique SANS couture), amplitudes et
    // phases pilotées côté Python par l'état LISSÉ du Lorenz caché : l'attracteur
    // tient le manche des virages. Calculé UNE fois ici ; la paroi, les anneaux
    // rythmiques et la teinte azimutale s'y réfèrent tous.
    float axx=0.0f, axy=0.0f;
    if(tunnel>0.0f){
        const float kc=3.14159265f/L;
        axx=tunnel_ax*__sinf(kc*pz+tunnel_phx);
        axy=tunnel_ay*__sinf(2.0f*kc*pz+tunnel_phy);
    }

    // ----- ONDES DE CHOC PERCUSSIVES : coques sphériques qui TRAVERSENT -----
    // Là où la coque gaussienne d'une onde passe (r ≈ vitesse·âge), on ajoute une
    // POUSSÉE radiale (kick), un CISAILLEMENT tangentiel (snare) et un ÉCLAT
    // (charley). Le rythme devient une météo qu'on VOIT voyager dans la matière.
    // EN TUNNEL, le front devient un ANNEAU AXIAL : un plan-z (périodique) qui
    // balaie le tube vers la caméra — kick = anneau épais qui GONFLE la paroi et
    // EMPORTE la matière, snare = anneau qui TORD (tangentiel), hat = anneau fin
    // qui SCINTILLE. Le rythme se lit comme des portes de lumière qu'on franchit.
    float wave_bright=0.0f;
    for(int w=0; w<n_waves; ++w){
        float st=wpar[w*6+0];               // force (déjà fondue dans le temps)
        if(st<1e-4f) continue;              // onde éteinte -> on saute
        float ex=wpos[w*3+0], ey=wpos[w*3+1], ez=wpos[w*3+2];
        float radius=wpar[w*6+1];           // rayon courant du front
        float thick=wpar[w*6+2];
        float push=wpar[w*6+3];
        float curl=wpar[w*6+4];
        if(tunnel>0.0f){
            // Anneau axial : le front part de l'épicentre (z du Lorenz) et VOLE
            // vers +Z (la caméra), distance z repliée (tunnel périodique).
            float Wz=2.0f*L;
            float dz=pz-(ez+radius);
            dz-=Wz*floorf(dz/Wz+0.5f);      // plus courte distance en z (mod 2L)
            float s=dz/thick;
            float a=st*__expf(-s*s);
            if(a<1e-4f) continue;
            float rx=px-axx, ry=py-axy;
            float rr=sqrtf(rx*rx+ry*ry)+1e-4f;
            float irr=1.0f/rr;
            float nx=rx*irr, ny=ry*irr;
            tvx+=nx*push*a*0.8f; tvy+=ny*push*a*0.8f;   // l'anneau GONFLE la paroi
            tvz+=push*a*0.6f;                           // ...et EMPORTE la matière
            if(curl!=0.0f){ tvx+=-ny*curl*a; tvy+=nx*curl*a; }   // anneau qui TORD
            wave_bright+=wpar[w*6+5]*a;
            continue;
        }
        float dx=px-ex, dy=py-ey, dz=pz-ez;
        float r=sqrtf(dx*dx+dy*dy+dz*dz)+1e-4f;
        float s=(r-radius)/thick;
        float shell=__expf(-s*s);           // coque gaussienne (1 sur le front)
        float a=st*shell;
        if(a<1e-4f) continue;
        float inv=1.0f/r;
        float nx=dx*inv, ny=dy*inv, nz=dz*inv;
        tvx+=nx*push*a; tvy+=ny*push*a; tvz+=nz*push*a;   // poussée radiale
        if(curl!=0.0f){                     // tournoiement (cisaillement autour de Y)
            float ttx=-nz, ttz=nx;          // = cross(n, up=(0,1,0)) horizontal
            float tl=rsqrtf(ttx*ttx+ttz*ttz+1e-8f);
            tvx+=ttx*tl*curl*a; tvz+=ttz*tl*curl*a;
        }
        wave_bright+=wpar[w*6+5]*a;         // éclat lumineux du front
    }

    // ----- PAYSAGE TONAL : relief radial stable (les notes TENUES le sculptent)
    // On remonte DOUCEMENT le gradient du relief -> striations concentriques
    // (graves au cœur, aigus en périphérie). Fenêtré + plafonné pour préserver le
    // remplissage de l'espace ; l'advection continue de brasser -> jamais figé.
    float prad=sqrtf(px*px+py*py+pz*pz)+1e-4f;
    float ir=1.0f/prad;
    float u=prad/L; if(u>1.0f)u=1.0f;
    float tonal_here=relief_at(relief, nrel, u);
    if(tonal_strength>0.0f){
        float du=2.0f/(float)nrel;
        float g=relief_at(relief,nrel,u+du)-relief_at(relief,nrel,u-du);
        float win=u*(1.0f-u)*4.0f;          // 0 aux extrêmes, 1 au milieu
        float tf=tonal_strength*g*win;
        if(tf>tonal_cap)tf=tonal_cap; else if(tf<-tonal_cap)tf=-tonal_cap;
        tvx+=px*ir*tf; tvy+=py*ir*tf; tvz+=pz*ir*tf;
    }

    // ----- RESPIRATION (pouls anticipé) : inspir AVANT le temps fort (vers le
    // cœur), expir SUR le beat (vers l'extérieur). breath<0 -> converge, >0 ->
    // s'épanouit. Transitoire et oscillant -> pas d'amas ; porté par le groove.
    tvx+=px*ir*breath; tvy+=py*ir*breath; tvz+=pz*ir*breath;

    // ----- ONDES GRAVITATIONNELLES : tout le champ ONDULE sous la BASSE PROFONDE.
    // Front radial qui VOYAGE (crête là où grav_k·r = grav_phase, se propage vers
    // l'extérieur) -> on le VOIT parcourir la matière. Amplitude ∝ sub (force
    // ressentie) ; plus la basse est deep, plus grav_k est petit -> longueur d'onde
    // grande -> l'impact devient GÉNÉRAL (toute la matière bouge ensemble). Léger
    // bruit de phase -> fronts ORGANIQUES (pas une sphère parfaite). La vitesse étant
    // mémorisée plus bas, les traînées héritent du soubresaut -> ondes dans les traînées.
    if(grav_amp>1e-5f){
        float gph=grav_k*prad - grav_phase
                  + (vnoise3(px*0.5f+11.0f, py*0.5f, pz*0.5f-7.0f)-0.5f)*1.3f;
        float gw=__sinf(gph);
        tvx+=px*ir*grav_amp*gw; tvy+=py*ir*grav_amp*gw; tvz+=pz*ir*grav_amp*gw;
        wave_bright+=grav_amp*fmaxf(gw,0.0f)*0.9f;   // les fronts BRILLENT en passant
    }

    // ----- TUNNEL HYPERSPACE : on FONCE le long de l'axe Z (dérive axiale) et la
    // matière se met en forme de PAROI CYLINDRIQUE autour de l'AXE COURBE (axx,axy
    // calculés plus haut : le tube SERPENTE, conduit par le Lorenz caché lissé).
    // Le champ ABC chaotique devient la turbulence ORGANIQUE des parois ;
    // l'enroulement périodique en Z (plus bas) rend le tunnel INFINI (ce qui sort
    // au fond ré-entre devant). tunnel_wall règle la netteté : 0 = volumétrique
    // flou (cœur pas vide), grand = paroi nette + cœur creux. La VRILLE est SIGNÉE
    // (tunnel_swirl, conduite elle aussi par le Lorenz lissé -> elle s'inverse
    // organiquement) et s'emballe sur le build. La vitesse étant mémorisée plus
    // bas, les TRAÎNÉES filent le long de l'axe = stries de vitesse.
    if(tunnel>0.0f){
        tvz += tunnel_speed;                              // le vol
        float rx = px-axx, ry = py-axy;                   // position vs axe COURBE
        float rxy = sqrtf(rx*rx+ry*ry)+1e-4f;             // distance à l'axe local
        float inv = 1.0f/rxy;
        float pull = tunnel_wall*(tunnel_radius - rxy);   // rappel vers la paroi
        tvx += rx*inv*pull;  tvy += ry*inv*pull;
        tvx += -ry*inv*tunnel_speed*tunnel_swirl;         // vrille (signée, musicale)
        tvy +=  rx*inv*tunnel_speed*tunnel_swirl;
    }

    // ----- MANDELBULB : la nuée se CONDENSE sur la surface de la fractale.
    // Chaque particule évalue le champ de distance (DE) du bulbe et trois forces
    // la sculptent : (1) un RESSORT proportionnel au DE la tire vers la surface
    // (clampé -> approche exponentielle, AUCUNE oscillation) ; (2) près de la
    // surface, la composante NORMALE du flot s'efface -> la matière GLISSE LE
    // LONG de la fractale (le champ ABC devient le courant qui en révèle la
    // forme) ; (3) une particule INTÉRIEURE est poussée dehors, braise remontant
    // vers la coquille. La COULEUR suit l'orbit trap (chaque lobe sa teinte), la
    // surface LUIT par proximité. bulb_power MUTE avec la musique côté Python ->
    // la fractale se MÉTAMORPHOSE en continu. OPTIMISATION : loin de la surface
    // (de>0.45), le gradient est remplacé par la direction radiale -> 3 évals DE
    // économisées pendant la phase de condensation.
    float bulb_trap=-1.0f, bulb_prox=0.0f;
    if(bulb>0.0f){
        float inv_s=1.0f/bulb_scale;
        float tr_; int ins;
        float de=bulb_de(px*inv_s, py*inv_s, pz*inv_s, bulb_power, &tr_, &ins);
        bulb_trap=fminf(tr_,1.2f)*0.8333f;            // trap -> teinte (0..1)
        if(ins){
            float esc_v=2.5f*bulb;                    // braise : sortie radiale
            tvx+=px*ir*esc_v; tvy+=py*ir*esc_v; tvz+=pz*ir*esc_v;
            bulb_prox=0.12f;                          // lueur intérieure discrète
        } else {
            float de_w=de*bulb_scale;                 // DE en unités MONDE
            float gx,gy,gz;
            if(de>0.45f){                             // LOIN : direction radiale
                gx=px*ir; gy=py*ir; gz=pz*ir;
            } else {                                  // PRÈS : vrai gradient du DE
                const float e=0.002f;
                float t2; int i2;
                gx=bulb_de(px*inv_s+e,py*inv_s,pz*inv_s,bulb_power,&t2,&i2)-de;
                gy=bulb_de(px*inv_s,py*inv_s+e,pz*inv_s,bulb_power,&t2,&i2)-de;
                gz=bulb_de(px*inv_s,py*inv_s,pz*inv_s+e,bulb_power,&t2,&i2)-de;
                float gl=rsqrtf(gx*gx+gy*gy+gz*gz+1e-12f);
                gx*=gl; gy*=gl; gz*=gl;
            }
            float w=0.06f*bulb_scale;                 // largeur de la couche de glisse
            float surf=__expf(-(de_w*de_w)/(w*w));
            float vn=tvx*gx+tvy*gy+tvz*gz;            // GLISSE : flot tangentiel
            tvx-=gx*vn*0.55f*surf; tvy-=gy*vn*0.55f*surf; tvz-=gz*vn*0.55f*surf;
            // BRASSAGE : le flot tangentiel converge par endroits (sa projection
            // sur la coquille n'est plus à divergence nulle) -> sans contre-feu,
            // la matière s'AGGLUTINE en nœuds. On redouble le curl noise PRÈS de
            // la surface : le revêtement reste homogène, la fractale lisible.
            tvx+=c[0]*turb*1.5f*surf; tvy+=c[1]*turb*1.5f*surf; tvz+=c[2]*turb*1.5f*surf;
            float pull=fminf(de_w,1.4f)*7.0f*bulb;    // RESSORT vers la surface
            tvx-=gx*pull; tvy-=gy*pull; tvz-=gz*pull;
            bulb_prox=surf;                           // lueur de surface
        }
    }

    px+=tvx*dt; py+=tvy*dt; pz+=tvz*dt;
    float W=2.0f*L;
    if(px>L)px-=W; else if(px<-L)px+=W;
    if(py>L)py-=W; else if(py<-L)py+=W;
    if(pz>L)pz-=W; else if(pz<-L)pz+=W;
    pos[i*3+0]=px; pos[i*3+1]=py; pos[i*3+2]=pz;
    vel[i*3+0]=tvx; vel[i*3+1]=tvy; vel[i*3+2]=tvz;
    // couleur RICHE pilotée par les bandes de fréquences (cf. spectral_color).
    float vlen=sqrtf(tvx*tvx+tvy*tvy+tvz*tvz)+1e-4f;
    float local=tvz/vlen*0.5f+0.5f;
    if(tunnel>0.0f){
        // Teinte AZIMUTALE : en tunnel, tout vole en +Z (local≈cste). On la
        // remplace par l'angle AUTOUR de l'axe courbe -> la couleur tourne avec
        // la vrille, le vortex devient lisible.
        local=atan2f(py-axy, px-axx)*0.15915494f+0.5f;   // 1/(2π)
    }
    if(bulb>0.0f && bulb_trap>=0.0f){
        // Teinte par ORBIT TRAP : la couleur suit la structure INTERNE de la
        // fractale -> chaque lobe, chaque vallée du bulbe porte sa propre teinte.
        local=bulb_trap;
    }
    // basses -> + lumineux ; + ÉCLAT des fronts d'onde ; + LUEUR des strates tonales.
    float val=0.50f+amp*0.25f+beat*0.30f+bass*0.30f
              + wave_bright*0.6f + tonal_glow*tonal_here
              + 0.30f*bulb_prox;   // MANDELBULB : la surface de la fractale LUIT
    val=val*(1.0f-0.35f*build) + 0.80f*drop;   // build ASSOMBRIT (charge), drop FLASHE
    val=fminf(val, 1.6f);
    if(tunnel>0.0f){
        // BRUME DE PROFONDEUR : sombre au fond (zu=0), pleine lumière à la caméra
        // (zu=1). Donne l'échelle du tube ET masque la couture du ré-enroulement
        // (une particule renaît au fond DANS le noir, jamais en plein éclat).
        float zu=(pz+L)/(2.0f*L);
        val*=0.55f+0.45f*zu;
    }
    float rgb[3];
    spectral_color(hash_f((unsigned int)i), local, t, bass, mid, high, beat, centroid, harm_hue, val, rgb);
    // ÉTINCELLE DE CISAILLEMENT : |a| = Dv/Dt (différence finie vs vitesse n-1),
    // compressée par tanh -> brille aux nœuds violents du flot ET au passage des fronts.
    float ax=(tvx-ovx)/dt, ay=(tvy-ovy)/dt, az=(tvz-ovz)/dt;
    float ahat=tanhf(sqrtf(ax*ax+ay*ay+az*az)*accel_inv_scale);
    float brightness=0.55f + wave_bright*0.5f + accel_gain*ahat + 0.60f*drop;   // fronts + cisaillement + FLASH du drop
    gl_pos[i*4+0]=px; gl_pos[i*4+1]=py; gl_pos[i*4+2]=pz; gl_pos[i*4+3]=brightness;
    gl_col[i*4+0]=rgb[0]; gl_col[i*4+1]=rgb[1]; gl_col[i*4+2]=rgb[2]; gl_col[i*4+3]=1.0f;
}

// ÉMISSION : E particules réécrites dans le ring depuis des origines.
//   es[slot*7+0..2]=pos, +3..5=vel (déplacement immédiat de l'origine), +6=age(0).
__global__ void emit_particles(
    const float* opos, const float* ovel, float* es,
    const int E, const int n_origin, const int n_emit, const int head)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=E) return;
    int slot=(head+tid)%n_emit;
    int src=(head+tid)%n_origin;             // balaie les origines uniformément
    es[slot*7+0]=opos[src*3+0]; es[slot*7+1]=opos[src*3+1]; es[slot*7+2]=opos[src*3+2];
    es[slot*7+3]=ovel[src*3+0]; es[slot*7+4]=ovel[src*3+1]; es[slot*7+5]=ovel[src*3+2];
    es[slot*7+6]=0.0f;
}

// UPDATE émises : intégration balistique + fondu par l'âge + écrit GL (offset n_origin).
// EN TUNNEL (tunnel>0) : les traînées s'ENROULENT dans la boîte périodique comme
// les origines. Sans ça, lancées à la vitesse du vol (~10-30 u/s), elles quittaient
// la boîte en <1 s et passaient ~90 % de leur vie INVISIBLES hors du tube — on
// rendait des dizaines de millions de points pour rien. Enroulées, TOUTES les
// stries restent dans le tunnel -> densité multipliée sans une particule de plus.
__global__ void update_emitted(
    float* es, float* gl_pos, float* gl_col,
    const int n_emit, const int n_origin, const float dt,
    const float lifetime, const float emit_bright, const float centroid,
    const float t, const float bass, const float mid, const float high, const float beat,
    const float harm_hue, const float L, const float tunnel)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n_emit) return;
    float px=es[i*7+0],py=es[i*7+1],pz=es[i*7+2];
    float vx=es[i*7+3],vy=es[i*7+4],vz=es[i*7+5];
    float age=es[i*7+6]+dt;
    // MOUVEMENT calé sur les BASSES : les graves font surgir les traînées.
    float em_speed=1.0f+1.2f*bass;
    px+=vx*dt*em_speed; py+=vy*dt*em_speed; pz+=vz*dt*em_speed;
    if(tunnel>0.0f){
        float W=2.0f*L;
        if(px>L)px-=W; else if(px<-L)px+=W;
        if(py>L)py-=W; else if(py<-L)py+=W;
        if(pz>L)pz-=W; else if(pz<-L)pz+=W;
    }
    es[i*7+0]=px; es[i*7+1]=py; es[i*7+2]=pz; es[i*7+6]=age;

    int gi=n_origin+i;
    float bright=0.0f, alpha=0.0f; float rgb[3]={0.0f,0.0f,0.0f};
    if (age < lifetime) {
        float u = age / lifetime;                // 0 -> 1
        // Enveloppe DOUCE : court fondu d'apparition + extinction en (1-u)^2
        // (valeur ET pente nulles à la mort) -> AUCUNE coupure franche.
        float fin = fminf(u * 12.5f, 1.0f);
        float fout = 1.0f - u; fout = fout * fout;
        float env = fin * fout;
        float vlen=sqrtf(vx*vx+vy*vy+vz*vz)+1e-4f;
        float local=vz/vlen*0.5f+0.5f;
        spectral_color(hash_f((unsigned int)i), local, t, bass, mid, high, beat, centroid, harm_hue, 0.75f, rgb);
        bright=emit_bright*env;                  // apparition + extinction douces
        if(tunnel>0.0f){
            // Même BRUME DE PROFONDEUR que les origines (cf. update_origin).
            float zu=(pz+L)/(2.0f*L);
            bright*=0.35f+0.65f*zu;
        }
        alpha=1.0f;
    }
    gl_pos[gi*4+0]=px; gl_pos[gi*4+1]=py; gl_pos[gi*4+2]=pz; gl_pos[gi*4+3]=bright;
    gl_col[gi*4+0]=rgb[0]; gl_col[gi*4+1]=rgb[1]; gl_col[gi*4+2]=rgb[2]; gl_col[gi*4+3]=alpha;
}

// PRÉ-REMPLISSAGE one-shot du ring : âge réparti sur [0,lifetime) + position
//   balistique, écrits EN PLACE (densité de régime dès la 1re frame). 1 thread par
//   slot ; origines lues en mémoire GLOBALE (elles font des Mo -> hors de portée des
//   64 Ko de constant memory). AUCUN tableau temporaire (cf. prefill_emitted Python,
//   qui remplaçait une version vectorisée tombant en OOM sur les presets longs).
__global__ void prefill_emitted(
    const float* opos, const float* ovel, float* es,
    const int n_emit, const int n_origin, const float lifetime,
    const float L, const float tunnel)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n_emit) return;
    int src=i%n_origin;                                          // même balayage qu'emit_particles
    float age=hash_f((unsigned int)i ^ 0x9E3779B9u)*lifetime;    // âge réparti [0,lifetime)
    float ox=opos[src*3+0], oy=opos[src*3+1], oz=opos[src*3+2];
    float vx=ovel[src*3+0], vy=ovel[src*3+1], vz=ovel[src*3+2];
    float ex=ox+vx*age, ey=oy+vy*age, ez=oz+vz*age;              // position balistique à cet âge
    if(tunnel>0.0f){
        // En tunnel les émises s'enroulent (cf. update_emitted) : on replie la
        // position balistique mod 2L (v·age peut traverser PLUSIEURS boîtes) ->
        // le pré-remplissage tombe pile sur le régime établi.
        float W=2.0f*L;
        ex-=W*floorf((ex+L)/W); ey-=W*floorf((ey+L)/W); ez-=W*floorf((ez+L)/W);
    }
    es[i*7+0]=ex; es[i*7+1]=ey; es[i*7+2]=ez;
    es[i*7+3]=vx; es[i*7+4]=vy; es[i*7+5]=vz;                        // vitesse héritée de l'origine
    es[i*7+6]=age;                                                   // âge -> densité de régime
}

}  // extern "C"
