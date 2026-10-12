#version 450
#extension GL_GOOGLE_include_directive : require
#define PI 3.14159265358979324
#define M_L 0.3819660113
#define M_R 0.6180339887
#define MAXR 8
#define SOLVER 8
float kernel(vec3 ver)
;
layout(push_constant) uniform Camera { vec4 cameraRight; vec4 cameraUp; vec4 cameraForward; vec4 cameraOrigin; } camera;
#define right camera.cameraRight.xyz
#define up camera.cameraUp.xyz
#define forward camera.cameraForward.xyz
#define origin camera.cameraOrigin.xyz
#define len camera.cameraOrigin.w
layout(location=0) out vec4 outColor;
layout(location=0) in vec3 dir;
layout(location=1) in vec3 localdir;
vec3 ver;
int hitSign;
float v, v1, v2;
float r1, r2, r3, r4, m1, m2, m3, m4;
vec3 n, reflection;
const float step=0.002;
vec3 color;
void main() {
   color.r=0.0;
   color.g=0.0;
   color.b=0.0;
   hitSign=0;
   v1=kernel(origin + dir * (step*len));
   v2=kernel(origin);
   for (int k=2; k < 1002; k++) {
      ver=origin + dir * (step*len*float(k));
      v=kernel(ver);
      if (v > 0.0 && v1 < 0.0) {
         r1=step * len*float(k - 1);
         r2=step * len*float(k);
         m1=kernel(origin + dir * r1);
         m2=kernel(origin + dir * r2);
         for (int l=0; l < SOLVER; l++) {
            r3=r1 * 0.5 + r2 * 0.5;
            m3=kernel(origin + dir * r3);
            if (m3 > 0.0) {
               r2=r3;
               m2=m3;
            }
            else {
               r1=r3;
               m1=m3;
            }
         }
         if (r3 < 2.0 * len) {
               hitSign=1;
            break;
         }
      }
      if (v < v1&&v1>v2&&v1 < 0.0 && (v1*2.0 > v || v1 * 2.0 > v2)) {
         r1=step * len*float(k - 2);
         r2=step * len*(float(k) - 2.0 + 2.0*M_L);
         r3=step * len*(float(k) - 2.0 + 2.0*M_R);
         r4=step * len*float(k);
         m2=kernel(origin + dir * r2);
         m3=kernel(origin + dir * r3);
         for (int l=0; l < MAXR; l++) {
            if (m2 > m3) {
               r4=r3;
               r3=r2;
               r2=r4 * M_L + r1 * M_R;
               m3=m2;
               m2=kernel(origin + dir * r2);
            }
            else {
               r1=r2;
               r2=r3;
               r3=r4 * M_R + r1 * M_L;
               m2=m3;
               m3=kernel(origin + dir * r3);
            }
         }
         if (m2 > 0.0) {
            r1=step * len*float(k - 2);
            r2=r2;
            m1=kernel(origin + dir * r1);
            m2=kernel(origin + dir * r2);
            for (int l=0; l < SOLVER; l++) {
               r3=r1 * 0.5 + r2 * 0.5;
               m3=kernel(origin + dir * r3);
               if (m3 > 0.0) {
                  r2=r3;
                  m2=m3;
               }
               else {
                  r1=r3;
                  m1=m3;
               }
            }
            if (r3 < 2.0 * len&&r3> step*len) {
                   hitSign=1;
               break;
            }
         }
         else if (m3 > 0.0) {
            r1=step * len*float(k - 2);
            r2=r3;
            m1=kernel(origin + dir * r1);
            m2=kernel(origin + dir * r2);
            for (int l=0; l < SOLVER; l++) {
               r3=r1 * 0.5 + r2 * 0.5;
               m3=kernel(origin + dir * r3);
               if (m3 > 0.0) {
                  r2=r3;
                  m2=m3;
               }
               else {
                  r1=r3;
                  m1=m3;
               }
            }
            if (r3 < 2.0 * len&&r3> step*len) {
                   hitSign=1;
               break;
            }
         }
      }
      v2=v1;
      v1=v;
   }
   if (hitSign==1) {
      ver=origin + dir*r3 ;
       r1=ver.x*ver.x+ver.y*ver.y+ver.z*ver.z;
      n.x=kernel(ver - right * (r3*0.00025)) - kernel(ver + right * (r3*0.00025));
      n.y=kernel(ver - up * (r3*0.00025)) - kernel(ver + up * (r3*0.00025));
      n.z=kernel(ver + forward * (r3*0.00025)) - kernel(ver - forward * (r3*0.00025));
      r3=n.x*n.x+n.y*n.y+n.z*n.z;
      n=n * (1.0 / sqrt(r3));
      ver=localdir;
      r3=ver.x*ver.x+ver.y*ver.y+ver.z*ver.z;
      ver=ver * (1.0 / sqrt(r3));
      reflection=n * (-2.0*dot(ver, n)) + ver;
      r3=reflection.x*0.276+reflection.y*0.920+reflection.z*0.276;
      r4=n.x*0.276+n.y*0.920+n.z*0.276;
      r3=max(0.0,r3);
      r3=r3 * r3*r3*r3;
      r3=r3 * 0.45 + r4 * 0.25 + 0.3;
      n.x=sin(r1*10.0)*0.5+0.5;
      n.y=sin(r1*10.0+2.05)*0.5+0.5;
      n.z=sin(r1*10.0-2.05)*0.5+0.5;
      color=n*r3;
   }
   outColor=vec4(color.x, color.y, color.z, 1.0);
}
#include "kernel.glsl"
