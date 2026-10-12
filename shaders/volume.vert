#version 450
layout(push_constant) uniform Camera { vec4 right; vec4 up; vec4 forward; vec4 origin; } camera;
layout(location=0) out vec3 dir;
layout(location=1) out vec3 localdir;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
    // Vulkan positive-height viewport is top-down; keep the original WebGL orientation.
    localdir = vec3(p.x, -p.y, -1.0);
    dir = camera.forward.xyz + camera.right.xyz * p.x - camera.up.xyz * p.y;
}
