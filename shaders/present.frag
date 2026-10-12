#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform sampler2D scene;
layout(push_constant) uniform Output { vec2 extent; } outputSize;
void main() {
    // A full-surface source (AI path) passes extent=(1,1) to avoid applying the letterbox twice.
    vec2 p = (uv - 0.5) * outputSize.extent / min(outputSize.extent.x, outputSize.extent.y) + 0.5;
    if (any(lessThan(p, vec2(0.0))) || any(greaterThan(p, vec2(1.0)))) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        color = texture(scene, p);
    }
}
