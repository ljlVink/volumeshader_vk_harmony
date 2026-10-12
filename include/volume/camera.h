#pragma once
#include "volume/renderer.h"
#include <algorithm>
#include <cmath>

namespace volume {
class Camera {
public:
    bool autoRotate = true;
    void reset() { distance_ = 1.6f; azimuth_ = 2.8f; elevation_ = 0.4f; center_ = {}; }
    void advance(double seconds) { if (autoRotate) azimuth_ += static_cast<float>(seconds * 0.6); }
    void rotate(float dx, float dy) { azimuth_ += dx * 0.002f; elevation_ += dy * 0.002f; }
    void pan(float dx, float dy, float factor, float viewportSize) {
        if (viewportSize <= 0) return;
        const float scale = distance_ * factor / (viewportSize * 2);
        const float s = std::sin(azimuth_), c = std::cos(azimuth_);
        const float se = std::sin(elevation_), ce = std::cos(elevation_);
        center_[0] += scale * (-dx*s - dy*se*c);
        center_[1] += scale * dy*ce;
        center_[2] += scale * (dx*c - dy*se*s);
    }
    void zoom(float delta) { pinch(std::exp(-0.001f * delta)); }
    void pinch(float ratio) { if (std::isfinite(ratio) && ratio > 0) distance_ = std::clamp(distance_ * ratio, 0.05f, 100.0f); }
    ViewFrame frame() const {
        const float s = std::sin(azimuth_), c = std::cos(azimuth_);
        const float se = std::sin(elevation_), ce = std::cos(elevation_);
        return {{s,0,-c,0}, {-se*c,ce,-se*s,0}, {-c*ce,-se,-s*ce,0},
                {distance_*c*ce+center_[0],distance_*se+center_[1],distance_*s*ce+center_[2],distance_}};
    }
    CameraMatrices matrices() const {
        // Original shader: ray = forward + right*clip.x - up*clip.y. 90-degree square perspective.
        // Vulkan clip y grows down. Build exactly that projection, near=.01, far=1000.
        const auto v = frame();
        constexpr float nearPlane = 0.01f, farPlane = 1000.0f;
        constexpr float a = farPlane / (farPlane - nearPlane);
        constexpr float b = -nearPlane * a;
        CameraMatrices m;
        float rows[4][4]{};
        for (int i=0; i<3; ++i) {
            rows[0][i] = v.right[i];
            rows[1][i] = -v.up[i];
            rows[2][i] = a*v.forward[i];
            rows[3][i] = v.forward[i];
            rows[0][3] -= v.right[i]*v.origin[i];
            rows[1][3] += v.up[i]*v.origin[i];
            rows[2][3] -= a*v.forward[i]*v.origin[i];
            rows[3][3] -= v.forward[i]*v.origin[i];
        }
        rows[2][3] += b;
        for (int row=0; row<4; ++row) for (int col=0; col<4; ++col) m.viewProjection[col*4+row]=rows[row][col];
        // Analytic inverse: d=(clip.z-a*clip.w)/b, world=right*x-up*y+forward*w+origin*d.
        for (int i=0; i<3; ++i) {
            m.inverseViewProjection[i] = v.right[i];
            m.inverseViewProjection[4+i] = -v.up[i];
            m.inverseViewProjection[8+i] = v.origin[i]/b;
            m.inverseViewProjection[12+i] = v.forward[i]-v.origin[i]*a/b;
        }
        m.inverseViewProjection[11] = 1/b;
        m.inverseViewProjection[15] = -a/b;
        return m;
    }
private:
    float distance_ = 1.6f, azimuth_ = 2.8f, elevation_ = 0.4f;
    std::array<float,3> center_{};
};
}
