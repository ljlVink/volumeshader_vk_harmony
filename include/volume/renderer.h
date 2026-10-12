#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace volume {
// Matches the GLSL push-constant block: four aligned vec4 values (64 bytes).
struct alignas(16) ViewFrame {
    std::array<float, 4> right{};
    std::array<float, 4> up{};
    std::array<float, 4> forward{};
    std::array<float, 4> origin{}; // w = camera distance
};
struct CameraMatrices {
    std::array<float, 16> viewProjection{};
    std::array<float, 16> inverseViewProjection{};
};
struct RenderConfig {
    uint32_t renderSize = 1024;
    bool optimizedKernel = true;
    bool aiFrameGeneration = true;
    bool openGtx = true;
    bool xengineUpscale = false;
    int targetFps = 120;
};
struct RenderStats {
    double realFps = 0;
    double gpuMilliseconds = -1; // -1 if timestamp queries unavailable
    uint64_t realFrames = 0;
    uint32_t renderSize = 0;
    std::string deviceName;
    std::string frameGenerationStatus;
    std::string openGtxStatus;
    std::string xengineStatus;
    int thermalLevel = 0;
};
// Surface is created after instance creation. No callback = offscreen mode.
using SurfaceFactory = VkSurfaceKHR (*)(VkInstance, void*);
struct RendererOptions {
    uint32_t width = 1024;
    uint32_t height = 1024;
    std::vector<const char*> instanceExtensions;
    SurfaceFactory createSurface = nullptr;
    void* nativeWindow = nullptr;
    bool validation = false;
};
class Renderer {
public:
    explicit Renderer(const RendererOptions& options);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    void configure(const RenderConfig& config);
    void resize(uint32_t width, uint32_t height);
    // Returns false for temporarily unavailable/out-of-date surface, never counts it as a rendered frame.
    bool draw(const ViewFrame& view, const CameraMatrices& matrices);
    RenderStats stats() const;
    // Waits for GPU and reads the last real scene (square RGBA8), not an interpolated frame.
    std::vector<uint8_t> capture();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
