#pragma once

#include "volume/camera.h"
#include "volume/renderer.h"

#include <ace/xcomponent/native_interface_xcomponent.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace volume {

// Immutable snapshot of renderer stats plus the last native error, refreshed by
// the render worker at ~4 Hz and copied out for NAPI callers under statsMutex.
struct StatsSnapshot {
    double realFps = 0.0;
    double gpuMilliseconds = -1.0;
    uint64_t realFrames = 0;
    uint32_t renderSize = 0;
    std::string deviceName;
    std::string frameGenerationStatus;
    std::string openGtxStatus;
    int thermalLevel = 0;
    std::string xengineStatus;
    std::string qosStatus;
    std::string linxStatus;
    std::string error;
};

// Owns the XComponent surface lifecycle, the dedicated render worker thread,
// camera input state and the renderer configuration. All Vulkan/renderer work
// happens on the worker thread; ArkUI threads only post control flags.
class VolumeApp {
public:
    static VolumeApp& instance();

    void attachComponent(OH_NativeXComponent* component);

    // NAPI entry points (ArkUI thread).
    void configure(const RenderConfig& config);
    void setPaused(bool paused);
    void setForeground(bool foreground);
    void setLinxEnabled(bool enabled);
    void setAutoRotate(bool enabled);
    void resetCamera();
    StatsSnapshot stats() const;

    // XComponent callbacks (ArkUI thread).
    void onSurfaceCreated(OH_NativeXComponent* component, void* window);
    void onSurfaceChanged(OH_NativeXComponent* component, void* window);
    void onSurfaceDestroyed(OH_NativeXComponent* component, void* window);
    void onTouchEvent(OH_NativeXComponent* component, void* window);

private:
    VolumeApp() = default;
    ~VolumeApp() = default;
    VolumeApp(const VolumeApp&) = delete;
    VolumeApp& operator=(const VolumeApp&) = delete;

    void ensureWorker(void* window, uint32_t width, uint32_t height);
    void stopWorker();
    void workerLoop(void* window, uint32_t width, uint32_t height);
    void setError(const std::string& message);
    void publishStats(const Renderer& renderer);
    void applyTouch(float x0, float y0, bool hasSecond, float x1, float y1);
    void requestRedraw();

    static void surfaceCreatedThunk(OH_NativeXComponent* component, void* window);
    static void surfaceChangedThunk(OH_NativeXComponent* component, void* window);
    static void surfaceDestroyedThunk(OH_NativeXComponent* component, void* window);
    static void touchEventThunk(OH_NativeXComponent* component, void* window);

    static OH_NativeXComponent_Callback callbacks_;

    // --- ArkUI-thread-only members (no locking) ---
    OH_NativeXComponent* component_ = nullptr; // set once during module init
    uint32_t surfaceWidth_ = 0;
    uint32_t surfaceHeight_ = 0;
    // Two-finger gesture anchor state.
    bool touchAnchored_ = false;
    int touchCount_ = 0;
    float anchorCentroidX_ = 0.0f;
    float anchorCentroidY_ = 0.0f;
    float anchorDistance_ = 0.0f;

    // --- Worker control (guarded by controlMutex_) ---
    mutable std::mutex controlMutex_;
    std::condition_variable controlCv_;
    std::thread worker_;
    bool workerRunning_ = false;
    RenderConfig config_{};
    bool configDirty_ = false;
    uint32_t resizeWidth_ = 0;
    uint32_t resizeHeight_ = 0;
    bool resizePending_ = false;
    bool redrawPending_ = true;

    std::atomic<bool> stopRequested_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> foreground_{true};
    std::atomic<bool> linxEnabled_{false};
    std::atomic<bool> surfaceReady_{false};
    std::atomic<int32_t> pendingAutoRotate_{-1}; // -1 none, 0 disabled, 1 enabled
    std::atomic<bool> pendingReset_{false};

    // --- Camera shared between ArkUI input thread and worker ---
    std::mutex cameraMutex_;
    Camera camera_;

    // --- Stats published by the worker, read by NAPI ---
    mutable std::mutex statsMutex_;
    StatsSnapshot snapshot_;
};

} // namespace volume
