#include "volume_app.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <new>
#include <utility>
#include <qos/qos.h>

#include <dlfcn.h>

namespace volume {
namespace {

// Loaded per instance through the Vulkan loader; never linked directly.
VkSurfaceKHR createOhosSurface(VkInstance instance, void* nativeWindow) {
    if (instance == VK_NULL_HANDLE || nativeWindow == nullptr) {
        return VK_NULL_HANDLE;
    }
    const auto create = reinterpret_cast<PFN_vkCreateSurfaceOHOS>(
        vkGetInstanceProcAddr(instance, "vkCreateSurfaceOHOS"));
    if (create == nullptr) {
        return VK_NULL_HANDLE;
    }
    VkSurfaceCreateInfoOHOS info{};
    info.sType = VK_STRUCTURE_TYPE_SURFACE_CREATE_INFO_OHOS;
    info.pNext = nullptr;
    info.flags = 0;
    info.window = static_cast<OHNativeWindow*>(nativeWindow);
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (create(instance, &info, nullptr, &surface) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return surface;
}

bool querySurfaceSize(OH_NativeXComponent* component, void* window, uint32_t& width, uint32_t& height) {
    uint64_t w = 0;
    uint64_t h = 0;
    if (component == nullptr || window == nullptr) {
        return false;
    }
    if (OH_NativeXComponent_GetXComponentSize(component, window, &w, &h) != OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        return false;
    }
    width = static_cast<uint32_t>(std::min<uint64_t>(w, 0xFFFFFFFFull));
    height = static_cast<uint32_t>(std::min<uint64_t>(h, 0xFFFFFFFFull));
    return true;
}

constexpr int kMinTargetFps = 30;
constexpr int kMaxTargetFps = 144;
constexpr auto kStatsInterval = std::chrono::milliseconds(250); // ~4 Hz native refresh

// Linx Kit hotspot acceleration is API26 and AppGallery-release gated; resolve it dynamically so
// sideloaded or older builds never fail to load.
struct LinxHotspot {
    int32_t (*init)();
    int32_t (*begin)(uint32_t*);
    int32_t (*end)(uint32_t);
    void* library;
};

bool loadLinx(LinxHotspot& linx) {
    linx.library = dlopen("liblinx.so", RTLD_NOW | RTLD_LOCAL);
    if (!linx.library) return false;
    linx.init = reinterpret_cast<int32_t (*)()>(dlsym(linx.library, "HMS_LINX_HotspotAccelerateInit"));
    linx.begin = reinterpret_cast<int32_t (*)(uint32_t*)>(dlsym(linx.library, "HMS_LINX_HotspotAccelerateBegin"));
    linx.end = reinterpret_cast<int32_t (*)(uint32_t)>(dlsym(linx.library, "HMS_LINX_HotspotAccelerateEnd"));
    return linx.init && linx.begin && linx.end;
}

} // namespace

OH_NativeXComponent_Callback VolumeApp::callbacks_ = {};

VolumeApp& VolumeApp::instance() {
    // Intentionally never destroyed: avoids joining the render thread from a
    // static destructor while the ArkUI runtime tears down.
    static VolumeApp* app = new VolumeApp();
    return *app;
}

void VolumeApp::attachComponent(OH_NativeXComponent* component) {
    if (component == nullptr) {
        return;
    }
    component_ = component;
    callbacks_.OnSurfaceCreated = &VolumeApp::surfaceCreatedThunk;
    callbacks_.OnSurfaceChanged = &VolumeApp::surfaceChangedThunk;
    callbacks_.OnSurfaceDestroyed = &VolumeApp::surfaceDestroyedThunk;
    callbacks_.DispatchTouchEvent = &VolumeApp::touchEventThunk;
    OH_NativeXComponent_RegisterCallback(component, &callbacks_);
}

void VolumeApp::configure(const RenderConfig& config) {
    if (config.renderSize < 64 || config.renderSize > 4096) {
        setError("configure: renderSize must be within [64, 4096]");
        return;
    }
    if (config.targetFps < kMinTargetFps || config.targetFps > kMaxTargetFps) {
        setError("configure: targetFps must be within [30, 144]");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(controlMutex_);
        config_ = config;
        configDirty_ = true;
    }
    controlCv_.notify_all();
    if (component_ != nullptr) {
        OH_NativeXComponent_ExpectedRateRange range{};
        range.min = config.targetFps;
        range.max = config.targetFps;
        range.expected = config.targetFps;
        OH_NativeXComponent_SetExpectedFrameRateRange(component_, &range);
    }
}

void VolumeApp::setPaused(bool paused) {
    paused_.store(paused, std::memory_order_release);
    controlCv_.notify_all();
}

void VolumeApp::setForeground(bool foreground) {
    foreground_.store(foreground, std::memory_order_release);
    controlCv_.notify_all();
}

void VolumeApp::setAutoRotate(bool enabled) {
    pendingAutoRotate_.store(enabled ? 1 : 0, std::memory_order_release);
    controlCv_.notify_all();
}

void VolumeApp::setLinxEnabled(bool enabled) {
    linxEnabled_.store(enabled, std::memory_order_release);
    // The worker picks the change up on its next lifecycle restart; also update the label now.
    std::lock_guard<std::mutex> lock(statsMutex_);
    snapshot_.linxStatus = enabled ? "pending worker restart" : "disabled";
    controlCv_.notify_all();
}

void VolumeApp::resetCamera() {
    pendingReset_.store(true, std::memory_order_release);
    requestRedraw();
}

StatsSnapshot VolumeApp::stats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return snapshot_;
}

void VolumeApp::setError(const std::string& message) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    snapshot_.error = message;
}

void VolumeApp::publishStats(const Renderer& renderer) {
    const RenderStats current = renderer.stats();
    std::lock_guard<std::mutex> lock(statsMutex_);
    snapshot_.realFps = current.realFps;
    snapshot_.gpuMilliseconds = current.gpuMilliseconds;
    snapshot_.realFrames = current.realFrames;
    snapshot_.renderSize = current.renderSize;
    snapshot_.deviceName = current.deviceName;
    snapshot_.frameGenerationStatus = current.frameGenerationStatus;
    snapshot_.openGtxStatus = current.openGtxStatus;
    snapshot_.xengineStatus = current.xengineStatus;
    snapshot_.thermalLevel = current.thermalLevel;
}

void VolumeApp::onSurfaceCreated(OH_NativeXComponent* component, void* window) {
    uint32_t width = 0;
    uint32_t height = 0;
    querySurfaceSize(component, window, width, height);
    surfaceWidth_ = width;
    surfaceHeight_ = height;
    surfaceReady_.store(width > 0 && height > 0, std::memory_order_release);
    ensureWorker(window, width, height);
}

void VolumeApp::onSurfaceChanged(OH_NativeXComponent* component, void* window) {
    uint32_t width = 0;
    uint32_t height = 0;
    querySurfaceSize(component, window, width, height);
    surfaceWidth_ = width;
    surfaceHeight_ = height;
    surfaceReady_.store(width > 0 && height > 0, std::memory_order_release);
    controlCv_.notify_all();
    if (width == 0 || height == 0) {
        return; // Zero-sized surface: keep the renderer, the worker skips draws.
    }
    {
        std::lock_guard<std::mutex> lock(controlMutex_);
        if (!workerRunning_) {
            resizeWidth_ = width;
            resizeHeight_ = height;
        } else {
            resizeWidth_ = width;
            resizeHeight_ = height;
            resizePending_ = true;
            controlCv_.notify_all();
            return;
        }
    }
    ensureWorker(window, width, height);
}

void VolumeApp::onSurfaceDestroyed(OH_NativeXComponent* /*component*/, void* /*window*/) {
    surfaceWidth_ = 0;
    surfaceHeight_ = 0;
    surfaceReady_.store(false, std::memory_order_release);
    touchAnchored_ = false;
    touchCount_ = 0;
    stopWorker();
}

void VolumeApp::onTouchEvent(OH_NativeXComponent* component, void* window) {
    OH_NativeXComponent_TouchEvent event{};
    if (component == nullptr || window == nullptr) {
        return;
    }
    if (OH_NativeXComponent_GetTouchEvent(component, window, &event) != OH_NATIVEXCOMPONENT_RESULT_SUCCESS) {
        return;
    }
    if (event.type == OH_NATIVEXCOMPONENT_CANCEL) {
        touchAnchored_ = false;
        touchCount_ = 0;
        return;
    }
    if (event.type != OH_NATIVEXCOMPONENT_DOWN && event.type != OH_NATIVEXCOMPONENT_MOVE &&
        event.type != OH_NATIVEXCOMPONENT_UP) {
        return;
    }
    if (event.type != OH_NATIVEXCOMPONENT_MOVE) {
        touchAnchored_ = false; // Re-anchor when the finger set changes.
    }
    float xs[2] = {0.0f, 0.0f};
    float ys[2] = {0.0f, 0.0f};
    int count = 0;
    const uint32_t points = std::min<uint32_t>(event.numPoints, OH_NATIVE_XCOMPONENT_MAX_TOUCH_POINTS_NUMBER);
    for (uint32_t i = 0; i < points && count < 2; ++i) {
        const OH_NativeXComponent_TouchPoint& point = event.touchPoints[i];
        // ArkUI sets isPressed only on DOWN, not while moving. On UP the
        // array can still contain the released finger; other fingers remain active.
        if (event.type == OH_NATIVEXCOMPONENT_UP && point.id == event.id) {
            continue;
        }
        xs[count] = point.x;
        ys[count] = point.y;
        ++count;
    }
    if (count == 0) {
        touchAnchored_ = false;
        touchCount_ = 0;
        return;
    }
    applyTouch(xs[0], ys[0], count == 2, xs[1], ys[1]);
}

void VolumeApp::applyTouch(float x0, float y0, bool hasSecond, float x1, float y1) {
    const int count = hasSecond ? 2 : 1;
    if (!touchAnchored_ || count != touchCount_) {
        // Finger added/removed: re-anchor without applying a jump.
        touchAnchored_ = true;
        touchCount_ = count;
        if (hasSecond) {
            anchorCentroidX_ = (x0 + x1) * 0.5f;
            anchorCentroidY_ = (y0 + y1) * 0.5f;
            anchorDistance_ = std::hypot(x1 - x0, y1 - y0);
        } else {
            anchorCentroidX_ = x0;
            anchorCentroidY_ = y0;
            anchorDistance_ = 0.0f;
        }
        return;
    }
    if (hasSecond) {
        const float cx = (x0 + x1) * 0.5f;
        const float cy = (y0 + y1) * 0.5f;
        const float distance = std::hypot(x1 - x0, y1 - y0);
        const float viewport = std::min(surfaceWidth_, surfaceHeight_);
        {
            std::lock_guard<std::mutex> lock(cameraMutex_);
            camera_.pan(cx - anchorCentroidX_, cy - anchorCentroidY_, 4.0f, viewport);
            if (anchorDistance_ > 1.0f && distance > 1.0f) {
                camera_.pinch(anchorDistance_ / distance);
            }
        }
        anchorCentroidX_ = cx;
        anchorCentroidY_ = cy;
        anchorDistance_ = distance;
    } else {
        {
            std::lock_guard<std::mutex> lock(cameraMutex_);
            camera_.rotate(x0 - anchorCentroidX_, y0 - anchorCentroidY_);
        }
        anchorCentroidX_ = x0;
        anchorCentroidY_ = y0;
    }
    requestRedraw();
}

void VolumeApp::requestRedraw() {
    {
        std::lock_guard<std::mutex> lock(controlMutex_);
        redrawPending_ = true;
    }
    controlCv_.notify_all();
}

void VolumeApp::ensureWorker(void* window, uint32_t width, uint32_t height) {
    if (window == nullptr || width == 0 || height == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(controlMutex_);
    if (workerRunning_) {
        return;
    }
    stopRequested_.store(false, std::memory_order_release);
    configDirty_ = false;
    resizePending_ = false;
    redrawPending_ = true;
    try {
        worker_ = std::thread(&VolumeApp::workerLoop, this, window, width, height);
        workerRunning_ = true;
    } catch (const std::exception& e) {
        setError(std::string("render thread start: ") + e.what());
    } catch (...) {
        setError("render thread start: unknown failure");
    }
}

void VolumeApp::stopWorker() {
    std::thread joinable;
    {
        std::lock_guard<std::mutex> lock(controlMutex_);
        if (!workerRunning_) {
            return;
        }
        workerRunning_ = false;
        stopRequested_.store(true, std::memory_order_release);
        joinable = std::move(worker_);
    }
    controlCv_.notify_all();
    // The renderer (and therefore the VkSurfaceKHR backed by this native
    // window) is destroyed on the worker thread; joining before this callback
    // returns keeps the window valid for the whole Vulkan teardown.
    if (joinable.joinable()) {
        joinable.join();
    }
}

void VolumeApp::workerLoop(void* window, uint32_t width, uint32_t height) {
    std::string qosStatus = "inactive";
    // QOS_USER_INTERACTIVE: the documented level for foreground rendering threads.
    qosStatus = OH_QoS_SetThreadQoS(QOS_USER_INTERACTIVE) == 0
        ? "active: user-interactive render thread"
        : "inactive: OH_QoS_SetThreadQoS failed";
    LinxHotspot linx{};
    std::string linxStatus = "disabled";
    bool linxTried = false;
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        snapshot_.qosStatus = qosStatus;
        snapshot_.linxStatus = linxStatus;
    }
    std::unique_ptr<Renderer> renderer;
    RenderConfig localConfig{};
    {
        std::lock_guard<std::mutex> lock(controlMutex_);
        localConfig = config_;
        configDirty_ = false;
    }
    try {
        RendererOptions options;
        options.width = width;
        options.height = height;
        options.instanceExtensions = {"VK_KHR_surface", VK_OHOS_SURFACE_EXTENSION_NAME};
        options.createSurface = &createOhosSurface;
        options.nativeWindow = window;
        options.validation = false;
        renderer = std::make_unique<Renderer>(options);
        renderer->configure(localConfig);
        std::lock_guard<std::mutex> lock(statsMutex_);
        snapshot_.error.clear();
    } catch (const std::exception& e) {
        renderer.reset();
        setError(std::string("renderer init: ") + e.what());
    } catch (...) {
        renderer.reset();
        setError("renderer init: unknown failure");
    }

    using clock = std::chrono::steady_clock;
    auto nextFrame = clock::now();
    auto lastFrameTime = nextFrame;
    auto lastStatsTime = nextFrame;

    while (!stopRequested_.load(std::memory_order_acquire)) {
        const int fps = std::min(std::max(localConfig.targetFps, kMinTargetFps), kMaxTargetFps);
        const auto frameInterval = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / static_cast<double>(fps)));
        bool redraw = false;
        {
            std::unique_lock<std::mutex> lock(controlMutex_);
            const bool idle = renderer == nullptr || paused_.load(std::memory_order_acquire) ||
                !foreground_.load(std::memory_order_acquire) || !surfaceReady_.load(std::memory_order_acquire);
            if (idle) {
                controlCv_.wait_for(lock, std::chrono::milliseconds(100), [this, &renderer] {
                    return stopRequested_.load(std::memory_order_acquire) || configDirty_ || resizePending_ ||
                        (renderer != nullptr && foreground_.load(std::memory_order_acquire) &&
                         surfaceReady_.load(std::memory_order_acquire) &&
                         (!paused_.load(std::memory_order_acquire) || redrawPending_));
                });
            } else {
                controlCv_.wait_until(lock, nextFrame, [this] {
                    return stopRequested_.load(std::memory_order_acquire) || configDirty_ || resizePending_ ||
                        paused_.load(std::memory_order_acquire) || !foreground_.load(std::memory_order_acquire) ||
                        !surfaceReady_.load(std::memory_order_acquire);
                });
            }
            if (stopRequested_.load(std::memory_order_acquire)) {
                break;
            }
            try {
                if (configDirty_) {
                    localConfig = config_;
                    configDirty_ = false;
                    redrawPending_ = true;
                    if (renderer != nullptr) {
                        renderer->configure(localConfig);
                    }
                }
                if (resizePending_ && renderer != nullptr) {
                    renderer->resize(resizeWidth_, resizeHeight_);
                    redrawPending_ = true;
                }
            } catch (const std::exception& e) {
                renderer.reset(); // Surface the failure; worker stays alive for stats.
                setError(std::string("renderer reconfigure: ") + e.what());
            } catch (...) {
                renderer.reset();
                setError("renderer reconfigure: unknown failure");
            }
            resizePending_ = false;
            if (renderer != nullptr && foreground_.load(std::memory_order_acquire) &&
                surfaceReady_.load(std::memory_order_acquire)) {
                redraw = redrawPending_;
                redrawPending_ = false;
            }
            if (!idle) {
                nextFrame += frameInterval;
                const auto now = clock::now();
                if (nextFrame < now) {
                    nextFrame = now; // Do not burst after a stall.
                }
            }
        }

        // Camera commands from the UI thread; applied on the render thread.
        const int32_t autoRotate = pendingAutoRotate_.exchange(-1);
        const bool reset = pendingReset_.exchange(false);
        if (autoRotate >= 0 || reset) {
            std::lock_guard<std::mutex> lock(cameraMutex_);
            if (autoRotate >= 0) {
                camera_.autoRotate = autoRotate != 0;
            }
            if (reset) {
                camera_.reset();
            }
        }

        const bool paused = paused_.load(std::memory_order_acquire);
        if (renderer == nullptr || (paused && !redraw) ||
            !foreground_.load(std::memory_order_acquire) || !surfaceReady_.load(std::memory_order_acquire)) {
            lastFrameTime = clock::now();
            nextFrame = lastFrameTime;
            continue;
        }

        const auto now = clock::now();
        double dt = std::chrono::duration<double>(now - lastFrameTime).count();
        lastFrameTime = now;
        dt = std::min(dt, 0.1); // Clamp after pause so auto-rotation cannot jump.

        ViewFrame view{};
        CameraMatrices matrices{};
        {
            std::lock_guard<std::mutex> lock(cameraMutex_);
            if (!paused) {
                camera_.advance(dt);
            }
            view = camera_.frame();
            matrices = camera_.matrices();
        }
        // Live enable: the UI toggle takes effect without restarting the worker.
        const bool wantLinx = linxEnabled_.load(std::memory_order_acquire);
        if (wantLinx && !linxTried) {
            linxTried = true;
            if (loadLinx(linx) && linx.init() == 0) {
                linxStatus = "active: hotspot accelerate";
            } else if (!linx.library) {
                linxStatus = "unavailable: liblinx.so not present (API 26+, AG release)";
            } else {
                linxStatus = "inactive: init rejected (AG release only)";
            }
            std::lock_guard<std::mutex> lock(statsMutex_);
            snapshot_.linxStatus = linxStatus;
        }
        if (!wantLinx && linxStatus.rfind("active", 0) == 0) {
            linxStatus = "disabled";
            std::lock_guard<std::mutex> lock(statsMutex_);
            snapshot_.linxStatus = linxStatus;
        }
        uint32_t linxContext = 0;
        const bool linxFramed = linxStatus.rfind("active", 0) == 0 && linx.begin(&linxContext) == 0;
        try {
            renderer->draw(view, matrices);
        } catch (const std::exception& e) {
            renderer.reset(); // Keep the worker alive for the error display.
            setError(std::string("render: ") + e.what());
            if (linxFramed) linx.end(linxContext);
            continue;
        } catch (...) {
            renderer.reset();
            setError("render: unknown failure");
            if (linxFramed) linx.end(linxContext);
            continue;
        }
        if (linxFramed) linx.end(linxContext);
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            snapshot_.error.clear();
        }
        if (now - lastStatsTime >= kStatsInterval) {
            publishStats(*renderer);
            lastStatsTime = now;
        }
    }
}

void VolumeApp::surfaceCreatedThunk(OH_NativeXComponent* component, void* window) {
    instance().onSurfaceCreated(component, window);
}

void VolumeApp::surfaceChangedThunk(OH_NativeXComponent* component, void* window) {
    instance().onSurfaceChanged(component, window);
}

void VolumeApp::surfaceDestroyedThunk(OH_NativeXComponent* component, void* window) {
    instance().onSurfaceDestroyed(component, window);
}

void VolumeApp::touchEventThunk(OH_NativeXComponent* component, void* window) {
    instance().onTouchEvent(component, window);
}

} // namespace volume
