#include "napi/native_api.h"
#include "volume_app.h"

#include <ace/xcomponent/native_interface_xcomponent.h>

#include <cstdint>

namespace {

bool asNumber(napi_env env, napi_value value, double& out) {
    napi_valuetype type = napi_undefined;
    if (napi_typeof(env, value, &type) != napi_ok || type != napi_number) {
        return false;
    }
    return napi_get_value_double(env, value, &out) == napi_ok;
}

bool asBool(napi_env env, napi_value value, bool& out) {
    napi_valuetype type = napi_undefined;
    if (napi_typeof(env, value, &type) != napi_ok || type != napi_boolean) {
        return false;
    }
    return napi_get_value_bool(env, value, &out) == napi_ok;
}

napi_value Configure(napi_env env, napi_callback_info info) {
    size_t argc = 6;
    napi_value args[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok || argc != 6) {
        napi_throw_type_error(env, nullptr, "configure(renderSize, optimizedKernel, aiFrameGeneration, openGtx, xengineUpscale, targetFps)");
        return nullptr;
    }
    double renderSize = 0.0;
    double targetFps = 0.0;
    bool optimizedKernel = false;
    bool aiFrameGeneration = false;
    bool openGtx = false;
    bool xengineUpscale = false;
    if (!asNumber(env, args[0], renderSize) || !asBool(env, args[1], optimizedKernel) ||
        !asBool(env, args[2], aiFrameGeneration) || !asBool(env, args[3], openGtx) ||
        !asBool(env, args[4], xengineUpscale) || !asNumber(env, args[5], targetFps)) {
        napi_throw_type_error(env, nullptr, "configure: expected (number, boolean, boolean, boolean, boolean, number)");
        return nullptr;
    }
    volume::RenderConfig config{};
    config.renderSize = static_cast<uint32_t>(renderSize);
    config.optimizedKernel = optimizedKernel;
    config.aiFrameGeneration = aiFrameGeneration;
    config.openGtx = openGtx;
    config.xengineUpscale = xengineUpscale;
    config.targetFps = static_cast<int>(targetFps);
    volume::VolumeApp::instance().configure(config);
    return nullptr;
}

napi_value SetPaused(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    bool paused = false;
    if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok || argc != 1 ||
        !asBool(env, args[0], paused)) {
        napi_throw_type_error(env, nullptr, "setPaused(paused: boolean)");
        return nullptr;
    }
    volume::VolumeApp::instance().setPaused(paused);
    return nullptr;
}

napi_value SetLinxEnabled(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    bool enabled = false;
    if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok || argc != 1 ||
        !asBool(env, args[0], enabled)) {
        napi_throw_type_error(env, nullptr, "setLinxEnabled(enabled: boolean)");
        return nullptr;
    }
    volume::VolumeApp::instance().setLinxEnabled(enabled);
    return nullptr;
}

napi_value SetForeground(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    bool foreground = false;
    if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok || argc != 1 ||
        !asBool(env, args[0], foreground)) {
        napi_throw_type_error(env, nullptr, "setForeground(foreground: boolean)");
        return nullptr;
    }
    volume::VolumeApp::instance().setForeground(foreground);
    return nullptr;
}

napi_value SetAutoRotate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    bool enabled = false;
    if (napi_get_cb_info(env, info, &argc, args, nullptr, nullptr) != napi_ok || argc != 1 ||
        !asBool(env, args[0], enabled)) {
        napi_throw_type_error(env, nullptr, "setAutoRotate(enabled: boolean)");
        return nullptr;
    }
    volume::VolumeApp::instance().setAutoRotate(enabled);
    return nullptr;
}

napi_value ResetCamera(napi_env env, napi_callback_info info) {
    (void)env;
    (void)info;
    volume::VolumeApp::instance().resetCamera();
    return nullptr;
}

void setNumber(napi_env env, napi_value object, const char* key, double value) {
    napi_value item = nullptr;
    if (napi_create_double(env, value, &item) == napi_ok) {
        napi_set_named_property(env, object, key, item);
    }
}

void setUint(napi_env env, napi_value object, const char* key, double value) {
    napi_value item = nullptr;
    if (napi_create_uint32(env, static_cast<uint32_t>(value), &item) == napi_ok) {
        napi_set_named_property(env, object, key, item);
    }
}

void setString(napi_env env, napi_value object, const char* key, const std::string& value) {
    napi_value item = nullptr;
    if (napi_create_string_utf8(env, value.c_str(), value.size(), &item) == napi_ok) {
        napi_set_named_property(env, object, key, item);
    }
}

napi_value Stats(napi_env env, napi_callback_info info) {
    (void)info;
    const volume::StatsSnapshot snapshot = volume::VolumeApp::instance().stats();
    napi_value object = nullptr;
    if (napi_create_object(env, &object) != napi_ok) {
        return nullptr;
    }
    setNumber(env, object, "realFps", snapshot.realFps);
    setNumber(env, object, "gpuMilliseconds", snapshot.gpuMilliseconds);
    setNumber(env, object, "realFrames", static_cast<double>(snapshot.realFrames));
    setUint(env, object, "renderSize", snapshot.renderSize);
    setString(env, object, "deviceName", snapshot.deviceName);
    setString(env, object, "frameGenerationStatus", snapshot.frameGenerationStatus);
    setString(env, object, "openGtxStatus", snapshot.openGtxStatus);
    setString(env, object, "xengineStatus", snapshot.xengineStatus);
    setString(env, object, "qosStatus", snapshot.qosStatus);
    setString(env, object, "linxStatus", snapshot.linxStatus);
    setNumber(env, object, "thermalLevel", static_cast<double>(snapshot.thermalLevel));
    setString(env, object, "error", snapshot.error);
    return object;
}

} // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"configure", nullptr, Configure, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setPaused", nullptr, SetPaused, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setForeground", nullptr, SetForeground, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setLinxEnabled", nullptr, SetLinxEnabled, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setAutoRotate", nullptr, SetAutoRotate, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetCamera", nullptr, ResetCamera, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"stats", nullptr, Stats, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    if (napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc) != napi_ok) {
        return nullptr;
    }

    // The ArkUI XComponent passes its native handle through the module exports
    // during load; register the surface/touch callbacks on it.
    napi_value exportInstance = nullptr;
    if (napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &exportInstance) == napi_ok &&
        exportInstance != nullptr) {
        OH_NativeXComponent* nativeXComponent = nullptr;
        if (napi_unwrap(env, exportInstance, reinterpret_cast<void**>(&nativeXComponent)) == napi_ok &&
            nativeXComponent != nullptr) {
            volume::VolumeApp::instance().attachComponent(nativeXComponent);
        }
    }
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void) {
    napi_module_register(&demoModule);
}
