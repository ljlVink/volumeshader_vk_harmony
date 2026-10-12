#include "volume/renderer.h"

#include "volume_shaders.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if VOLUME_HMS
#include <graphics_game_sdk/frame_generation_base.h>
#include <graphics_game_sdk/frame_generation_vk.h>
#include <graphics_game_sdk/opengtx_base.h>
#include <dlfcn.h>
#include <xengine/xeg_extension_defs.h>
#include <xengine/xeg_vulkan_extension.h>
#include <xengine/xeg_vulkan_spatial_upscale.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace volume {
namespace {

constexpr uint32_t kFramesInFlight = 2;
constexpr uint32_t kQueriesPerFrame = 2;
constexpr uint32_t kMinRenderSize = 16;
constexpr uint32_t kMaxRenderSize = 4096;
constexpr float kFpsSmoothing = 0.1f;
constexpr float kGpuSmoothing = 0.25f;

void throwIf(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(result)) + ")");
    }
}

uint32_t clampU32(uint32_t value, uint32_t low, uint32_t high) {
    return value < low ? low : (value > high ? high : value);
}

VKAPI_ATTR VkBool32 VKAPI_CALL onDebugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[vulkan] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

struct AllocatedImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

#if VOLUME_HMS
// HMS_OpenGTX_CreateContext takes no user-data pointer; a single renderer owns the context.
std::atomic<int> g_thermalLevel{0};

void onThermalLevel(OpenGTX_TempLevel level) { g_thermalLevel.store(static_cast<int>(level)); }

char kGtxPackageName[] = "com.ljlvink.volumeshader";
char kGtxAppVersion[] = "1.0.0";
char kGtxEngineVersion[] = "";
char kGtxSceneDescription[] = "volumeshader_bm raymarch";
#endif

} // namespace

struct Renderer::Impl {
    // Core Vulkan (1.1, ordinary linked prototypes).
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphicsFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties deviceProperties{};
    uint32_t timestampValidBits = 0;
    std::string deviceName;

    // Window. Offscreen mode keeps surface == VK_NULL_HANDLE and skips every WSI path.
    SurfaceFactory createSurface = nullptr;
    void* nativeWindow = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkExtent2D pendingExtent{0, 0};

    // Swapchain.
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D swapExtent{0, 0};
    VkFormat swapFormat = VK_FORMAT_R8G8B8A8_UNORM;
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    std::vector<VkFramebuffer> swapFramebuffers;
    std::vector<VkSemaphore> presentSemaphores; // One per swapchain image; re-signaled only after re-acquire.
    std::vector<VkFence> imagesInFlight;        // Render fence currently owning each swapchain image.
    bool swapDirty = true;

    // Command/sync resources: two frames in flight, reused every frame without allocation.
    VkCommandPool commandPool = VK_NULL_HANDLE;
    struct FrameSlot {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence rendered = VK_NULL_HANDLE; // Signaled when the frame's commands complete.
        VkFence acquired = VK_NULL_HANDLE; // Signaled by vkAcquireNextImageKHR.
        bool drained = true;
    };
    std::array<FrameSlot, kFramesInFlight> frames{};
    uint32_t frameIdx = 0;
    int lastSubmitted = -1;

    // Timestamps for the scene pass only (present/FG work excluded).
    VkQueryPool queryPool = VK_NULL_HANDLE;
    bool timestampsUsable = false;
    uint64_t timestampMask = 0;

    // Pipelines and layouts.
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout sampledSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kFramesInFlight> composeSet{};
    std::array<VkDescriptorSet, kFramesInFlight> presentSet{};
    VkPipelineLayout sceneLayout = VK_NULL_HANDLE;
    VkPipelineLayout presentLayout = VK_NULL_HANDLE;
    VkRenderPass scenePass = VK_NULL_HANDLE;
    VkPipeline sceneReference = VK_NULL_HANDLE;
    VkPipeline sceneOptimized = VK_NULL_HANDLE;
    VkRenderPass composePass = VK_NULL_HANDLE; // RGBA8 full-surface target feeding frame generation.
    VkPipeline composePipeline = VK_NULL_HANDLE;
    VkRenderPass presentPass = VK_NULL_HANDLE; // Swapchain format; recreated with the swapchain.
    VkPipeline presentPipeline = VK_NULL_HANDLE;

    // Square scene targets (RGBA8_UNORM, one per frame in flight).
    struct SceneTarget {
        AllocatedImage image;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
    };
    std::array<SceneTarget, kFramesInFlight> scene{};
    uint32_t sceneSize = 0;

    // Capture staging (host visible, persistently mapped).
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void* stagingMapped = nullptr;
    VkDeviceSize stagingSize = 0;
    bool stagingCoherent = false;
    VkCommandBuffer captureCommand = VK_NULL_HANDLE;
    VkFence captureFence = VK_NULL_HANDLE;

    // Stats.
    RenderConfig config{};
    uint64_t realFrames = 0;
    double fpsEma = -1.0;
    double frameSecondsEma = -1.0;
    double gpuEmaMs = -1.0;
    std::chrono::steady_clock::time_point lastFrameClock{};
    bool hasLastFrameClock = false;

#if VOLUME_HMS
    // HMS frame generation: AI interpolation (system present), two in-flight real frames.
    FG_Context_VK* fgContext = VK_NULL_HANDLE;
    struct FgFrame {
        AllocatedImage compose; // Letterboxed square scene at full surface extent (AI input).
        FG_Image_VK* composeWrap = VK_NULL_HANDLE;
        VkFramebuffer composeFramebuffer = VK_NULL_HANDLE;
    };
    std::array<FgFrame, kFramesInFlight> fg{};
    bool fgWanted = false;
    bool fgActive = false;
    bool fgErrorLatch = false; // Sticky after failure; cleared by configure().
    std::string fgStatus = "disabled";

    // HMS OpenGTX scheduler hints.
    OpenGTX_Context* gtxContext = VK_NULL_HANDLE;
    bool gtxWanted = false;
    bool gtxActive = false;
    std::string gtxStatus = "disabled";
    int32_t renderThreadId = 0;
    ViewFrame gtxLastView{};
    bool gtxHasLastView = false;
    // XEngine spatial upscaler (optional): low-res square scene -> native square via the Maleoon GPU.
    XEG_SpatialUpscale spatialUpscale = VK_NULL_HANDLE;
    AllocatedImage upscaleTarget;
    uint32_t upscaleSide = 0;
    uint32_t upscaleSceneSize = 0;
    bool xegWanted = false;
    bool xegSettled = false; // Terminal condition reached; only configure() retries.
    std::string xegStatus = "disabled";
#endif

    explicit Impl(const RendererOptions& options);
    ~Impl();
    void cleanup();

    void createInstance(const RendererOptions& options);
    void createDevice();
    void createFrameResources();
    void createPipelineObjects();
    void createFullscreenPipeline(VkRenderPass renderPass, VkPipelineLayout layout, const uint32_t* vertexCode,
                                  size_t vertexWords, const uint32_t* fragmentCode, size_t fragmentWords,
                                  VkPipeline* pipeline);
    void createSceneTargets();
    void destroySceneTargets();
    void createStaging();
    void destroyStaging();

    bool recreateSwapchain();
    void destroySwapchainObjects(bool destroyHandle);
    bool acquireNextSwapImage(uint32_t& imageIndex);

    void drainFrame(uint32_t frame);
    void readFrameTimestamps(uint32_t frame);
    void sampleFrameRate();

    void writeSampledDescriptor(VkDescriptorSet set, VkImageView view);
    AllocatedImage createImage2D(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage);
    void destroyImage(AllocatedImage& image);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required, bool strict);

#if VOLUME_HMS
    void trySetupFrameGeneration();
    void teardownFrameGeneration();
    void dispatchFrameGeneration(VkCommandBuffer command, uint32_t frame, const CameraMatrices& matrices);
    void trySetupOpenGtx();
    void teardownOpenGtx();
    void dispatchOpenGtxFrameInfo(const ViewFrame& view);
    void trySetupSpatialUpscale();
    void teardownSpatialUpscale();
    void recordSpatialUpscale(VkCommandBuffer command, uint32_t frame);
#endif

    bool drawFrame(const ViewFrame& view, const CameraMatrices& matrices);
    bool drawOffscreen(const ViewFrame& view);
    void recordScenePass(VkCommandBuffer command, uint32_t frame, const ViewFrame& view);
    void recordSamplingPass(VkCommandBuffer command, VkRenderPass renderPass, VkFramebuffer framebuffer,
                            VkPipeline pipeline, VkExtent2D extent, VkDescriptorSet set, const float* pushExtent);

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

Renderer::Impl::Impl(const RendererOptions& options) {
    createSurface = options.createSurface;
    nativeWindow = options.nativeWindow;
    pendingExtent.width = options.width ? options.width : 1024;
    pendingExtent.height = options.height ? options.height : 1024;
#if VOLUME_HMS
    renderThreadId = static_cast<int32_t>(::syscall(SYS_gettid));
#endif
    try {
    createInstance(options);
    if (createSurface) {
        surface = createSurface(instance, nativeWindow);
        if (!surface) throw std::runtime_error("Surface factory returned VK_NULL_HANDLE");
    }
    createDevice();
    createFrameResources();
    createPipelineObjects();
    createSceneTargets();
    } catch (...) {
        cleanup();
        throw;
    }
}

Renderer::Impl::~Impl() { cleanup(); }

void Renderer::Impl::cleanup() {
    if (device != VK_NULL_HANDLE) {
        // Pending presents and command buffers finish before anything they reference is destroyed.
        vkDeviceWaitIdle(device);
    }
#if VOLUME_HMS
    // HMS contexts are torn down orderly, strictly before the VkDevice they use.
    teardownOpenGtx();
    teardownFrameGeneration();
    teardownSpatialUpscale();
#endif
    destroySceneTargets();
    destroyStaging();
    destroySwapchainObjects(true);
    if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (sampler != VK_NULL_HANDLE) vkDestroySampler(device, sampler, nullptr);
    if (sampledSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, sampledSetLayout, nullptr);
    if (sceneReference != VK_NULL_HANDLE) vkDestroyPipeline(device, sceneReference, nullptr);
    if (sceneOptimized != VK_NULL_HANDLE) vkDestroyPipeline(device, sceneOptimized, nullptr);
    if (composePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, composePipeline, nullptr);
    if (sceneLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, sceneLayout, nullptr);
    if (presentLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, presentLayout, nullptr);
    if (scenePass != VK_NULL_HANDLE) vkDestroyRenderPass(device, scenePass, nullptr);
    if (composePass != VK_NULL_HANDLE) vkDestroyRenderPass(device, composePass, nullptr);
    if (queryPool != VK_NULL_HANDLE) vkDestroyQueryPool(device, queryPool, nullptr);
    for (auto& frame : frames) {
        if (frame.rendered != VK_NULL_HANDLE) vkDestroyFence(device, frame.rendered, nullptr);
        if (frame.acquired != VK_NULL_HANDLE) vkDestroyFence(device, frame.acquired, nullptr);
    }
    if (captureFence != VK_NULL_HANDLE) vkDestroyFence(device, captureFence, nullptr);
    if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
    if (debugMessenger != VK_NULL_HANDLE) {
        const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(instance, debugMessenger, nullptr);
    }
    if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}

void Renderer::Impl::createInstance(const RendererOptions& options) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "volumeshader_bmvk";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "volume";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    std::vector<const char*> layers;
    std::vector<const char*> extensions(options.instanceExtensions.begin(), options.instanceExtensions.end());
    bool debugUtilsAdded = false;
    if (options.validation) {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        if (count) vkEnumerateInstanceLayerProperties(&count, available.data());
        bool hasValidation = false;
        for (const auto& layer : available) {
            if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) hasValidation = true;
        }
        if (hasValidation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            uint32_t extCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            if (extCount) vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());
            for (const auto& ext : exts) {
                if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    debugUtilsAdded = true;
                    break;
                }
            }
        } else {
            std::fprintf(stderr, "volume: validation requested but VK_LAYER_KHRONOS_validation is absent\n");
        }
    }

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    throwIf(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");

    if (debugUtilsAdded) {
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create) {
            VkDebugUtilsMessengerCreateInfoEXT debug{};
            debug.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug.pfnUserCallback = onDebugMessage;
            create(instance, &debug, nullptr, &debugMessenger);
        }
    }
}

void Renderer::Impl::createDevice() {
    uint32_t count = 0;
    throwIf(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physicals(count);
    throwIf(vkEnumeratePhysicalDevices(instance, &count, physicals.data()), "vkEnumeratePhysicalDevices");

    int bestScore = -1;
    for (const VkPhysicalDevice candidate : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_1) continue;
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        if (familyCount) vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());

        bool hasSwapchainExtension = true;
        if (surface != VK_NULL_HANDLE) {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            if (extCount) vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extCount, exts.data());
            hasSwapchainExtension = false;
            for (const auto& ext : exts) {
                if (std::strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) hasSwapchainExtension = true;
            }
        }
        if (!hasSwapchainExtension) continue;

        for (uint32_t family = 0; family < familyCount; ++family) {
            if (!(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (surface != VK_NULL_HANDLE) {
                VkBool32 supported = VK_FALSE;
                if (vkGetPhysicalDeviceSurfaceSupportKHR(candidate, family, surface, &supported) != VK_SUCCESS ||
                    supported != VK_TRUE) {
                    continue;
                }
            }
            int score = 1 + (families[family].timestampValidBits > 0 ? 1 : 0);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 4;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 2;
            if (score > bestScore) {
                bestScore = score;
                physical = candidate;
                graphicsFamily = family;
                deviceProperties = props;
                timestampValidBits = families[family].timestampValidBits;
                deviceName = props.deviceName;
            }
        }
    }
    if (physical == VK_NULL_HANDLE) throw std::runtime_error("No Vulkan 1.1 device with a graphics/present queue");

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = graphicsFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    const char* swapchainExtension[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (surface != VK_NULL_HANDLE) {
        deviceInfo.enabledExtensionCount = 1;
        deviceInfo.ppEnabledExtensionNames = swapchainExtension;
    }
    throwIf(vkCreateDevice(physical, &deviceInfo, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, graphicsFamily, 0, &queue);

    timestampsUsable = timestampValidBits > 0;
    timestampMask = timestampValidBits >= 64 ? ~0ull : ((1ull << timestampValidBits) - 1ull);
}

void Renderer::Impl::createFrameResources() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    throwIf(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = commandPool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = kFramesInFlight + 1; // frames + one-shot capture buffer
    std::vector<VkCommandBuffer> commands(kFramesInFlight + 1);
    throwIf(vkAllocateCommandBuffers(device, &allocate, commands.data()), "vkAllocateCommandBuffers");
    for (uint32_t i = 0; i < kFramesInFlight; ++i) frames[i].command = commands[i];
    captureCommand = commands[kFramesInFlight];

    for (auto& frame : frames) {
        VkFenceCreateInfo renderedInfo{};
        renderedInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        renderedInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        throwIf(vkCreateFence(device, &renderedInfo, nullptr, &frame.rendered), "vkCreateFence(rendered)");
        VkFenceCreateInfo acquiredInfo{};
        acquiredInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        throwIf(vkCreateFence(device, &acquiredInfo, nullptr, &frame.acquired), "vkCreateFence(acquired)");
    }
    VkFenceCreateInfo captureFenceInfo{};
    captureFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    throwIf(vkCreateFence(device, &captureFenceInfo, nullptr, &captureFence), "vkCreateFence(capture)");

    if (timestampsUsable) {
        VkQueryPoolCreateInfo queryInfo{};
        queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = kFramesInFlight * kQueriesPerFrame;
        throwIf(vkCreateQueryPool(device, &queryInfo, nullptr, &queryPool), "vkCreateQueryPool");
    }
}

void Renderer::Impl::createPipelineObjects() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.0f;
    throwIf(vkCreateSampler(device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = 1;
    setLayoutInfo.pBindings = &binding;
    throwIf(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &sampledSetLayout),
            "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = kFramesInFlight * 2;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = kFramesInFlight * 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    throwIf(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo setsInfo{};
    setsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setsInfo.descriptorPool = descriptorPool;
    setsInfo.descriptorSetCount = 1;
    setsInfo.pSetLayouts = &sampledSetLayout;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        throwIf(vkAllocateDescriptorSets(device, &setsInfo, &composeSet[i]), "vkAllocateDescriptorSets");
        throwIf(vkAllocateDescriptorSets(device, &setsInfo, &presentSet[i]), "vkAllocateDescriptorSets");
    }

    // Scene: the 64-byte ViewFrame push block is shared by vertex and fragment stages.
    VkPushConstantRange sceneRange{};
    sceneRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneRange.offset = 0;
    sceneRange.size = sizeof(ViewFrame);
    VkPipelineLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.pushConstantRangeCount = 1;
    sceneLayoutInfo.pPushConstantRanges = &sceneRange;
    throwIf(vkCreatePipelineLayout(device, &sceneLayoutInfo, nullptr, &sceneLayout), "vkCreatePipelineLayout");

    // Present/compose: sampled set plus vec2 extent push constant (fragment stage).
    VkPushConstantRange presentRange{};
    presentRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    presentRange.offset = 0;
    presentRange.size = 8;
    VkDescriptorSetLayout presentSetLayouts[] = {sampledSetLayout};
    VkPipelineLayoutCreateInfo presentLayoutInfo{};
    presentLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    presentLayoutInfo.setLayoutCount = 1;
    presentLayoutInfo.pSetLayouts = presentSetLayouts;
    presentLayoutInfo.pushConstantRangeCount = 1;
    presentLayoutInfo.pPushConstantRanges = &presentRange;
    throwIf(vkCreatePipelineLayout(device, &presentLayoutInfo, nullptr, &presentLayout), "vkCreatePipelineLayout");

    // Scene render pass: load DONT_CARE (the fullscreen triangle overwrites every pixel), store,
    // final layout SHADER_READ_ONLY so the sampling pass in the same command buffer can read it.
    VkAttachmentDescription sceneAttachment{};
    sceneAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    sceneAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    sceneAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    sceneAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    sceneAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    sceneAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    sceneAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    VkSubpassDependency writesToSampled{};
    writesToSampled.srcSubpass = 0;
    writesToSampled.dstSubpass = VK_SUBPASS_EXTERNAL;
    writesToSampled.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    writesToSampled.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    writesToSampled.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    writesToSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    writesToSampled.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo scenePassInfo{};
    scenePassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    scenePassInfo.attachmentCount = 1;
    scenePassInfo.pAttachments = &sceneAttachment;
    scenePassInfo.subpassCount = 1;
    scenePassInfo.pSubpasses = &subpass;
    scenePassInfo.dependencyCount = 1;
    scenePassInfo.pDependencies = &writesToSampled;
    throwIf(vkCreateRenderPass(device, &scenePassInfo, nullptr, &scenePass), "vkCreateRenderPass(scene)");

    // Full-surface AI input is already sampleable. Keeping initialSync == finalSync also makes
    // collecting-history/error dispatches safe without assuming whether the SDK recorded commands.
    VkAttachmentDescription composeAttachment = sceneAttachment;
    VkRenderPassCreateInfo composePassInfo{};
    composePassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    composePassInfo.attachmentCount = 1;
    composePassInfo.pAttachments = &composeAttachment;
    composePassInfo.subpassCount = 1;
    composePassInfo.pSubpasses = &subpass;
    composePassInfo.dependencyCount = 1;
    composePassInfo.pDependencies = &writesToSampled;
    throwIf(vkCreateRenderPass(device, &composePassInfo, nullptr, &composePass), "vkCreateRenderPass(compose)");

    createFullscreenPipeline(scenePass, sceneLayout, shaders::vertex.data(), shaders::vertex.size(),
                             shaders::fragmentReference.data(), shaders::fragmentReference.size(), &sceneReference);
    createFullscreenPipeline(scenePass, sceneLayout, shaders::vertex.data(), shaders::vertex.size(),
                             shaders::fragmentOptimized.data(), shaders::fragmentOptimized.size(),
                             &sceneOptimized);
    createFullscreenPipeline(composePass, presentLayout, shaders::presentVertex.data(), shaders::presentVertex.size(),
                             shaders::presentFragment.data(), shaders::presentFragment.size(), &composePipeline);
}

void Renderer::Impl::createFullscreenPipeline(VkRenderPass renderPass, VkPipelineLayout layout,
                                              const uint32_t* vertexCode, size_t vertexWords,
                                              const uint32_t* fragmentCode, size_t fragmentWords,
                                              VkPipeline* pipeline) {
    VkShaderModuleCreateInfo vertInfo{};
    vertInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vertInfo.codeSize = vertexWords * sizeof(uint32_t);
    vertInfo.pCode = vertexCode;
    VkShaderModuleCreateInfo fragInfo{};
    fragInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fragInfo.codeSize = fragmentWords * sizeof(uint32_t);
    fragInfo.pCode = fragmentCode;
    VkShaderModule vertModule = VK_NULL_HANDLE, fragModule = VK_NULL_HANDLE;
    throwIf(vkCreateShaderModule(device, &vertInfo, nullptr, &vertModule), "vkCreateShaderModule");
    throwIf(vkCreateShaderModule(device, &fragInfo, nullptr, &fragModule), "vkCreateShaderModule");
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertexInput{}; // Vertex data comes from gl_VertexIndex only.
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                           VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{}; // No depth buffer anywhere in the renderer.
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = layout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    const VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pipeline);
    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    throwIf(result, "vkCreateGraphicsPipelines");
}

uint32_t Renderer::Impl::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required, bool strict) {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    if (strict) throw std::runtime_error("No memory type satisfies required properties");
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if (typeBits & (1u << i)) return i;
    }
    throw std::runtime_error("No compatible memory type");
}

AllocatedImage Renderer::Impl::createImage2D(uint32_t width, uint32_t height, VkFormat format,
                                             VkImageUsageFlags usage) {
    AllocatedImage out;
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    throwIf(vkCreateImage(device, &imageInfo, nullptr, &out.image), "vkCreateImage");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, out.image, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    throwIf(vkAllocateMemory(device, &allocate, nullptr, &out.memory), "vkAllocateMemory");
    throwIf(vkBindImageMemory(device, out.image, out.memory, 0), "vkBindImageMemory");
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = out.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    throwIf(vkCreateImageView(device, &viewInfo, nullptr, &out.view), "vkCreateImageView");
    return out;
}

void Renderer::Impl::destroyImage(AllocatedImage& image) {
    if (image.view != VK_NULL_HANDLE) vkDestroyImageView(device, image.view, nullptr);
    if (image.image != VK_NULL_HANDLE) vkDestroyImage(device, image.image, nullptr);
    if (image.memory != VK_NULL_HANDLE) vkFreeMemory(device, image.memory, nullptr);
    image = AllocatedImage{};
}

void Renderer::Impl::createSceneTargets() {
    sceneSize = clampU32(config.renderSize, kMinRenderSize, kMaxRenderSize);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        scene[i].image = createImage2D(sceneSize, sceneSize, VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = scenePass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &scene[i].image.view;
        framebufferInfo.width = sceneSize;
        framebufferInfo.height = sceneSize;
        framebufferInfo.layers = 1;
        throwIf(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &scene[i].framebuffer),
                "vkCreateFramebuffer(scene)");
    }
}

void Renderer::Impl::destroySceneTargets() {
    for (auto& target : scene) {
        if (target.framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, target.framebuffer, nullptr);
        destroyImage(target.image);
        target.framebuffer = VK_NULL_HANDLE;
    }
    sceneSize = 0;
}

void Renderer::Impl::createStaging() {
    stagingSize = static_cast<VkDeviceSize>(sceneSize) * sceneSize * 4;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = stagingSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    throwIf(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, stagingBuffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t chosen = ~0u;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) continue;
        const VkMemoryPropertyFlags props = memory.memoryTypes[i].propertyFlags;
        if ((props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
        if (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            chosen = i;
            break;
        }
        if (chosen == ~0u) chosen = i;
    }
    if (chosen == ~0u) throw std::runtime_error("No host-visible memory for capture staging");
    stagingCoherent = (memory.memoryTypes[chosen].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = chosen;
    throwIf(vkAllocateMemory(device, &allocate, nullptr, &stagingMemory), "vkAllocateMemory(staging)");
    throwIf(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0), "vkBindBufferMemory");
    throwIf(vkMapMemory(device, stagingMemory, 0, VK_WHOLE_SIZE, 0, &stagingMapped), "vkMapMemory");
}

void Renderer::Impl::destroyStaging() {
    if (stagingMapped) {
        vkUnmapMemory(device, stagingMemory);
        stagingMapped = nullptr;
    }
    if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, stagingBuffer, nullptr);
    if (stagingMemory != VK_NULL_HANDLE) vkFreeMemory(device, stagingMemory, nullptr);
    stagingBuffer = VK_NULL_HANDLE;
    stagingMemory = VK_NULL_HANDLE;
    stagingSize = 0;
}

bool Renderer::Impl::recreateSwapchain() {
    // Safe teardown point: pending presents and command buffers complete first, then FG objects
    // (whose wraps reference images sized to the old surface extent) go away.
    vkDeviceWaitIdle(device);
#if VOLUME_HMS
    teardownFrameGeneration();
    teardownSpatialUpscale();
#endif
    destroySwapchainObjects(true);

    const auto bail = [this]() {
        destroySwapchainObjects(true);
        swapDirty = true;
        return false;
    };

    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps) != VK_SUCCESS) return bail();
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) {
        swapDirty = true; // Surface hidden; retry when it has a size again.
        return false;
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) {
        extent.width = clampU32(pendingExtent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = clampU32(pendingExtent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) return bail();

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    if (formatCount) vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &formatCount, formats.data());
    VkSurfaceFormatKHR chosen = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    bool have = false;
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_UNORM) {
            chosen = format;
            have = true;
            break;
        }
    }
    if (!have && !formats.empty()) chosen = formats[0];

    uint32_t imageCount = caps.minImageCount > 3 ? caps.minImageCount : 3;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;
    if (imageCount < caps.minImageCount) imageCount = caps.minImageCount;

    const VkImageUsageFlags wantedUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((caps.supportedUsageFlags & wantedUsage) != wantedUsage) {
        std::fprintf(stderr, "volume: surface does not support color attachment usage\n");
        return bail();
    }
    const VkCompositeAlphaFlagBitsKHR alphaPreferences[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
    VkCompositeAlphaFlagBitsKHR alpha = alphaPreferences[3];
    for (const auto candidate : alphaPreferences) {
        if (caps.supportedCompositeAlpha & candidate) {
            alpha = candidate;
            break;
        }
    }

    VkSwapchainCreateInfoKHR swapInfo{};
    swapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapInfo.surface = surface;
    swapInfo.minImageCount = imageCount;
    swapInfo.imageFormat = chosen.format;
    swapInfo.imageColorSpace = chosen.colorSpace;
    swapInfo.imageExtent = extent;
    swapInfo.imageArrayLayers = 1;
    swapInfo.imageUsage = wantedUsage;
    swapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapInfo.preTransform = caps.currentTransform;
    swapInfo.compositeAlpha = alpha;
    swapInfo.clipped = VK_TRUE;
    swapInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (vkCreateSwapchainKHR(device, &swapInfo, nullptr, &swapchain) != VK_SUCCESS) return bail();

    uint32_t actualImages = 0;
    if (vkGetSwapchainImagesKHR(device, swapchain, &actualImages, nullptr) != VK_SUCCESS) return bail();
    swapImages.resize(actualImages);
    if (vkGetSwapchainImagesKHR(device, swapchain, &actualImages, swapImages.data()) != VK_SUCCESS) return bail();
    swapFormat = chosen.format;
    swapExtent = extent;

    // Present render pass and pipeline for the new swapchain format.
    VkAttachmentDescription presentAttachment{};
    presentAttachment.format = swapFormat;
    presentAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    presentAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // Fullscreen pass overwrites everything.
    presentAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    presentAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    presentAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    presentAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    presentAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    VkSubpassDependency toPresent{};
    toPresent.srcSubpass = 0;
    toPresent.dstSubpass = VK_SUBPASS_EXTERNAL;
    toPresent.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    toPresent.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toPresent.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    VkRenderPassCreateInfo presentPassInfo{};
    presentPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    presentPassInfo.attachmentCount = 1;
    presentPassInfo.pAttachments = &presentAttachment;
    presentPassInfo.subpassCount = 1;
    presentPassInfo.pSubpasses = &subpass;
    presentPassInfo.dependencyCount = 1;
    presentPassInfo.pDependencies = &toPresent;
    if (vkCreateRenderPass(device, &presentPassInfo, nullptr, &presentPass) != VK_SUCCESS) return bail();
    createFullscreenPipeline(presentPass, presentLayout, shaders::presentVertex.data(),
                             shaders::presentVertex.size(), shaders::presentFragment.data(),
                             shaders::presentFragment.size(), &presentPipeline);

    swapViews.resize(actualImages);
    swapFramebuffers.resize(actualImages);
    presentSemaphores.resize(actualImages);
    imagesInFlight.assign(actualImages, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < actualImages; ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapFormat;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &viewInfo, nullptr, &swapViews[i]) != VK_SUCCESS) return bail();
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = presentPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &swapViews[i];
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;
        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapFramebuffers[i]) != VK_SUCCESS) {
            return bail();
        }
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &presentSemaphores[i]) != VK_SUCCESS) return bail();
    }
    swapDirty = false;
#if VOLUME_HMS
    if (fgWanted && !fgErrorLatch) trySetupFrameGeneration();
#endif
    return true;
}

void Renderer::Impl::destroySwapchainObjects(bool destroyHandle) {
    for (auto framebuffer : swapFramebuffers) {
        if (framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    swapFramebuffers.clear();
    for (auto view : swapViews) {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
    }
    swapViews.clear();
    for (auto semaphore : presentSemaphores) {
        if (semaphore != VK_NULL_HANDLE) vkDestroySemaphore(device, semaphore, nullptr);
    }
    presentSemaphores.clear();
    imagesInFlight.clear();
    swapImages.clear();
    if (presentPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, presentPipeline, nullptr);
        presentPipeline = VK_NULL_HANDLE;
    }
    if (presentPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, presentPass, nullptr);
        presentPass = VK_NULL_HANDLE;
    }
    if (destroyHandle && swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

bool Renderer::Impl::acquireNextSwapImage(uint32_t& imageIndex) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        VkSurfaceCapabilitiesKHR caps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps) != VK_SUCCESS) return false;
        if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) return false; // Surface hidden: pause.
        if (swapDirty || swapchain == VK_NULL_HANDLE) {
            if (!recreateSwapchain()) return false;
        }
        vkResetFences(device, 1, &frames[frameIdx].acquired);
        // Fence-based acquire: no semaphore is left signaled across OUT_OF_DATE retries.
        const VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE,
                                                      frames[frameIdx].acquired, &imageIndex);
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
            vkWaitForFences(device, 1, &frames[frameIdx].acquired, VK_TRUE, UINT64_MAX);
            if (result == VK_SUBOPTIMAL_KHR) swapDirty = true;
            return true;
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // A failed acquire may still have signaled the fence; consume it before reuse.
            if (vkGetFenceStatus(device, frames[frameIdx].acquired) == VK_SUCCESS) {
                vkResetFences(device, 1, &frames[frameIdx].acquired);
            }
            swapDirty = true;
            continue;
        }
        return false;
    }
    return false;
}

void Renderer::Impl::drainFrame(uint32_t frame) {
    auto& slot = frames[frame];
    if (slot.drained) return;
    vkWaitForFences(device, 1, &slot.rendered, VK_TRUE, UINT64_MAX);
    readFrameTimestamps(frame);
    slot.drained = true;
}

void Renderer::Impl::readFrameTimestamps(uint32_t frame) {
    if (!timestampsUsable) return;
    uint64_t data[kQueriesPerFrame * 2] = {};
    const VkResult result = vkGetQueryPoolResults(device, queryPool, frame * kQueriesPerFrame, kQueriesPerFrame,
                                                  sizeof(data), data, sizeof(uint64_t) * 2,
                                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) return;
    if (!data[1] || !data[3]) return; // begin/end availability
    const uint64_t delta = (data[2] - data[0]) & timestampMask; // Wraps when validBits < 64.
    const double ms = static_cast<double>(delta) * deviceProperties.limits.timestampPeriod / 1.0e6;
    gpuEmaMs = gpuEmaMs < 0.0 ? ms : gpuEmaMs + (ms - gpuEmaMs) * kGpuSmoothing;
}

void Renderer::Impl::sampleFrameRate() {
    const auto now = std::chrono::steady_clock::now();
    if (hasLastFrameClock) {
        const double seconds = std::chrono::duration<double>(now - lastFrameClock).count();
        if (seconds > 0.0 && seconds <= 1.0) { // Skip pause gaps.
            frameSecondsEma = frameSecondsEma < 0.0 ? seconds : frameSecondsEma + (seconds - frameSecondsEma) * kFpsSmoothing;
            fpsEma = 1.0 / frameSecondsEma;
        }
    }
    lastFrameClock = now;
    hasLastFrameClock = true;
}

void Renderer::Impl::writeSampledDescriptor(VkDescriptorSet set, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void Renderer::Impl::recordScenePass(VkCommandBuffer command, uint32_t frame, const ViewFrame& view) {
    if (timestampsUsable) {
        vkCmdResetQueryPool(command, queryPool, frame * kQueriesPerFrame, kQueriesPerFrame);
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, frame * kQueriesPerFrame);
    }
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = scenePass;
    begin.framebuffer = scene[frame].framebuffer;
    begin.renderArea.extent = {sceneSize, sceneSize};
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.width = static_cast<float>(sceneSize);
    viewport.height = static_cast<float>(sceneSize);
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = {sceneSize, sceneSize};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      config.optimizedKernel ? sceneOptimized : sceneReference);
    vkCmdPushConstants(command, sceneLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(ViewFrame), &view);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
    if (timestampsUsable) {
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool,
                            frame * kQueriesPerFrame + 1);
    }
}

void Renderer::Impl::recordSamplingPass(VkCommandBuffer command, VkRenderPass renderPass, VkFramebuffer framebuffer,
                                        VkPipeline pipeline, VkExtent2D extent, VkDescriptorSet set,
                                        const float* pushExtent) {
    VkClearValue clear{};
    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = renderPass;
    begin.framebuffer = framebuffer;
    begin.renderArea.extent = extent;
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdPushConstants(command, presentLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 2, pushExtent);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, presentLayout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
}

bool Renderer::Impl::drawOffscreen(const ViewFrame& view) {
    const uint32_t frame = frameIdx;
    drainFrame(frame);
    vkResetFences(device, 1, &frames[frame].rendered);
    VkCommandBuffer command = frames[frame].command;
    vkResetCommandBuffer(command, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    throwIf(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer");
    recordScenePass(command, frame, view);
    throwIf(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    throwIf(vkQueueSubmit(queue, 1, &submit, frames[frame].rendered), "vkQueueSubmit");
    frames[frame].drained = false;
    lastSubmitted = static_cast<int>(frame);
    frameIdx = (frame + 1) % kFramesInFlight;
    ++realFrames;
    sampleFrameRate();
    return true;
}

bool Renderer::Impl::drawFrame(const ViewFrame& view, const CameraMatrices& matrices) {
#if !VOLUME_HMS
    (void)matrices; // Camera matrices only feed HMS frame generation on HarmonyOS.
#endif
    if (surface == VK_NULL_HANDLE) {
#if VOLUME_HMS
        dispatchOpenGtxFrameInfo(view);
#endif
        return drawOffscreen(view);
    }
#if VOLUME_HMS
    if (fgErrorLatch && fgContext) {
        // A failed dispatch may have recorded commands referencing its input. Finish them before teardown.
        vkDeviceWaitIdle(device);
        teardownFrameGeneration();
        // Keep the failure latched; only an explicit configure() retries the feature.
    }
    if (fgWanted && !fgActive && !fgErrorLatch && !swapDirty && swapchain != VK_NULL_HANDLE) {
        trySetupFrameGeneration(); // Deferred setup: swapchain extent must exist first.
    }
    if (xegWanted && spatialUpscale == VK_NULL_HANDLE && !xegSettled && !swapDirty &&
        swapchain != VK_NULL_HANDLE) {
        trySetupSpatialUpscale(); // Also deferred: the native square side comes from the swapchain.
    }
    dispatchOpenGtxFrameInfo(view);
#endif

    uint32_t imageIndex = 0;
    if (!acquireNextSwapImage(imageIndex)) return false;

    const uint32_t frame = frameIdx;
    drainFrame(frame);
    vkResetFences(device, 1, &frames[frame].rendered);
    // Image-in-flight ownership: never render into a swapchain image still owned by another frame.
    VkFence owner = imagesInFlight[imageIndex];
    if (owner != VK_NULL_HANDLE && owner != frames[frame].rendered &&
        vkGetFenceStatus(device, owner) != VK_SUCCESS) {
        vkWaitForFences(device, 1, &owner, VK_TRUE, UINT64_MAX);
    }

    VkCommandBuffer command = frames[frame].command;
    vkResetCommandBuffer(command, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    throwIf(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer");

    recordScenePass(command, frame, view);
    // The application ALWAYS presents its real frame. In FG_PRESENT_BY_SYSTEM mode the system
    // owns and presents predictions between real frames; presenting a prediction here would double-interpolate.
    VkImageView sourceView = scene[frame].image.view;
    const float letterboxExtent[2] = {static_cast<float>(swapExtent.width), static_cast<float>(swapExtent.height)};
    const float* pushExtent = letterboxExtent;
#if VOLUME_HMS
    // XEngine optional GPU upscale first: square scene -> native square; the present letterbox is unchanged.
    if (spatialUpscale != VK_NULL_HANDLE && sceneSize == upscaleSceneSize) {
        recordSpatialUpscale(command, frame);
        sourceView = upscaleTarget.view;
    }
    const float identityExtent[2] = {1.0f, 1.0f};
    if (fgActive) {
        writeSampledDescriptor(composeSet[frame], sourceView);
        recordSamplingPass(command, composePass, fg[frame].composeFramebuffer, composePipeline, swapExtent,
                           composeSet[frame], letterboxExtent);
        dispatchFrameGeneration(command, frame, matrices);
        sourceView = fg[frame].compose.view;
        pushExtent = identityExtent;
    }
#endif
    writeSampledDescriptor(presentSet[frame], sourceView);
    recordSamplingPass(command, presentPass, swapFramebuffers[imageIndex], presentPipeline, swapExtent,
                       presentSet[frame], pushExtent);
    throwIf(vkEndCommandBuffer(command), "vkEndCommandBuffer");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &presentSemaphores[imageIndex];
    throwIf(vkQueueSubmit(queue, 1, &submit, frames[frame].rendered), "vkQueueSubmit");
    imagesInFlight[imageIndex] = frames[frame].rendered;

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &presentSemaphores[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &imageIndex;
    const VkResult presented = vkQueuePresentKHR(queue, &present);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) swapDirty = true;
    else throwIf(presented, "vkQueuePresentKHR");

    frames[frame].drained = false;
    lastSubmitted = static_cast<int>(frame);
    frameIdx = (frame + 1) % kFramesInFlight;
    if (presented == VK_ERROR_OUT_OF_DATE_KHR) return false;
    ++realFrames;
    sampleFrameRate();
    return true;
}

#if VOLUME_HMS
void Renderer::Impl::trySetupFrameGeneration() {
    teardownFrameGeneration();
    if (!fgWanted || surface == VK_NULL_HANDLE) {
        fgStatus = fgWanted ? "disabled (offscreen benchmark)" : "disabled";
        return;
    }
    if (swapExtent.width == 0 || swapExtent.height == 0) return; // Retry once the surface has a size.
    // Symbol presence is the compatibility guard. APIAVAILABLE would introduce a compiler-rt
    // version helper whose private OHOS getter is absent on some real HarmonyOS distributions.
    using QuerySupport = bool (*)(FG_FeatureType);
    const auto querySupport = reinterpret_cast<QuerySupport>(
        dlsym(dlopen("libframegeneration.so", RTLD_NOLOAD | RTLD_NOW), "HMS_FG_IsFrameGenerationSupported"));
    if (!querySupport) {
        fgStatus = "unavailable: runtime lacks the API26 AI capability query";
        fgErrorLatch = true;
        return;
    }
    if (!querySupport(INTERPOLATION_AI_VULKAN)) {
        fgStatus = "unsupported: INTERPOLATION_AI_VULKAN not available on this device";
        fgErrorLatch = true;
        return;
    }
    FG_ContextDescription_VK contextDescription{};
    contextDescription.vkInstance = instance;
    contextDescription.vkPhysicalDevice = physical;
    contextDescription.vkDevice = device;
    contextDescription.framesInFlight = kFramesInFlight;
    contextDescription.fnVulkanLoaderFunction = &vkGetInstanceProcAddr;
    fgContext = HMS_FG_CreateContext_VK(&contextDescription);
    if (!fgContext) {
        fgStatus = "HMS_FG_CreateContext_VK failed";
        fgErrorLatch = true;
        return;
    }
    const auto fail = [&](const char* what, FG_ErrorCode code) {
        fgStatus = std::string("deactivated: ") + what + " error " + std::to_string(static_cast<int>(code));
        teardownFrameGeneration();
        fgErrorLatch = true;
    };
    FG_AlgorithmModeInfo mode{};
    mode.predictionMode = FG_PREDICTION_MODE_INTERPOLATION;
    mode.meMode = FG_ME_MODE_ENHANCED; // AI path: enhanced motion estimation, no depth required.
    FG_ErrorCode code = HMS_FG_SetAlgorithmMode_VK(fgContext, &mode);
    if (code != FG_SUCCESS) {
        fail("HMS_FG_SetAlgorithmMode_VK", code);
        return;
    }
    FG_ResolutionInfo resolution{};
    resolution.inputColorResolution = {swapExtent.width, swapExtent.height}; // Full-surface AI input.
    resolution.inputDepthStencilResolution = {0, 0};                          // No depth buffer in AI mode.
    resolution.outputColorResolution = {swapExtent.width, swapExtent.height};
    code = HMS_FG_SetResolution_VK(fgContext, &resolution);
    if (code != FG_SUCCESS) {
        fail("HMS_FG_SetResolution_VK", code);
        return;
    }
    FG_ImageFormat_VK format{};
    format.inputColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    format.inputDepthStencilFormat = VK_FORMAT_D24_UNORM_S8_UINT; // SDK default, no depth image is supplied.
    format.outputColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    code = HMS_FG_SetImageFormat_VK(fgContext, &format);
    if (code != FG_SUCCESS) {
        fail("HMS_FG_SetImageFormat_VK", code);
        return;
    }
    FG_IntegrationInfo integration{};
    integration.presentMode = FG_PRESENT_BY_SYSTEM;
    integration.textureCachedByGame = false; // The algorithm buffers textures itself.
    integration.needFlipInputColor = false;
    integration.needFlipOutputColor = false;
    code = HMS_FG_SetIntegrationMode_VK(fgContext, &integration);
    if (code != FG_SUCCESS) {
        fail("HMS_FG_SetIntegrationMode_VK", code);
        return;
    }
    code = HMS_FG_SetUiPredictionEnabled_VK(fgContext, false);
    if (code != FG_SUCCESS) { fail("HMS_FG_SetUiPredictionEnabled_VK", code); return; }
    code = HMS_FG_SetTargetFps_VK(fgContext, config.targetFps);
    if (code != FG_SUCCESS) { fail("HMS_FG_SetTargetFps_VK", code); return; }

    const VkImageUsageFlags composeUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        fg[i].compose = createImage2D(swapExtent.width, swapExtent.height, VK_FORMAT_R8G8B8A8_UNORM, composeUsage);
        // Wrappers exist only while the underlying Vulkan images do.
        fg[i].composeWrap = HMS_FG_CreateImage_VK(fgContext, fg[i].compose.image, fg[i].compose.view);
        if (!fg[i].composeWrap) {
            fgStatus = "HMS_FG_CreateImage_VK failed";
            teardownFrameGeneration();
            fgErrorLatch = true;
            return;
        }
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = composePass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &fg[i].compose.view;
        framebufferInfo.width = swapExtent.width;
        framebufferInfo.height = swapExtent.height;
        framebufferInfo.layers = 1;
        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &fg[i].composeFramebuffer) != VK_SUCCESS) {
            fgStatus = "vkCreateFramebuffer(compose) failed";
            teardownFrameGeneration();
            fgErrorLatch = true;
            return;
        }
    }
    code = HMS_FG_Activate_VK(fgContext);
    if (code != FG_SUCCESS) {
        fail("HMS_FG_Activate_VK", code);
        return;
    }
    fgActive = true;
    fgStatus = "active: AI interpolation (system present, " + std::to_string(config.targetFps) + " fps target)";
}

void Renderer::Impl::teardownFrameGeneration() {
    if (!fgContext) {
        fgActive = false;
        return;
    }
    if (fgActive) HMS_FG_Deactivate_VK(fgContext);
    // Unwrap FG images strictly before destroying the underlying Vulkan objects and the context.
    for (auto& entry : fg) {
        if (entry.composeWrap) HMS_FG_DestroyImage_VK(fgContext, entry.composeWrap);
        entry.composeWrap = nullptr;
    }
    HMS_FG_DestroyContext_VK(&fgContext);
    fgContext = VK_NULL_HANDLE;
    for (auto& entry : fg) {
        if (entry.composeFramebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device, entry.composeFramebuffer, nullptr);
            entry.composeFramebuffer = VK_NULL_HANDLE;
        }
        destroyImage(entry.compose);
    }
    fgActive = false;
}

void Renderer::Impl::dispatchFrameGeneration(VkCommandBuffer command, uint32_t frame,
                                             const CameraMatrices& matrices) {
    FG_DispatchDescription_VK description{};
    description.inputColorInfo.image = fg[frame].composeWrap;
    description.inputColorInfo.initialSync = {VK_ACCESS_SHADER_READ_BIT,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
    description.inputColorInfo.finalSync = {VK_ACCESS_SHADER_READ_BIT,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
    std::memset(&description.inputDepthStencilInfo, 0, sizeof(description.inputDepthStencilInfo));
    std::memcpy(description.viewProj.data, matrices.viewProjection.data(), sizeof(FG_Mat4x4));
    std::memcpy(description.invViewProj.data, matrices.inverseViewProjection.data(), sizeof(FG_Mat4x4));
    // The camera describes the square scene. Map its clip coordinates into the full-surface letterbox.
    const float shortSide = static_cast<float>(std::min(swapExtent.width, swapExtent.height));
    const float scaleX = shortSide / static_cast<float>(swapExtent.width);
    const float scaleY = shortSide / static_cast<float>(swapExtent.height);
    for (uint32_t i = 0; i < 4; ++i) {
        description.viewProj.data[i * 4] *= scaleX;
        description.viewProj.data[i * 4 + 1] *= scaleY;
        description.invViewProj.data[i] /= scaleX;
        description.invViewProj.data[4 + i] /= scaleY;
    }
    description.vkCommandBuffer = command;
    description.frameIdx = static_cast<uint8_t>(frame);
    const FG_ErrorCode code = HMS_FG_Dispatch_VK(fgContext, &description);
    if (code == FG_SUCCESS) {
        if (fgStatus == "active: collecting history") {
            fgStatus = "active: AI interpolation (system present, " + std::to_string(config.targetFps) +
                       " fps target)";
        }
        return;
    }
    if (code == FG_COLLECTING_PREVIOUS_FRAMES) {
        fgStatus = "active: collecting history"; // Nonfatal: present the real composed frame meanwhile.
    } else {
        fgStatus = "deactivated: HMS_FG_Dispatch_VK error " + std::to_string(static_cast<int>(code));
        fgErrorLatch = true; // Tear down after the frame referencing these images completes.
    }
}

void Renderer::Impl::trySetupOpenGtx() {
    teardownOpenGtx();
    if (!gtxWanted) {
        gtxStatus = "disabled";
        return;
    }
    gtxContext = HMS_OpenGTX_CreateContext(&onThermalLevel);
    if (!gtxContext) {
        gtxStatus = "HMS_OpenGTX_CreateContext failed";
        return;
    }
    const auto fail = [&](const char* what, OpenGTX_ErrorCode code) {
        gtxStatus = std::string("inactive: ") + what + " error " + std::to_string(static_cast<int>(code));
        teardownOpenGtx();
    };
    OpenGTX_ConfigDescription configuration{};
    configuration.mode = ADAPTIVE_MODE;
    configuration.targetFPS = static_cast<int32_t>(clampU32(static_cast<uint32_t>(config.targetFps), 30, 144));
    configuration.packageName = kGtxPackageName;
    configuration.appVersion = kGtxAppVersion;
    configuration.engineType = OTHERS_ENGINE;
    configuration.engineVersion = kGtxEngineVersion;
    configuration.gameType = OTHERS_TYPE;
    configuration.pictureQualityMaxLevel = UHD;
    const uint32_t maxWidth = surface != VK_NULL_HANDLE ? swapExtent.width : sceneSize;
    const uint32_t maxHeight = surface != VK_NULL_HANDLE ? swapExtent.height : sceneSize;
    configuration.resolutionMaxValue.width = static_cast<int32_t>(clampU32(maxWidth, 360, 7680));
    configuration.resolutionMaxValue.height = static_cast<int32_t>(clampU32(maxHeight, 360, 7680));
    configuration.gameMainThreadId = renderThreadId;
    configuration.gameRenderThreadId = renderThreadId; // Configured from the render thread itself.
    configuration.gameKeyThreadIds[0] = renderThreadId;
    configuration.gameKeyThreadIds[1] = 0;
    configuration.gameKeyThreadIds[2] = 0;
    configuration.gameKeyThreadIds[3] = 0;
    configuration.gameKeyThreadIds[4] = 0;
    configuration.vulkanSupport = true;
    OpenGTX_ErrorCode code = HMS_OpenGTX_SetConfiguration(gtxContext, &configuration);
    if (code != OPENGTX_SUCCESS) {
        fail("HMS_OpenGTX_SetConfiguration", code);
        return;
    }
    code = HMS_OpenGTX_Activate(gtxContext);
    if (code != OPENGTX_SUCCESS) {
        fail("HMS_OpenGTX_Activate", code);
        return;
    }
    OpenGTX_GameSceneInfo sceneInfo{};
    sceneInfo.sceneID = HEAVY_LOAD; // 1000-step raymarch workload.
    sceneInfo.description = kGtxSceneDescription;
    sceneInfo.recommendFPS = 0; // 0 = no recommendation; the scheduler decides.
    sceneInfo.minFPS = 0;
    sceneInfo.maxFPS = 0;
    sceneInfo.resolutionCurValue.width = static_cast<int32_t>(clampU32(sceneSize, 360, 7680));
    sceneInfo.resolutionCurValue.height = static_cast<int32_t>(clampU32(sceneSize, 360, 7680));
    HMS_OpenGTX_DispatchGameSceneInfo(gtxContext, &sceneInfo);
    gtxActive = true;
    gtxStatus = "active: adaptive scheduler, target " + std::to_string(config.targetFps) + " fps";
}

void Renderer::Impl::teardownOpenGtx() {
    if (!gtxContext) {
        gtxActive = false;
        return;
    }
    if (gtxActive) HMS_OpenGTX_Deactivate(gtxContext);
    HMS_OpenGTX_DestroyContext(&gtxContext);
    gtxContext = VK_NULL_HANDLE;
    gtxActive = false;
    g_thermalLevel.store(0);
}

void Renderer::Impl::dispatchOpenGtxFrameInfo(const ViewFrame& view) {
    if (!gtxActive) return;
    if (gtxHasLastView && std::memcmp(&gtxLastView, &view, sizeof(ViewFrame)) == 0) return;
    gtxLastView = view;
    gtxHasLastView = true;
    const auto clampAngle = [](float value) { return std::max(-360.0f, std::min(360.0f, value)); };
    const float fx = view.forward[0], fy = view.forward[1], fz = view.forward[2];
    const float horizontal = std::sqrt(fx * fx + fz * fz);
    OpenGTX_FrameRenderInfo frameInfo{};
    frameInfo.mainCameraPosition = {clampAngle(view.origin[0]), clampAngle(view.origin[1]),
                                    clampAngle(view.origin[2])};
    frameInfo.mainCameraRotate = {clampAngle(std::atan2(fx, fz) * 57.2957795f),
                                  clampAngle(std::atan2(-fy, horizontal) * 57.2957795f), 0.0f};
    HMS_OpenGTX_DispatchFrameRenderInfo(gtxContext, &frameInfo);
}

void Renderer::Impl::teardownSpatialUpscale() {
    if (spatialUpscale != VK_NULL_HANDLE) {
        HMS_XEG_DestroySpatialUpscale(spatialUpscale);
        spatialUpscale = VK_NULL_HANDLE;
    }
    destroyImage(upscaleTarget);
    upscaleSide = 0;
    upscaleSceneSize = 0;
}

void Renderer::Impl::trySetupSpatialUpscale() {
    teardownSpatialUpscale();
    if (!xegWanted || surface == VK_NULL_HANDLE || swapchain == VK_NULL_HANDLE || swapExtent.width == 0) {
        xegStatus = xegWanted ? "pending surface" : "disabled";
        return; // Retried lazily once the swapchain exists.
    }
    static bool enumerated = false;
    static bool supported = false;
    if (!enumerated) {
        uint32_t count = 0;
        if (HMS_XEG_EnumerateDeviceExtensionProperties(physical, &count, nullptr) != VK_SUCCESS || count == 0) {
            xegStatus = "unsupported: XEngine reports no extensions";
            xegSettled = true;
            return;
        }
        std::vector<XEG_ExtensionProperties> extensions(count);
        const VkResult enumeratedResult = HMS_XEG_EnumerateDeviceExtensionProperties(physical, &count,
                                                                                     extensions.data());
        supported = false;
        if (enumeratedResult == VK_SUCCESS || enumeratedResult == VK_INCOMPLETE) {
            for (uint32_t i = 0; i < count; ++i) {
                if (std::strcmp(extensions[i].extensionName, XEG_SPATIAL_UPSCALE_EXTENSION_NAME) == 0) {
                    supported = true;
                    break;
                }
            }
        }
        enumerated = true;
    }
    if (!supported) {
        xegStatus = "unsupported: XEG_spatial_upscale not offered by this GPU";
        xegSettled = true;
        return;
    }
    const uint32_t side = std::min(swapExtent.width, swapExtent.height);
    if (sceneSize >= side) {
        xegStatus = "inactive: render size already at native square";
        xegSettled = true;
        return;
    }
    XEG_SpatialUpscaleCreateInfo createInfo{};
    createInfo.inputSize = {sceneSize, sceneSize};
    createInfo.inputRegion = {{0, 0}, {sceneSize, sceneSize}};
    createInfo.outputSize = {side, side};
    createInfo.outputRegion = {{0, 0}, {side, side}};
    createInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    createInfo.sharpness = 0.5f;
    if (HMS_XEG_CreateSpatialUpscale(device, &createInfo, &spatialUpscale) != VK_SUCCESS) {
        spatialUpscale = VK_NULL_HANDLE;
        xegStatus = "HMS_XEG_CreateSpatialUpscale failed";
        xegSettled = true;
        return;
    }
    upscaleTarget = createImage2D(side, side, VK_FORMAT_R8G8B8A8_UNORM,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    upscaleSide = side;
    upscaleSceneSize = sceneSize;
    xegSettled = false;
    xegStatus = "active: GPU upscaling " + std::to_string(sceneSize) + " -> " + std::to_string(side);
}

void Renderer::Impl::recordSpatialUpscale(VkCommandBuffer command, uint32_t frame) {
    // The SDK dispatch records its own passes; give it a writable GENERAL target and publish a
    // sampled image afterwards. The scene input is already SHADER_READ_ONLY from the scene pass.
    VkImageMemoryBarrier prepare{};
    prepare.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    prepare.srcAccessMask = 0;
    prepare.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    prepare.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    prepare.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    prepare.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prepare.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prepare.image = upscaleTarget.image;
    prepare.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &prepare);
    XEG_SpatialUpscaleDescription description{};
    description.inputImage = scene[frame].image.view;
    description.outputImage = upscaleTarget.view;
    HMS_XEG_CmdRenderSpatialUpscale(command, spatialUpscale, &description);
    VkImageMemoryBarrier publish{};
    publish.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    publish.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    publish.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    publish.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    publish.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    publish.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    publish.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    publish.image = upscaleTarget.image;
    publish.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &publish);
}
#endif

Renderer::Renderer(const RendererOptions& options) : impl_(new Impl(options)) {}

Renderer::~Renderer() = default;

void Renderer::configure(const RenderConfig& config) {
    RenderConfig next = config;
    next.renderSize = clampU32(next.renderSize ? next.renderSize : 1024, kMinRenderSize, kMaxRenderSize);
    const bool sizeChanged = next.renderSize != impl_->sceneSize;
#if VOLUME_HMS
    const bool gtxSettingsChanged = sizeChanged || next.targetFps != impl_->config.targetFps;
#endif
    if (sizeChanged) {
        vkDeviceWaitIdle(impl_->device);
        impl_->destroySceneTargets();
        impl_->destroyStaging();
        impl_->config = next;
        impl_->createSceneTargets();
        for (auto& frame : impl_->frames) frame.drained = true; // Idle: query results already consumed.
        impl_->lastSubmitted = -1;                              // Old scene images are gone.
        impl_->hasLastFrameClock = false;                       // Avoid a reconfigure-gap FPS sample.
    } else {
        impl_->config = next;
    }
#if VOLUME_HMS
    const bool wantFg = next.aiFrameGeneration && impl_->surface != VK_NULL_HANDLE;
    if (wantFg != impl_->fgWanted || !wantFg) {
        vkDeviceWaitIdle(impl_->device);
        impl_->teardownFrameGeneration();
        impl_->fgWanted = wantFg;
        impl_->fgErrorLatch = false;
        impl_->fgStatus =
            wantFg ? "pending" : (next.aiFrameGeneration ? "disabled (offscreen benchmark)" : "disabled");
    } else {
        impl_->fgWanted = wantFg;
        impl_->fgErrorLatch = false; // configure() is the explicit user retry for a latched failure.
        if (impl_->fgActive) {
            HMS_FG_SetTargetFps_VK(impl_->fgContext,
                                   static_cast<int>(clampU32(static_cast<uint32_t>(next.targetFps), 30, 144)));
            impl_->fgStatus =
                "active: AI interpolation (system present, " + std::to_string(next.targetFps) + " fps target)";
        }
    }
    if (next.openGtx != impl_->gtxWanted || gtxSettingsChanged) {
        vkDeviceWaitIdle(impl_->device);
        impl_->teardownOpenGtx();
        impl_->gtxWanted = next.openGtx;
        if (impl_->gtxWanted) impl_->trySetupOpenGtx();
        else impl_->gtxStatus = "disabled";
    }
    if (next.xengineUpscale != impl_->xegWanted || (next.xengineUpscale && sizeChanged)) {
        vkDeviceWaitIdle(impl_->device);
        impl_->teardownSpatialUpscale();
        impl_->xegWanted = next.xengineUpscale;
        impl_->xegSettled = false;
        if (impl_->xegWanted) impl_->trySetupSpatialUpscale();
        else impl_->xegStatus = "disabled";
    }
#endif
}

void Renderer::resize(uint32_t width, uint32_t height) {
    if (impl_->surface == VK_NULL_HANDLE) return;
    impl_->pendingExtent.width = width;
    impl_->pendingExtent.height = height;
    impl_->swapDirty = true; // Rebuilt (with FG reset) at the next draw.
}

bool Renderer::draw(const ViewFrame& view, const CameraMatrices& matrices) {
    return impl_->drawFrame(view, matrices);
}

RenderStats Renderer::stats() const {
    // UI statistics must not force a GPU/CPU synchronization point every poll.
    if (impl_->lastSubmitted >= 0) {
        const auto frame = static_cast<uint32_t>(impl_->lastSubmitted);
        if (vkGetFenceStatus(impl_->device, impl_->frames[frame].rendered) == VK_SUCCESS) impl_->drainFrame(frame);
    }
    RenderStats stats;
    stats.realFps = impl_->fpsEma < 0.0 ? 0.0 : impl_->fpsEma;
    stats.gpuMilliseconds = impl_->gpuEmaMs; // -1 when timestamp queries are unavailable.
    stats.realFrames = impl_->realFrames;
    stats.renderSize = impl_->sceneSize;
    stats.deviceName = impl_->deviceName;
#if VOLUME_HMS
    stats.frameGenerationStatus = impl_->fgStatus;
    stats.openGtxStatus = impl_->gtxStatus;
    stats.xengineStatus = impl_->xegStatus;
    stats.thermalLevel = g_thermalLevel.load();
#else
    stats.frameGenerationStatus = impl_->surface == VK_NULL_HANDLE
                                      ? std::string("offscreen")
                                      : std::string("unavailable: requires HarmonyOS Graphics Accelerate Kit");
    stats.openGtxStatus = impl_->surface == VK_NULL_HANDLE
                              ? std::string("offscreen")
                              : std::string("unavailable: requires HarmonyOS Graphics Accelerate Kit");
    stats.xengineStatus = impl_->surface == VK_NULL_HANDLE
                              ? std::string("offscreen")
                              : std::string("unavailable: requires HarmonyOS XEngine Kit");
    stats.thermalLevel = 0;
#endif
    return stats;
}

std::vector<uint8_t> Renderer::capture() {
    std::vector<uint8_t> pixels;
    if (impl_->lastSubmitted < 0 || impl_->sceneSize == 0) return pixels;
    const uint32_t frame = static_cast<uint32_t>(impl_->lastSubmitted);
    impl_->drainFrame(frame); // The scene pass for this slot has completed; its image is intact.
    if (impl_->stagingBuffer == VK_NULL_HANDLE) impl_->createStaging();
    vkResetFences(impl_->device, 1, &impl_->captureFence);
    VkCommandBuffer command = impl_->captureCommand;
    vkResetCommandBuffer(command, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command, &beginInfo) != VK_SUCCESS) return pixels;
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = impl_->scene[frame].image.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {impl_->sceneSize, impl_->sceneSize, 1};
    vkCmdCopyImageToBuffer(command, impl_->scene[frame].image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           impl_->stagingBuffer, 1, &region);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) return pixels;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (vkQueueSubmit(impl_->queue, 1, &submit, impl_->captureFence) != VK_SUCCESS) return pixels;
    vkWaitForFences(impl_->device, 1, &impl_->captureFence, VK_TRUE, UINT64_MAX);
    if (!impl_->stagingCoherent) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = impl_->stagingMemory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(impl_->device, 1, &range);
    }
    pixels.resize(static_cast<size_t>(impl_->stagingSize));
    std::memcpy(pixels.data(), impl_->stagingMapped, static_cast<size_t>(impl_->stagingSize));
    return pixels;
}

} // namespace volume
