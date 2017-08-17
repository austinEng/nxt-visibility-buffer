
#include <common/Platform.h>
#include <utils/BackendBinding.h>
#include <wire/TerribleCommandBuffer.h>

#include <nxt/nxt.h>
#include <nxt/nxtcpp.h>
#include <nxt/nxt_wsi.h>
#include <GLFW/glfw3.h>

#include <cstring>
#include <iostream>

void PrintDeviceError(const char* message, nxt::CallbackUserdata) {
    std::cout << "Device error: " << message << std::endl;
}

enum class CmdBufType {
    None,
    Terrible,
    //TODO(cwallez@chromium.org) double terrible cmdbuf
};

#if defined(NXT_ENABLE_BACKEND_D3D12)
    static utils::BackendType backendType = utils::BackendType::D3D12;
#elif defined(NXT_ENABLE_BACKEND_METAL)
    static utils::BackendType backendType = utils::BackendType::Metal;
#elif defined(NXT_ENABLE_BACKEND_OPENGL)
    static utils::BackendType backendType = utils::BackendType::OpenGL;
#elif defined(NXT_ENABLE_BACKEND_VULKAN)
    static utils::BackendType backendType = utils::BackendType::Vulkan;
#else
    #error
#endif

static CmdBufType cmdBufType = CmdBufType::Terrible;
static utils::BackendBinding* binding = nullptr;

static GLFWwindow* window = nullptr;

static nxt::wire::CommandHandler* wireServer = nullptr;
static nxt::wire::CommandHandler* wireClient = nullptr;
static nxt::wire::TerribleCommandBuffer* c2sBuf = nullptr;
static nxt::wire::TerribleCommandBuffer* s2cBuf = nullptr;

nxt::Device CreateCppNXTDevice() {
    binding = utils::CreateBinding(backendType);
    if (binding == nullptr) {
        return nxt::Device();
    }

    if (!glfwInit()) {
        return nxt::Device();
    }

    binding->SetupGLFWWindowHints();
    window = glfwCreateWindow(1280, 960, "NXT window", nullptr, nullptr);
    if (!window) {
        return nxt::Device();
    }

    binding->SetWindow(window);

    nxtDevice backendDevice;
    nxtProcTable backendProcs;
    binding->GetProcAndDevice(&backendProcs, &backendDevice);

    nxtDevice cDevice = nullptr;
    nxtProcTable procs;
    switch (cmdBufType) {
        case CmdBufType::None:
            procs = backendProcs;
            cDevice = backendDevice;
            break;

        case CmdBufType::Terrible:
            {
                c2sBuf = new nxt::wire::TerribleCommandBuffer();
                s2cBuf = new nxt::wire::TerribleCommandBuffer();

                wireServer = nxt::wire::NewServerCommandHandler(backendDevice, backendProcs, s2cBuf);
                c2sBuf->SetHandler(wireServer);

                nxtDevice clientDevice;
                nxtProcTable clientProcs;
                wireClient = nxt::wire::NewClientDevice(&clientProcs, &clientDevice, c2sBuf);
                s2cBuf->SetHandler(wireClient);

                procs = clientProcs;
                cDevice = clientDevice;
            }
            break;
    }

    nxtSetProcs(&procs);
    procs.deviceSetErrorCallback(cDevice, PrintDeviceError, 0);
    return nxt::Device::Acquire(cDevice);
}

nxt::Device CreateDevice() {
    binding = utils::CreateBinding(backendType);
    if (binding == nullptr) {
        return nxt::Device();
    }

    nxtDevice backendDevice;
    nxtProcTable backendProcs;
    binding->GetProcAndDevice(&backendProcs, &backendDevice);

    nxtDevice cDevice = nullptr;
    nxtProcTable procs;
    switch (cmdBufType) {
        case CmdBufType::None:
            procs = backendProcs;
            cDevice = backendDevice;
            break;

        case CmdBufType::Terrible:
            {
                c2sBuf = new nxt::wire::TerribleCommandBuffer();
                s2cBuf = new nxt::wire::TerribleCommandBuffer();

                wireServer = nxt::wire::NewServerCommandHandler(backendDevice, backendProcs, s2cBuf);
                c2sBuf->SetHandler(wireServer);

                nxtDevice clientDevice;
                nxtProcTable clientProcs;
                wireClient = nxt::wire::NewClientDevice(&clientProcs, &clientDevice, c2sBuf);
                s2cBuf->SetHandler(wireClient);

                procs = clientProcs;
                cDevice = clientDevice;
            }
            break;
    }

    nxtSetProcs(&procs);
    procs.deviceSetErrorCallback(cDevice, PrintDeviceError, 0);
    return nxt::Device::Acquire(cDevice);
}

uint64_t GetSwapChainImplementation() {
    return binding->GetSwapChainImplementation();
}

nxt::SwapChain GetSwapChain(const nxt::Device &device) {
    return device.CreateSwapChainBuilder()
        .SetImplementation(GetSwapChainImplementation())
        .GetResult();
}

bool InitializeBackend(int argc, const char** argv) {
    for (int i = 0; i < argc; i++) {
        if (std::string("-b") == argv[i] || std::string("--backend") == argv[i]) {
            i++;
            if (i < argc && std::string("d3d12") == argv[i]) {
                backendType = utils::BackendType::D3D12;
                continue;
            }
            if (i < argc && std::string("metal") == argv[i]) {
                backendType = utils::BackendType::Metal;
                continue;
            }
            if (i < argc && std::string("null") == argv[i]) {
                backendType = utils::BackendType::Null;
                continue;
            }
            if (i < argc && std::string("opengl") == argv[i]) {
                backendType = utils::BackendType::OpenGL;
                continue;
            }
            if (i < argc && std::string("vulkan") == argv[i]) {
                backendType = utils::BackendType::Vulkan;
                continue;
            }
            fprintf(stderr, "--backend expects a backend name (opengl, metal, d3d12, null, vulkan)\n");
            return false;
        }
        if (std::string("-c") == argv[i] || std::string("--command-buffer") == argv[i]) {
            i++;
            if (i < argc && std::string("none") == argv[i]) {
                cmdBufType = CmdBufType::None;
                continue;
            }
            if (i < argc && std::string("terrible") == argv[i]) {
                cmdBufType = CmdBufType::Terrible;
                continue;
            }
            fprintf(stderr, "--command-buffer expects a command buffer name (none, terrible)\n");
            return false;
        }
        if (std::string("-h") == argv[i] || std::string("--help") == argv[i]) {
            printf("Usage: %s [-b BACKEND] [-c COMMAND_BUFFER]\n", argv[0]);
            printf("  BACKEND is one of: d3d12, metal, null, opengl, vulkan\n");
            printf("  COMMAND_BUFFER is one of: none, terrible\n");
            return false;
        }
    }
    return true;
}

void DoFlush() {
    if (cmdBufType == CmdBufType::Terrible) {
        c2sBuf->Flush();
        s2cBuf->Flush();
    }
}

bool ShouldQuit() {
    return glfwWindowShouldClose(window);
}

GLFWwindow* GetGLFWWindow() {
    return window;
}
