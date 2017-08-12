
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#define TINYGLTF_LOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <tinygltfloader/tiny_gltf_loader.h>

#include <utils/SystemUtils.h>
#include <utils/NXTHelpers.h>
#include "Binding.h"
#include "Camera.h"
#include "Renderer.h"
#include "Scene.h"
#include "Globals.h"
#include "Viewport.h"

uint64_t updateSerial = 1;
nxt::Device device;
nxt::Queue queue;

namespace {
    Camera camera;
    Scene* scenePtr;

    bool buttons[GLFW_MOUSE_BUTTON_LAST + 1] = { 0 };

    void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
        buttons[button] = (action == GLFW_PRESS);
    }

    void cursorPosCallback(GLFWwindow*, double mouseX, double mouseY) {
        static double oldX, oldY;
        float dX = static_cast<float>(mouseX - oldX);
        float dY = static_cast<float>(mouseY - oldY);
        oldX = mouseX;
        oldY = mouseY;

        if (buttons[2] || (buttons[0] && buttons[1])) {
            camera.Pan(-dX * 0.002f, dY * 0.002f);
            updateSerial++;
        }
        else if (buttons[0]) {
            camera.Rotate(dX * -0.01f, dY * 0.01f);
            updateSerial++;
        }
        else if (buttons[1]) {
            camera.Zoom(dY * -0.005f);
            updateSerial++;
        }
    }

    void scrollCallback(GLFWwindow*, double, double yoffset) {
        camera.Zoom(static_cast<float>(yoffset) * 0.04f);
        updateSerial++;
    }

    void dropCallback(GLFWwindow*, int count, const char** paths) {
        for (int i = 0; i < count; ++i) {
            scenePtr->AddModel(std::string(paths[i]));
        }
    }
}

namespace layout {
    nxt::BindGroupLayout cameraLayout;
    nxt::BindGroupLayout modelLayout;
}

namespace default {
    nxt::TextureView defaultDiffuse;
    nxt::TextureView defaultNormal;
    nxt::TextureView defaultSpecular;
    nxt::Sampler defaultSampler;
}

void frame(const nxt::SwapChain& swapchain) {
    nxt::Texture backbuffer = swapchain.GetNextTexture();

    backbuffer.TransitionUsage(nxt::TextureUsageBit::Present);
    swapchain.Present(backbuffer);
    DoFlush();
}

void init() {
    layout::cameraLayout = device.CreateBindGroupLayoutBuilder()
        .SetBindingsType(nxt::ShaderStageBit::Vertex | nxt::ShaderStageBit::Compute, nxt::BindingType::UniformBuffer, 0, 1)
        .GetResult();

    layout::modelLayout = device.CreateBindGroupLayoutBuilder()
        .SetBindingsType(nxt::ShaderStageBit::Vertex | nxt::ShaderStageBit::Compute, nxt::BindingType::UniformBuffer, 0, 1)
        .GetResult();

    {
        auto texture = device.CreateTextureBuilder()
            .SetDimension(nxt::TextureDimension::e2D)
            .SetExtent(1, 1, 1)
            .SetFormat(nxt::TextureFormat::R8G8B8A8Unorm)
            .SetMipLevels(1)
            .SetAllowedUsage(nxt::TextureUsageBit::TransferDst | nxt::TextureUsageBit::Sampled)
            .GetResult();

        uint32_t white = 0xffffffff;
        nxt::Buffer staging = utils::CreateFrozenBufferFromData(device, &white, sizeof(white), nxt::BufferUsageBit::TransferSrc);
        auto cmdbuf = device.CreateCommandBufferBuilder()
            .TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst)
            .CopyBufferToTexture(staging, 0, 256, texture, 0, 0, 0, 1, 1, 1, 0)
            .GetResult();
        queue.Submit(1, &cmdbuf);
        texture.FreezeUsage(nxt::TextureUsageBit::Sampled);

        default::defaultDiffuse = texture.CreateTextureViewBuilder().GetResult();
    }

    {
        auto texture = device.CreateTextureBuilder()
            .SetDimension(nxt::TextureDimension::e2D)
            .SetExtent(1, 1, 1)
            .SetFormat(nxt::TextureFormat::R8G8B8A8Unorm)
            .SetMipLevels(1)
            .SetAllowedUsage(nxt::TextureUsageBit::TransferDst | nxt::TextureUsageBit::Sampled)
            .GetResult();

        uint32_t up = 0x0000ff00;
        nxt::Buffer staging = utils::CreateFrozenBufferFromData(device, &up, sizeof(up), nxt::BufferUsageBit::TransferSrc);
        auto cmdbuf = device.CreateCommandBufferBuilder()
            .TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst)
            .CopyBufferToTexture(staging, 0, 256, texture, 0, 0, 0, 1, 1, 1, 0)
            .GetResult();
        queue.Submit(1, &cmdbuf);
        texture.FreezeUsage(nxt::TextureUsageBit::Sampled);

        default::defaultNormal = texture.CreateTextureViewBuilder().GetResult();
    }

    {
        auto texture = device.CreateTextureBuilder()
            .SetDimension(nxt::TextureDimension::e2D)
            .SetExtent(1, 1, 1)
            .SetFormat(nxt::TextureFormat::R8G8B8A8Unorm)
            .SetMipLevels(1)
            .SetAllowedUsage(nxt::TextureUsageBit::TransferDst | nxt::TextureUsageBit::Sampled)
            .GetResult();

        uint32_t black = 0x00000000;
        nxt::Buffer staging = utils::CreateFrozenBufferFromData(device, &black, sizeof(black), nxt::BufferUsageBit::TransferSrc);
        auto cmdbuf = device.CreateCommandBufferBuilder()
            .TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst)
            .CopyBufferToTexture(staging, 0, 256, texture, 0, 0, 0, 1, 1, 1, 0)
            .GetResult();
        queue.Submit(1, &cmdbuf);
        texture.FreezeUsage(nxt::TextureUsageBit::Sampled);

        default::defaultSpecular = texture.CreateTextureViewBuilder().GetResult();
    }

    default::defaultSampler = device.CreateSamplerBuilder().SetFilterMode(nxt::FilterMode::Linear, nxt::FilterMode::Linear, nxt::FilterMode::Linear).GetResult();
}

int main(int argc, const char* argv[]) {
    if (!InitializeBackend(argc, argv)) {
        return 1;
    }

    device = CreateCppNXTDevice();
    queue = device.CreateQueueBuilder().GetResult();

    init();

    Scene scene(device, queue);
    scenePtr = &scene;

    GLFWwindow* window = GetGLFWWindow();
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetDropCallback(window, dropCallback);

    Renderer* renderer = new Renderer(device, queue, camera, scene);

    Viewport* viewport = new Viewport(device, queue, renderer);

    for (int i = 1; i < argc; ++i) {
        scenePtr->AddModel(std::string(argv[i]));
    }

    while (!ShouldQuit()) {
        glfwPollEvents();
        utils::USleep(16000);
    }

    viewport->Quit();
    renderer->Quit();
    delete viewport;
    delete renderer;
}
