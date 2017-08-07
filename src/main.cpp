
#include <GLFW/glfw3.h>
#include <utils/SystemUtils.h>
#include "Binding.h"

void frame(const nxt::SwapChain& swapchain) {
    nxt::Texture backbuffer = swapchain.GetNextTexture();

    backbuffer.TransitionUsage(nxt::TextureUsageBit::Present);
    swapchain.Present(backbuffer);
    DoFlush();
}

int main(int argc, const char* argv[]) {
    if (!InitializeBackend(argc, argv)) {
        return 1;
    }

    nxt::Device device = CreateCppNXTDevice();
    nxt::Queue queue = device.CreateQueueBuilder().GetResult();
    nxt::SwapChain swapchain = GetSwapChain(device);
    swapchain.Configure(nxt::TextureFormat::R8G8B8A8Unorm, 640, 480);

    while (!ShouldQuit()) {
        frame(swapchain);
        glfwPollEvents();
        utils::USleep(16000);
    }
}
