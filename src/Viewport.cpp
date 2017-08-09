
#include "Viewport.h"
#include "Binding.h"

extern uint64_t updateSerial;

void Viewport::WaitForChanges() {
    uint64_t nextSerial = lastUpdatedSerial;
    while (!shouldQuit && lastUpdatedSerial == (nextSerial = updateSerial)) {
        std::this_thread::yield();
    }
    lastUpdatedSerial = nextSerial;
}

void Viewport::Quit() {
    shouldQuit = true;
    _thread.join();
}

bool Viewport::ShouldQuit() const {
    return shouldQuit;
}

void Viewport::Initialize(const nxt::Device& device) {
    swapchain = GetSwapChain(device);
    swapchain.Configure(nxt::TextureFormat::R8G8B8A8Unorm, 640, 480);
}

void Viewport::Frame(Renderer* renderer) {
    nxt::Texture backbuffer = swapchain.GetNextTexture();
    renderer->Render(backbuffer);
    backbuffer.TransitionUsage(nxt::TextureUsageBit::Present);
    swapchain.Present(backbuffer);
    DoFlush();
}

void Viewport::Loop(Viewport* viewport, const nxt::Device& device, const nxt::Queue& queue, Renderer* renderer) {
    viewport->Initialize(device);

    while (!viewport->ShouldQuit()) {
        viewport->WaitForChanges();
        viewport->Frame(renderer);
    }
}
