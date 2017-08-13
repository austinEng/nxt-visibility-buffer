
#include "Binding.h"
#include "Globals.h"
#include "Viewport.h"
#include <GLFW/glfw3.h>

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

void Viewport::Initialize() {
    swapchain = GetSwapChain(globalDevice.Get());
    swapchain.Configure(nxt::TextureFormat::R8G8B8A8Unorm, 640, 480);
}

void Viewport::Frame(Renderer* renderer) {
    nxt::Texture backbuffer = swapchain.GetNextTexture();
    renderer->Render(backbuffer);
    backbuffer.TransitionUsage(nxt::TextureUsageBit::Present);
    globalDevice.Lock();
    swapchain.Present(backbuffer);
    DoFlush();
    globalDevice.Unlock();
}

void Viewport::Loop(Viewport* viewport, Renderer* renderer) {
    viewport->Initialize();

    while (!viewport->ShouldQuit()) {
        viewport->WaitForChanges();
        viewport->Frame(renderer);
    }
}
