
#include <thread>
#include "ViewportThread.h"
#include "Binding.h"

ViewportThread::ViewportThread(const nxt::Device& device, const nxt::Queue& queue, const Camera& camera)
    : std::thread(Loop, this, std::ref(device), std::ref(queue), std::ref(camera)) {
}

void ViewportThread::WaitForChanges(const Camera& camera) {
    uint64_t nextCameraSerial = lastCameraSerial;
    while (!shouldQuit && lastCameraSerial == (nextCameraSerial = camera.GetSerial())) {
        std::this_thread::yield();
    }
    lastCameraSerial = nextCameraSerial;
}

void ViewportThread::Quit() {
    shouldQuit = true;
}

bool ViewportThread::ShouldQuit() const {
    return shouldQuit;
}

void ViewportThread::Initialize(const nxt::Device& device) {
    swapchain = GetSwapChain(device);
    swapchain.Configure(nxt::TextureFormat::R8G8B8A8Unorm, 640, 480);
}

void ViewportThread::Frame() {
    printf("Drawing!\n");
    nxt::Texture backbuffer = swapchain.GetNextTexture();

    backbuffer.TransitionUsage(nxt::TextureUsageBit::Present);
    swapchain.Present(backbuffer);
    DoFlush();
}

void ViewportThread::Loop(ViewportThread* viewport, const nxt::Device& device, const nxt::Queue& queue, const Camera& camera) {
    viewport->Initialize(device);

    while (!viewport->ShouldQuit()) {
        viewport->WaitForChanges(camera);
        viewport->Frame();
    }
}
