#pragma once

#include <thread>
#include <nxt/nxtcpp.h>

#include "Camera.h"

class ViewportThread : public std::thread {
    public:
        ViewportThread(const nxt::Device& device, const nxt::Queue& queue, const Camera& camera);

        void WaitForChanges(const Camera& camera);
        void Quit();
        bool ShouldQuit() const;

    private:
        nxt::SwapChain swapchain;
        uint64_t lastCameraSerial = 0;
        bool shouldQuit = false;

        void Initialize(const nxt::Device& device);
        void Frame();
        static void Loop(ViewportThread* viewport, const nxt::Device& device, const nxt::Queue& queue, const Camera& camera);
};
