#pragma once

#include <thread>
#include <nxt/nxtcpp.h>

#include "Camera.h"
#include "Renderer.h"

class Viewport {
    public:
        template<typename... Args>
        Viewport(Args&... args) : _thread(Loop, this, std::forward<Args>(args)...) { }
        Viewport(const Viewport &other) = delete;
        Viewport& operator=(const Viewport &other) = delete;
        Viewport(Viewport&& other) : _thread(std::move(other._thread)) { }
        Viewport& operator=(Viewport&& other) {
            _thread = std::move(other._thread);
        }

        void WaitForChanges();
        void Quit();
        bool ShouldQuit() const;

    private:
        std::thread _thread;
        nxt::SwapChain swapchain;
        uint64_t lastUpdatedSerial = 0;
        bool shouldQuit = false;

        void Initialize(const nxt::Device& device);
        void Frame(Renderer* renderer);
        static void Loop(Viewport* viewport, const nxt::Device& device, const nxt::Queue& queue, Renderer* renderer);
};
