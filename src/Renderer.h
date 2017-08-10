
#pragma once

#include <nxt/nxtcpp.h>
#include <thread>
#include "Camera.h"
#include "Scene.h"

class Renderer {

    public:
        Renderer(const nxt::Device &device, const nxt::Queue &queue, const Camera& camera, const Scene& scene);
        Renderer(const Renderer &other) = delete;
        Renderer& operator=(const Renderer &other) = delete;

        void Render(nxt::Texture &texture);
        void Quit();
        bool ShouldQuit() const;

    private:
        nxt::Device device;
        nxt::Queue queue;
        const Scene& scene;
        const Camera& camera;
        std::thread _thread;
        bool shouldQuit = false;

        nxt::RenderPass renderpass;
        nxt::TextureView gBufferView;
        nxt::TextureView depthStencilView;
        nxt::Buffer cameraBuffer;
        nxt::BindGroup cameraBindGroup;
        nxt::RenderPipeline rasterizePipeline;
        nxt::ComputePipeline shadingPipeline;
        nxt::RenderPipeline copyOutputPipeline;
        nxt::Buffer computeOutputBuffer;
        nxt::BindGroup computeOutputBindGroup;
        nxt::Texture outputTexture;
        nxt::BindGroupLayout copyBindGroupLayout;
};
