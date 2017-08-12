
#pragma once

#include <nxt/nxtcpp.h>
#include <thread>
#include "Camera.h"
#include "Scene.h"

class Renderer {

    public:
        Renderer(const Camera& camera, const Scene& scene);
        Renderer(const Renderer &other) = delete;
        Renderer& operator=(const Renderer &other) = delete;

        void Render(nxt::Texture &texture);
        void Quit();
        bool ShouldQuit() const;

    private:
        const Scene& scene;
        const Camera& camera;
        std::thread _thread;
        bool shouldQuit = false;

        nxt::RenderPass renderpass;
        nxt::Texture gBufferTexture;
        nxt::TextureView gBufferView;
        nxt::TextureView depthStencilView;
        nxt::Buffer cameraBuffer;
        nxt::BufferView cameraBufferView;
        nxt::BindGroup cameraBindGroup;
        nxt::BindGroupLayout constantsBindGroupLayout;
        nxt::RenderPipeline rasterizePipeline;
        nxt::ComputePipeline shadingPipeline;
        nxt::BindGroupLayout modelBindGroupLayout;
        nxt::BindGroupLayout computeBindGroupLayout;
        nxt::Buffer computeOutputBuffer;
        nxt::BufferView computeOutputBufferView;
        nxt::BindGroup computeOutputBindGroup;

        std::vector<nxt::BindGroup> constantBindGroups;
};
