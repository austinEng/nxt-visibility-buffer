
#include <utils/NXTHelpers.h>
#include "Renderer.h"
#include "Uniforms.h"

namespace {
    void Loop(Renderer* renderer) {
        // Wait for scene changes and cull stuff?
        std::this_thread::yield();
    }

    struct u_camera_block {
        glm::mat4 viewProj;
    };

    struct u_model_block {
        glm::mat4 model;
    };
}

Renderer::Renderer(const nxt::Device &device, const nxt::Queue &queue, const Camera& camera, const Scene& scene)
    : device(device.Clone()), queue(queue.Clone()), scene(scene), camera(camera), _thread(Loop, this) {

    renderpass = device.CreateRenderPassBuilder()
        .SetAttachmentCount(2)
        .AttachmentSetFormat(0, nxt::TextureFormat::R8G8B8A8Unorm)
        .AttachmentSetFormat(1, nxt::TextureFormat::D32FloatS8Uint)
        .SetSubpassCount(1)
        .SubpassSetColorAttachment(0, 0, 0)
        .SubpassSetDepthStencilAttachment(0, 1)
        .GetResult();

    auto gBufferTexture = device.CreateTextureBuilder()
        .SetDimension(nxt::TextureDimension::e2D)
        .SetExtent(640, 480, 1)
        .SetFormat(nxt::TextureFormat::R8G8B8A8Unorm)
        .SetMipLevels(1)
        .SetAllowedUsage(nxt::TextureUsageBit::OutputAttachment | nxt::TextureUsageBit::Sampled)
        .GetResult();

    gBufferView = gBufferTexture.CreateTextureViewBuilder().GetResult();

    auto depthStencilTexture = device.CreateTextureBuilder()
        .SetDimension(nxt::TextureDimension::e2D)
        .SetExtent(640, 480, 1)
        .SetFormat(nxt::TextureFormat::D32FloatS8Uint)
        .SetMipLevels(1)
        .SetAllowedUsage(nxt::TextureUsageBit::OutputAttachment)
        .GetResult();
    depthStencilTexture.FreezeUsage(nxt::TextureUsageBit::OutputAttachment);

    depthStencilView = depthStencilTexture.CreateTextureViewBuilder().GetResult();

    cameraBuffer = device.CreateBufferBuilder()
        .SetAllowedUsage(nxt::BufferUsageBit::Uniform | nxt::BufferUsageBit::TransferDst)
        .SetSize(sizeof(u_camera_block))
        .GetResult();

    auto cameraBufferView = cameraBuffer.CreateBufferViewBuilder()
        .SetExtent(0, sizeof(u_camera_block))
        .GetResult();

    cameraBindGroup = device.CreateBindGroupBuilder()
        .SetLayout(uniform::cameraLayout)
        .SetBufferViews(0, 1, &cameraBufferView)
        .SetUsage(nxt::BindGroupUsage::Frozen)
        .GetResult();
    
    {
        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, uniform::cameraLayout)
            .SetBindGroupLayout(1, uniform::modelLayout)
            .GetResult();

        auto depthStencilState = device.CreateDepthStencilStateBuilder()
            .SetDepthCompareFunction(nxt::CompareFunction::Less)
            .SetDepthWriteEnabled(true)
            .GetResult();

        auto inputState = device.CreateInputStateBuilder()
            .SetAttribute(0, 0, nxt::VertexFormat::FloatR32G32B32, 0)
            .SetInput(0, 12, nxt::InputStepMode::Vertex)
            .GetResult();

        auto vsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Vertex, R"(
            #version 450

            layout(set = 0, binding = 0) uniform u_camera_block {
                mat4 viewProj;
            } u_camera;

            layout(set = 1, binding = 0) uniform u_model_block {
                mat4 model;
            } u_model;

            layout(location = 0) in vec4 a_position;
        
            layout(location = 0) out flat uint primitiveID;

            void main() {
                gl_Position = u_camera.viewProj * u_model.model * a_position;
                primitiveID = 3 * int(gl_VertexIndex / 3);
            })");

        auto fsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Fragment, R"(
            #version 450
        
            layout(location = 0) flat in uint primitiveID;

            layout(location = 0) out vec4 visibilityBuffer;     

            void main() {
                uint id2 = (primitiveID >> 16) & 0xFF;
                uint id1 = (primitiveID >> 8) & 0xFF;
                uint id0 = primitiveID & 0xFF;
                visibilityBuffer = vec4(float(id2) / 255.0, float(id1) / 255.0, float(id0) / 255.0, 1.0);
            })");

        rasterizePipeline = device.CreateRenderPipelineBuilder()
            .SetSubpass(renderpass, 0)
            .SetStage(nxt::ShaderStage::Vertex, vsModule, "main")
            .SetStage(nxt::ShaderStage::Fragment, fsModule, "main")
            .SetLayout(pipelineLayout)
            .SetDepthStencilState(depthStencilState)
            .SetInputState(inputState)
            .SetPrimitiveTopology(nxt::PrimitiveTopology::TriangleList)
            .GetResult();
    }
}

void Renderer::Render(nxt::Texture &texture) {
    auto backBufferView = texture.CreateTextureViewBuilder().GetResult();

    auto framebuffer = device.CreateFramebufferBuilder()
        .SetRenderPass(renderpass)
        .SetDimensions(640, 480)
        .SetAttachment(0, backBufferView)
        .SetAttachment(1, depthStencilView)
        .GetResult();
    
    uniform::camera_block cameraBlock { camera.GetViewProj() };
    cameraBuffer.TransitionUsage(nxt::BufferUsageBit::TransferDst);
    cameraBuffer.SetSubData(0, sizeof(uniform::camera_block) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(&cameraBlock));

    auto commands = device.CreateCommandBufferBuilder();
    commands.TransitionBufferUsage(cameraBuffer, nxt::BufferUsageBit::Uniform);
    for (Model* model : scene.GetModels()) {
        commands.TransitionBufferUsage(model->GetUniformBuffer(), nxt::BufferUsageBit::Uniform);
        for (const RasterCommand& command : model->GetRasterCommands()) {
            commands.TransitionBufferUsage(command.indexBuffer, nxt::BufferUsageBit::Index);
            commands.TransitionBufferUsage(command.vertexBuffer, nxt::BufferUsageBit::Vertex);
        }
    }

    commands.BeginRenderPass(renderpass, framebuffer);
    commands.BeginRenderSubpass();

    commands.SetRenderPipeline(rasterizePipeline);
    commands.SetBindGroup(0, cameraBindGroup);
    for (Model* model : scene.GetModels()) {
        for (const RasterCommand& command : model->GetRasterCommands()) {
            commands.SetBindGroup(1, command.uniforms);
            commands.SetVertexBuffers(0, 1, &command.vertexBuffer, &command.vertexBufferOffset);
            commands.SetIndexBuffer(command.indexBuffer, command.indexBufferOffset, nxt::IndexFormat::Uint16);
            commands.DrawElements(command.count, 1, 0, 0);
        }
    }

    commands.EndRenderSubpass();
    commands.EndRenderPass();

    {
        auto cmds = commands.GetResult();
        queue.Submit(1, &cmds);
    }
}

void Renderer::Quit() {
    shouldQuit = true;
    _thread.join();
}

bool Renderer::ShouldQuit() const {
    return shouldQuit;
}
