
#include <array>
#include <utils/NXTHelpers.h>
#include "Renderer.h"
#include "Layouts.h"

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
        .SetLayout(layout::cameraLayout)
        .SetBufferViews(0, 1, &cameraBufferView)
        .SetUsage(nxt::BindGroupUsage::Frozen)
        .GetResult();

    {
        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, layout::cameraLayout)
            .SetBindGroupLayout(1, layout::modelLayout)
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

    {
        auto csModule = utils::CreateShaderModule(device, nxt::ShaderStage::Compute, R"(
            #version 450

            layout(set = 0, binding = 0) uniform u_camera_block {
                mat4 viewProj;
            } u_camera;

            layout(set = 1, binding = 0) uniform u_model_block {
                mat4 model;
            } u_model;

            layout(set = 2, binding = 0) buffer VertexBuffer {
                vec3 position;
            } vertices[];

            layout(set = 2, binding = 1) buffer IndexBuffer {
                uint index;
            } indicies[];

            layout(set = 3, binding = 0) buffer OutputBuffer {
                vec4 color;
            } fragColor[];

            void main() {
                uint index = gl_GlobalInvocationID.x + 640 * gl_GlobalInvocationID.y;
                fragColor[index].color = vec4(255, 255, 255, 255);
                // output[index].color.r = index & 0xff;
                // output[index].color.g = (index >> 8) & 0xff;
                // output[index].color.b = (index >> 16) & 0xff;
                // output[index].color.a = 255;
            })");

        auto computeOutputLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::StorageBuffer, 0, 1)
            .GetResult();

        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, layout::cameraLayout)
            .SetBindGroupLayout(1, layout::modelLayout)
            .SetBindGroupLayout(2, layout::computeBufferLayout)
            .SetBindGroupLayout(3, computeOutputLayout)
            .GetResult();

        computeOutputBuffer = device.CreateBufferBuilder()
            .SetAllowedUsage(nxt::BufferUsageBit::TransferSrc | nxt::BufferUsageBit::Storage | nxt::BufferUsageBit::TransferDst)
            .SetSize(640 * 480 * 4)
            .GetResult();

        auto computeOutputBufferView = computeOutputBuffer.CreateBufferViewBuilder()
            .SetExtent(0, 640 * 480 * 4)
            .GetResult();

        computeOutputBindGroup = device.CreateBindGroupBuilder()
            .SetLayout(computeOutputLayout)
            .SetBufferViews(0, 1, &computeOutputBufferView)
            .SetUsage(nxt::BindGroupUsage::Frozen)
            .GetResult();

        shadingPipeline = device.CreateComputePipelineBuilder()
            .SetStage(nxt::ShaderStage::Compute, csModule, "main")
            .SetLayout(pipelineLayout)
            .GetResult();
    }

    {
        outputTexture = device.CreateTextureBuilder()
            .SetAllowedUsage(nxt::TextureUsageBit::Sampled | nxt::TextureUsageBit::TransferDst)
            .SetDimension(nxt::TextureDimension::e2D)
            .SetFormat(nxt::TextureFormat::R8G8B8A8Unorm)
            .SetMipLevels(1)
            .SetExtent(640, 480, 1)
            .GetResult();

        copyBindGroupLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Fragment, nxt::BindingType::Sampler, 0, 1)
            .SetBindingsType(nxt::ShaderStageBit::Fragment, nxt::BindingType::SampledTexture, 1, 1)
            .GetResult();

        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, copyBindGroupLayout)
            .GetResult();

        auto vsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Vertex, R"(
            #version 450

            layout(location = 0) out vec2 uv;

            void main() {
                const vec2 pos[6] = vec2[6](vec2(-1.f, -1.f), vec2(1.f, -1.f), vec2(-1.f, 1.f), vec2(1.f, 1.f), vec2(-1.f, 1.f), vec2(1.f, -1.f));

                gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
                uv = gl_Position.xy * 0.5 + 0.5;
            })");

        auto fsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Fragment, R"(
            #version 450
            
            // layout(set = 0, binding = 0) uniform sampler mySampler;
            // layout(set = 0, binding = 1) uniform texture2D myTexture;            

            layout(location = 0) in vec2 uv;

            out vec4 fragColor;

            void main() {
                fragColor = vec4(1,0,0,1); // texture(sampler2D(myTexture, mySampler), uv);
            })");

        copyOutputPipeline = device.CreateRenderPipelineBuilder()
            .SetSubpass(renderpass, 0)
            // .SetLayout(pipelineLayout)
            .SetStage(nxt::ShaderStage::Vertex, vsModule, "main")
            .SetStage(nxt::ShaderStage::Fragment, fsModule, "main")
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

    layout::camera_block cameraBlock { camera.GetViewProj() };
    cameraBuffer.TransitionUsage(nxt::BufferUsageBit::TransferDst);
    cameraBuffer.SetSubData(0, sizeof(layout::camera_block) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(&cameraBlock));
     
    auto data = new std::array<uint8_t, 640 * 480 * 4>();
    data->fill(255);
    computeOutputBuffer.TransitionUsage(nxt::BufferUsageBit::TransferDst);
    computeOutputBuffer.SetSubData(0, sizeof(data) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(data->data()));
    delete data;

    auto commands = device.CreateCommandBufferBuilder();
    // commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::TransferSrc);
    // commands.TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst);
    // commands.CopyBufferToTexture(computeOutputBuffer, 0, 0, texture, 0, 0, 0, 640, 480, 1, 0);
    commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::TransferSrc);
    commands.TransitionTextureUsage(outputTexture, nxt::TextureUsageBit::TransferDst);
    commands.CopyBufferToTexture(computeOutputBuffer, 0, 0, outputTexture, 0, 0, 0, 640, 480, 1, 0);

    auto outputSampler = device.CreateSamplerBuilder().SetFilterMode(nxt::FilterMode::Nearest, nxt::FilterMode::Nearest, nxt::FilterMode::Nearest).GetResult();
    auto outputView = outputTexture.CreateTextureViewBuilder().GetResult();
    auto copyBindGroup = device.CreateBindGroupBuilder()
        .SetLayout(copyBindGroupLayout)
        .SetUsage(nxt::BindGroupUsage::Frozen)
        .SetSamplers(0, 1, &outputSampler)
        .SetTextureViews(1, 1, &outputView)
        .GetResult();

    commands.TransitionTextureUsage(outputTexture, nxt::TextureUsageBit::Sampled);

    commands.BeginRenderPass(renderpass, framebuffer);
    commands.BeginRenderSubpass();
    
    commands.SetRenderPipeline(copyOutputPipeline);
    commands.SetBindGroup(0, copyBindGroup);
    commands.DrawArrays(6, 1, 0, 0);
    
    commands.EndRenderSubpass();
    commands.EndRenderPass();

    /*commands.TransitionBufferUsage(cameraBuffer, nxt::BufferUsageBit::Uniform);

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
    commands.EndRenderPass();*/
    
    /*commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::Storage);
    commands.TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst);

    for (Model* model : scene.GetModels()) {
        for (const RasterCommand& draw : model->GetRasterCommands()) {

            commands.BeginRenderPass(renderpass, framebuffer);
            commands.BeginRenderSubpass();

            commands.SetRenderPipeline(rasterizePipeline);
            commands.SetBindGroup(0, cameraBindGroup);

            commands.SetBindGroup(1, draw.uniforms);
            commands.SetVertexBuffers(0, 1, &draw.vertexBuffer, &draw.vertexBufferOffset);
            commands.SetIndexBuffer(draw.indexBuffer, draw.indexBufferOffset, nxt::IndexFormat::Uint16);
            commands.DrawElements(draw.count, 1, 0, 0);

            commands.EndRenderSubpass();
            commands.EndRenderPass();

            commands.TransitionBufferUsage(draw.indexBuffer, nxt::BufferUsageBit::Storage);
            commands.TransitionBufferUsage(draw.vertexBuffer, nxt::BufferUsageBit::Storage);

            commands.BeginComputePass();
            commands.SetComputePipeline(shadingPipeline);
            commands.SetBindGroup(2, draw.storage);
            commands.SetBindGroup(3, computeOutputBindGroup);
            // commands.Dispatch(640, 480, 1);
            commands.EndComputePass();
        }
    }

    commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::TransferSrc);
    commands.CopyBufferToTexture(computeOutputBuffer, 0, 0, texture, 0, 0, 0, 640, 480, 1, 0);*/

    auto cmds = commands.GetResult();
    queue.Submit(1, &cmds);
}

void Renderer::Quit() {
    shouldQuit = true;
    _thread.join();
}

bool Renderer::ShouldQuit() const {
    return shouldQuit;
}
