
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
        .AttachmentSetFormat(0, nxt::TextureFormat::R8G8B8A8Uint)
        .AttachmentSetFormat(1, nxt::TextureFormat::D32FloatS8Uint)
        .SetSubpassCount(1)
        .SubpassSetColorAttachment(0, 0, 0)
        .SubpassSetDepthStencilAttachment(0, 1)
        .GetResult();

    gBufferTexture = device.CreateTextureBuilder()
        .SetDimension(nxt::TextureDimension::e2D)
        .SetExtent(640, 480, 1)
        .SetFormat(nxt::TextureFormat::R8G8B8A8Uint)
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

    cameraBufferView = cameraBuffer.CreateBufferViewBuilder()
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

            void main() {
                gl_Position = u_camera.viewProj * u_model.model * a_position;
            })");

        auto fsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Fragment, R"(
            #version 450

            layout(location = 0) out uvec4 visibilityBuffer;

            void main() {
                uint id3 = 0x80 | (0 & 0x7F);
                uint id2 = (gl_PrimitiveID >> 16) & 0xFF;
                uint id1 = (gl_PrimitiveID >> 8) & 0xFF;
                uint id0 = gl_PrimitiveID & 0xFF;
                visibilityBuffer = uvec4(id0, id1, id2, id3);
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

            layout(set = 2, binding = 0) uniform sampler gBufferSampler;
            layout(set = 2, binding = 1) uniform utexture2D gBuffer;

            layout(set = 2, binding = 2) buffer OutputBuffer {
                uint color;
            } fragColor[];
            
            layout(set = 2, binding = 3) buffer IndexBuffer {
                uint index;
            } indices[];

            layout(set = 2, binding = 4) buffer VertexBuffer {
                vec3 position;
            } vertices[];

            uint packColor(uvec4 color) {
                return (color.r) + (color.g << 8) + (color.b << 16) + (color.a << 24);
            }

            uint packColor(uint r, uint g, uint b, uint a) {
                return (r) + (g << 8) + (b << 16) + (a << 24);
            }

            void main() {
                uint outIndex = gl_GlobalInvocationID.x + 640 * gl_GlobalInvocationID.y; 
                uvec4 gBufferVal = texelFetch( usampler2D(gBuffer, gBufferSampler), ivec2(gl_GlobalInvocationID.xy), 0 );

                if ((gBufferVal.a & 0x80) == 0) {
                    if (mod(gl_GlobalInvocationID.x / 20 + gl_GlobalInvocationID.y / 20, 2) == 0) {
                        fragColor[outIndex].color = packColor(80, 80, 80, 255);
                    } else {
                        fragColor[outIndex].color = packColor(50, 50, 50, 255);
                    }
                    return;
                }                
                
                uint primID = packColor(uvec4(gBufferVal.rgb, 0));
                fragColor[outIndex].color = primID;
                
                uint i0 = indices[3 * primID + 0].index;
                uint i1 = indices[3 * primID + 1].index;
                uint i2 = indices[3 * primID + 2].index;
               
                vec4 p0_object = vec4(vertices[i0].position, 1);
                vec4 p1_object = vec4(vertices[i1].position, 1);
                vec4 p2_object = vec4(vertices[i2].position, 1);

                vec4 p0_world = u_model.model * p0_object;
                vec4 p1_world = u_model.model * p1_object;
                vec4 p2_world = u_model.model * p2_object;
                
                vec2 ndc = vec2(2.0, -2.0) * ((gl_GlobalInvocationID.xy / vec2(640.0, 480.0)) - vec2(0.5, 0.5));
                vec4 p0_screen = u_camera.viewProj * p0_world;
                vec4 p1_screen = u_camera.viewProj * p1_world;
                vec4 p2_screen = u_camera.viewProj * p2_world;
                
                vec2 p0_ndc = p0_screen.xy / p0_screen.w;
                vec2 p1_ndc = p1_screen.xy / p1_screen.w;
                vec2 p2_ndc = p2_screen.xy / p2_screen.w;
                
                vec2 v0 = p1_ndc - p0_ndc;
                vec2 v1 = p2_ndc - p0_ndc;
                vec2 v2 = ndc - p0_ndc;
                float d00 = dot(v0, v0);
                float d01 = dot(v0, v1);
                float d11 = dot(v1, v1);
                float d20 = dot(v2, v0);
                float d21 = dot(v2, v1);
                float denom = d00 * d11 - d01 * d01;
                float b1 = (d11 * d20 - d01 * d21) / denom;
                float b2 = (d00 * d21 - d01 * d20) / denom;
                float b0 = 1.0f - b1 - b2;

                float w = b0 / p0_screen.w + b1 / p1_screen.w + b2 / p2_screen.w;
                vec4 p_world = (b0 * (p0_world / p0_screen.w) + b1 * (p1_world / p1_screen.w) + b2 * (p2_world / p2_screen.w)) / w;

                vec3 N = normalize(cross(vec3(p1_world - p0_world), vec3(p2_world - p0_world)));

                // position, intensity
                const vec4 lights[2] = vec4[2](
                    vec4(1.0, 3.0, -2.0, 1.0),
                    vec4(-2.0, 1.0, 2.0, 0.2)
                );

                float diffuseTerm = 0;
                for (uint i = 0; i < 2; ++i) {
                    diffuseTerm += lights[i].w * max(dot(normalize(vec3(lights[i]) - p_world.xyz), N), 0);
                }

                // float diffuseTerm = max(0, dot(L1, N)) + max(0, dot(L2, N));
                float lightingTerm = 0.15 + 0.85 * diffuseTerm;
                fragColor[outIndex].color = packColor(uvec4(255 * vec4(vec3(lightingTerm), 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(vec3(b0, b1, b2), 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4((ndc + 1.0) * 0.5, 0, 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(vec3(p_world), 1)));
            })");

        auto computeOutputLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::StorageBuffer, 0, 1)
            .GetResult();

        computeBindGroupLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::Sampler, 0, 1) // gbuffer sampler
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::SampledTexture, 1, 1) // gbuffer texture
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::StorageBuffer, 2, 3) // output, index, vertex
            .GetResult();

        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, layout::cameraLayout)
            .SetBindGroupLayout(1, layout::modelLayout)
            .SetBindGroupLayout(2, computeBindGroupLayout)
            .GetResult();

        computeOutputBuffer = device.CreateBufferBuilder()
            .SetAllowedUsage(nxt::BufferUsageBit::TransferSrc | nxt::BufferUsageBit::Storage | nxt::BufferUsageBit::TransferDst)
            .SetSize(640 * 480 * 4)
            .GetResult();

        computeOutputBufferView = computeOutputBuffer.CreateBufferViewBuilder()
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
}

void Renderer::Render(nxt::Texture &texture) {
    auto backBufferView = texture.CreateTextureViewBuilder().GetResult();

    gBufferTexture.TransitionUsage(nxt::TextureUsageBit::OutputAttachment);
    auto framebuffer = device.CreateFramebufferBuilder()
        .SetRenderPass(renderpass)
        .SetDimensions(640, 480)
        .SetAttachment(0, gBufferView)
        .SetAttachment(1, depthStencilView)
        .GetResult();

    layout::camera_block cameraBlock { camera.GetViewProj() };
    cameraBuffer.TransitionUsage(nxt::BufferUsageBit::TransferDst);
    cameraBuffer.SetSubData(0, sizeof(layout::camera_block) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(&cameraBlock));
     
    auto commands = device.CreateCommandBufferBuilder();

    commands.TransitionBufferUsage(cameraBuffer, nxt::BufferUsageBit::Uniform);

    for (Model* model : scene.GetModels()) {
        commands.TransitionBufferUsage(model->GetUniformBuffer(), nxt::BufferUsageBit::Uniform);
        for (const RasterCommand& draw : model->GetCommands()) {
            commands.TransitionBufferUsage(draw.indexBuffer, nxt::BufferUsageBit::Index);
            commands.TransitionBufferUsage(draw.vertexBuffer, nxt::BufferUsageBit::Vertex);
        }
    }
    
    commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::Storage);
    commands.TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst);

    uint32_t drawID = 0;
    for (Model* model : scene.GetModels()) {
        for (const RasterCommand& draw : model->GetCommands()) {

            commands.BeginRenderPass(renderpass, framebuffer);
            commands.BeginRenderSubpass();

            commands.SetRenderPipeline(rasterizePipeline);
            commands.SetBindGroup(0, cameraBindGroup);

            auto modelBindGroup = device.CreateBindGroupBuilder()
                .SetUsage(nxt::BindGroupUsage::Frozen)
                .SetLayout(layout::modelLayout)
                .SetBufferViews(0, 1, &draw.uniformBufferView)
                .GetResult();
            commands.SetBindGroup(1, modelBindGroup);
            commands.SetVertexBuffers(0, 1, &draw.vertexBuffer, &draw.vertexBufferOffset);
            commands.SetIndexBuffer(draw.indexBuffer, draw.indexBufferOffset, nxt::IndexFormat::Uint32);
            commands.DrawElements(draw.count, 1, 0, 0);

            commands.EndRenderSubpass();
            commands.EndRenderPass();

            commands.TransitionBufferUsage(draw.indexBuffer, nxt::BufferUsageBit::Storage);
            commands.TransitionBufferUsage(draw.vertexBuffer, nxt::BufferUsageBit::Storage);
            commands.TransitionTextureUsage(gBufferTexture, nxt::TextureUsageBit::Sampled);

            nxt::BufferView computeBufferViews[] = {
                computeOutputBufferView.Clone(),
                draw.indexBufferView.Clone(),
                draw.vertexBufferView.Clone(),
            };

            auto sampler = device.CreateSamplerBuilder()
                .SetFilterMode(nxt::FilterMode::Nearest, nxt::FilterMode::Nearest, nxt::FilterMode::Nearest)
                .GetResult();

            auto computeBindGroup = device.CreateBindGroupBuilder()
                .SetUsage(nxt::BindGroupUsage::Frozen)
                .SetLayout(computeBindGroupLayout)
                .SetSamplers(0, 1, &sampler)
                .SetTextureViews(1, 1, &gBufferView)
                .SetBufferViews(2, 3, computeBufferViews)
                .GetResult();

            commands.BeginComputePass();
            commands.SetComputePipeline(shadingPipeline);
            commands.SetBindGroup(0, cameraBindGroup);
            commands.SetBindGroup(1, modelBindGroup);
            commands.SetBindGroup(2, computeBindGroup);
            commands.Dispatch(640, 480, 1);
            commands.EndComputePass();

            drawID++;
        }
    }

    commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::TransferSrc);
    commands.CopyBufferToTexture(computeOutputBuffer, 0, 0, texture, 0, 0, 0, 640, 480, 1, 0);

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
