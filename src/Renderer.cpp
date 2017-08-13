
#include <array>
#include <utils/NXTHelpers.h>
#include "Renderer.h"
#include "Globals.h"

namespace {
    void Loop(Renderer* renderer) {
        // Wait for scene changes and cull stuff?
        std::this_thread::yield();
    }
}

Renderer::Renderer(const Camera& camera, Scene& scene)
    : scene(scene), camera(camera), _thread(Loop, this) {

    auto& device = globalDevice.Lock();

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
        .SetSize(sizeof(layout::camera_block))
        .GetResult();

    cameraBufferView = cameraBuffer.CreateBufferViewBuilder()
        .SetExtent(0, sizeof(layout::camera_block))
        .GetResult();

    cameraBindGroup = device.CreateBindGroupBuilder()
        .SetLayout(layout::cameraLayout)
        .SetBufferViews(0, 1, &cameraBufferView)
        .SetUsage(nxt::BindGroupUsage::Frozen)
        .GetResult();

    constantsBindGroupLayout = device.CreateBindGroupLayoutBuilder()
        .SetBindingsType(nxt::ShaderStageBit::Fragment | nxt::ShaderStageBit::Compute, nxt::BindingType::UniformBuffer, 0, 1)
        .GetResult();

    {
        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, layout::cameraLayout)
            .SetBindGroupLayout(1, layout::modelLayout)
            .SetBindGroupLayout(2, constantsBindGroupLayout)
            .GetResult();

        auto depthStencilState = device.CreateDepthStencilStateBuilder()
            .SetDepthCompareFunction(nxt::CompareFunction::Less)
            .SetDepthWriteEnabled(true)
            .GetResult();

        auto inputState = device.CreateInputStateBuilder()
            .SetAttribute(0, 0, nxt::VertexFormat::FloatR32G32B32, 0)
            .SetInput(0, sizeof(Model::Vertex), nxt::InputStepMode::Vertex)
            .GetResult();

        auto vsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Vertex, R"(
            #version 450

            layout(set = 0, binding = 0) uniform u_camera_block {
                mat4 viewProj;
                vec3 eye;
            } u_camera;

            layout(set = 1, binding = 0) uniform u_model_block {
                mat4 model;
                mat4 modelInv;
            } u_model;

            layout(location = 0) in vec3 a_position;

            void main() {
                gl_Position = u_camera.viewProj * u_model.model * vec4(a_position, 1);
            })");

        auto fsModule = utils::CreateShaderModule(device, nxt::ShaderStage::Fragment, R"(
            #version 450

            layout(location = 0) out uvec4 visibilityBuffer;

            layout(set = 2, binding = 0) uniform constant_block {
                uint drawID;
            } u_constants;

            void main() {
                uint id3 = u_constants.drawID & 0xFF;
                uint id2 = 0x80 | ((gl_PrimitiveID >> 16) & 0x7F);
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

            layout(set = 0, binding = 0) uniform camera_block {
                mat4 viewProj;
                vec3 eye;
            } u_camera;

            layout(set = 1, binding = 0) uniform model_block {
                mat4 model;
                mat4 modelInv;
            } u_model;
            
            layout(set = 1, binding = 1) uniform sampler diffuseSampler;
            layout(set = 1, binding = 2) uniform sampler normalSampler;
            layout(set = 1, binding = 3) uniform sampler specularSampler;
            layout(set = 1, binding = 4) uniform texture2D diffuseTexture;
            layout(set = 1, binding = 5) uniform texture2D normalTexture;
            layout(set = 1, binding = 6) uniform texture2D specularTexture;

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
                uint normal;
                uint tangent;
                uint texCoord;
                uint materialID;
                uint pad;
            } vertices[];

            layout(set = 3, binding = 0) uniform constant_block {
                uint drawID;
            } u_constants;

            uint packColor(uvec4 color) {
                return (color.r & 0XFF) + ((color.g & 0XFF) << 8) + ((color.b & 0XFF) << 16) + ((color.a & 0XFF) << 24);
            }

            vec3 unpackNormal(uint normal, float zDir) {
                vec3 n;
                n.x = 2.0 * (float((normal >> 16) & 0xFFFF) / 0xFFFF - 0.5);
                n.y = 2.0 * (float(normal & 0xFFFF) / 0xFFFF - 0.5);
                n.z = zDir * sqrt(1-dot(n.xy, n.xy));
                return n;
            }

            vec2 unpackTexcoord(uint texCoord) {
                vec2 uv;
                uv.x = float((texCoord >> 16) & 0xFFFF) / 0xFFFF;
                uv.y = float(texCoord & 0xFFFF) / 0xFFFF;
                return uv;
            }

            vec3 applyNormalMap(vec3 geomnor, vec3 normap) {
                normap = normap * 2.0 - 1.0;
                vec3 up = normalize(vec3(0.001, 1, 0.001));
                vec3 surftan = normalize(cross(geomnor, up));
                vec3 surfbinor = cross(geomnor, surftan);
                return normap.y * surftan + normap.x * surfbinor + normap.z * geomnor;
            }
            
            // Returns (b0, b1, b2, perspCorrect)
            vec4 calculateBarycentrics(vec4 p0_ndc, vec4 p1_ndc, vec4 p2_ndc, vec2 ndc) {
                p0_ndc.xy /= p0_ndc.w;
                p1_ndc.xy /= p1_ndc.w;
                p2_ndc.xy /= p2_ndc.w;

                vec2 v0 = p1_ndc.xy - p0_ndc.xy;
                vec2 v1 = p2_ndc.xy - p0_ndc.xy;
                vec2 v2 = ndc - p0_ndc.xy;
                float denom = 1.f / (v0.x * v1.y - v1.x * v0.y);
                float b1 = (v2.x * v1.y - v1.x * v2.y) * denom;
                float b2 = (v0.x * v2.y - v2.x * v0.y) * denom;
                float b0 = 1.0f - b1 - b2;

                float oneOverW = 1.f / (b0 / p0_ndc.w + b1 / p1_ndc.w + b2 / p2_ndc.w);
                b0 /= p0_ndc.w;
                b1 /= p1_ndc.w;
                b2 /= p2_ndc.w;

                return vec4(b0, b1, b2, oneOverW);
            }

            float barycentricInterpolate(float v0, float v1, float v2, vec4 barys) {
                return (v0 * barys[0] + v1 * barys[1] + v2 * barys[2]) * barys[3];
            }

            vec2 barycentricInterpolate(vec2 v0, vec2 v1, vec2 v2, vec4 barys) {
                return (v0 * barys[0] + v1 * barys[1] + v2 * barys[2]) * barys[3];
            }

            vec3 barycentricInterpolate(vec3 v0, vec3 v1, vec3 v2, vec4 barys) {
                return (v0 * barys[0] + v1 * barys[1] + v2 * barys[2]) * barys[3];
            }

            vec4 barycentricInterpolate(vec4 v0, vec4 v1, vec4 v2, vec4 barys) {
                return (v0 * barys[0] + v1 * barys[1] + v2 * barys[2]) * barys[3];
            }

            void main() {
                uint outIndex = gl_GlobalInvocationID.x + 640 * gl_GlobalInvocationID.y;
                uvec4 gBufferVal = texelFetch( usampler2D(gBuffer, gBufferSampler), ivec2(gl_GlobalInvocationID.xy), 0 );

                if ((gBufferVal.b & 0x80) == 0) {
                    if (mod(gl_GlobalInvocationID.x / 20 + gl_GlobalInvocationID.y / 20, 2) == 0) {
                        fragColor[outIndex].color = packColor(uvec4(80, 80, 80, 255));
                    } else {
                        fragColor[outIndex].color = packColor(uvec4(50, 50, 50, 255));
                    }
                    return;
                }
                gBufferVal.b &= 0x7F;

                uint drawID = gBufferVal.a;
                if (drawID != u_constants.drawID) {
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

                vec4 barys = calculateBarycentrics(
                    u_camera.viewProj * p0_world, 
                    u_camera.viewProj * p1_world,
                    u_camera.viewProj * p2_world,
                    vec2(2.0, -2.0) * ((gl_GlobalInvocationID.xy / vec2(640.0, 480.0)) - vec2(0.5, 0.5))
                );

                vec4 p_world = barycentricInterpolate(p0_world, p1_world, p2_world, barys);

                vec3 faceNormal = normalize(cross(vec3(p1_world - p0_world), vec3(p2_world - p0_world)));
                float zDir = faceNormal.z / abs(faceNormal.z);

                vec3 N = normalize(barycentricInterpolate(unpackNormal(vertices[i0].normal, zDir), unpackNormal(vertices[i1].normal, zDir), unpackNormal(vertices[i2].normal, zDir), barys));
                vec2 texCoord = barycentricInterpolate(unpackTexcoord(vertices[i0].texCoord), unpackTexcoord(vertices[i1].texCoord), unpackTexcoord(vertices[i2].texCoord), barys);
                
                vec3 diffuseColor = texture(sampler2D(diffuseTexture, diffuseSampler), texCoord).rgb;
                vec3 normalMap = texture(sampler2D(normalTexture, normalSampler), texCoord).rgb;
                vec3 specularColor = texture(sampler2D(specularTexture, specularSampler), texCoord).rgb;

                N = applyNormalMap(N, normalMap);

                vec3 V = normalize(u_camera.eye - p_world.xyz);

                // position, intensity
                const vec4 lights[2] = vec4[2](
                    vec4(10.0, 30.0, -20.0, 1.0),
                    vec4(20.0, 20.0, 20.0, 0.5)
                );

                float diffuseTerm = 0;
                float specularTerm = 0;
                for (uint i = 0; i < 2; ++i) {
                    vec3 L = normalize(lights[i].xyz - p_world.xyz);
                    vec3 H = (L + V) / 2.0;

                    specularTerm += lights[i].w * pow(max(dot(H, N), 0.0), 20.0);
                    diffuseTerm += lights[i].w * max(dot(L, N), 0.0);
                }
                diffuseTerm = 0.15 + 0.85 * diffuseTerm;
                vec3 composite = diffuseTerm * diffuseColor + specularTerm * specularColor;

                composite = clamp(composite, vec3(0.0), vec3(1.0));
                
                fragColor[outIndex].color = packColor(uvec4(255 * vec4(composite, 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(vec3(barys) * barys[3], 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(abs(V), 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(normalMap, 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(vec3(b0, b1, b2), 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4((ndc + 1.0) * 0.5, 0, 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(vec3(p_world), 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(texCoord, 0, 1)));
                // fragColor[outIndex].color = packColor(uvec4(255 * vec4(N, 1)));
            })");

        auto computeOutputLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::StorageBuffer, 0, 1)
            .GetResult();

        modelBindGroupLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::UniformBuffer, 0, 1)
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::Sampler, 1, 3)
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::SampledTexture, 4, 3)
            .GetResult();

        computeBindGroupLayout = device.CreateBindGroupLayoutBuilder()
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::Sampler, 0, 1) // gbuffer sampler
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::SampledTexture, 1, 1) // gbuffer texture
            .SetBindingsType(nxt::ShaderStageBit::Compute, nxt::BindingType::StorageBuffer, 2, 3) // output, index, vertex
            .GetResult();


        auto pipelineLayout = device.CreatePipelineLayoutBuilder()
            .SetBindGroupLayout(0, layout::cameraLayout)
            .SetBindGroupLayout(1, modelBindGroupLayout)
            .SetBindGroupLayout(2, computeBindGroupLayout)
            .SetBindGroupLayout(3, constantsBindGroupLayout)
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

    globalDevice.Unlock();
}

void Renderer::Render(nxt::Texture &texture) {
    scene.UpdateModelList();
    const auto& models = scene.GetModels();

    auto backBufferView = texture.CreateTextureViewBuilder().GetResult();

    gBufferTexture.TransitionUsage(nxt::TextureUsageBit::OutputAttachment);
    auto framebuffer = LOCK_AND_RELEASE(globalDevice, CreateFramebufferBuilder())
        .SetRenderPass(renderpass)
        .SetDimensions(640, 480)
        .SetAttachment(0, gBufferView)
        .SetAttachment(1, depthStencilView)
        .GetResult();

    uint32_t drawCount = 0;

    for (Model* model : models) {
        for (const DrawInfo& draw : model->GetCommands()) {
            drawCount++;
        }
    }

    if (drawCount == 0) {
        return;
    }

    if (drawCount != constantBindGroups.size()) {
        constantBindGroups.clear();
        constantBindGroups.reserve(drawCount);

        uint32_t drawID = 0;

        struct alignas(256) DrawConstants {
            uint32_t drawID;
        };

        auto buffer = LOCK_AND_RELEASE(globalDevice, CreateBufferBuilder())
            .SetAllowedUsage(nxt::BufferUsageBit::TransferDst | nxt::BufferUsageBit::Uniform)
            .SetInitialUsage(nxt::BufferUsageBit::TransferDst)
            .SetSize(drawCount * sizeof(DrawConstants))
            .GetResult();
        std::vector<DrawConstants> staging;
        staging.reserve(drawCount);

        for (Model* model : scene.GetModels()) {
            for (const DrawInfo& draw : model->GetCommands()) {
                staging.push_back(DrawConstants{
                    drawID
                });

                auto bufferView = buffer.CreateBufferViewBuilder()
                    .SetExtent(sizeof(DrawConstants) * drawID, sizeof(DrawConstants))
                    .GetResult();

                constantBindGroups.push_back(LOCK_AND_RELEASE(globalDevice, CreateBindGroupBuilder())
                    .SetUsage(nxt::BindGroupUsage::Frozen)
                    .SetLayout(constantsBindGroupLayout)
                    .SetBufferViews(0, 1, &bufferView)
                    .GetResult()
                );

                drawID++;
            }
        }

        buffer.SetSubData(0, drawCount * sizeof(DrawConstants) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(staging.data()));
        buffer.FreezeUsage(nxt::BufferUsageBit::Uniform);
    }

    layout::camera_block cameraBlock{
        camera.GetViewProj(),
        camera.GetPosition()
    };
    cameraBuffer.TransitionUsage(nxt::BufferUsageBit::TransferDst);
    cameraBuffer.SetSubData(0, sizeof(layout::camera_block) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(&cameraBlock));

    auto commands = LOCK_AND_RELEASE(globalDevice, CreateCommandBufferBuilder());

    commands.TransitionBufferUsage(cameraBuffer, nxt::BufferUsageBit::Uniform);

    for (Model* model : scene.GetModels()) {
        commands.TransitionBufferUsage(model->GetUniformBuffer(), nxt::BufferUsageBit::Uniform);
        for (const DrawInfo& draw : model->GetCommands()) {
            commands.TransitionBufferUsage(draw.indexBuffer, nxt::BufferUsageBit::Index);
            commands.TransitionBufferUsage(draw.vertexBuffer, nxt::BufferUsageBit::Vertex);
        }
    }

    commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::Storage);
    commands.TransitionTextureUsage(texture, nxt::TextureUsageBit::TransferDst);

    uint32_t drawID = 0;

    commands.BeginRenderPass(renderpass, framebuffer);
    commands.BeginRenderSubpass();
    commands.SetRenderPipeline(rasterizePipeline);
    commands.SetBindGroup(0, cameraBindGroup);
    for (Model* model : models) {
        for (const DrawInfo& draw : model->GetCommands()) {
            auto modelBindGroup = LOCK_AND_RELEASE(globalDevice, CreateBindGroupBuilder())
                .SetUsage(nxt::BindGroupUsage::Frozen)
                .SetLayout(layout::modelLayout)
                .SetBufferViews(0, 1, &draw.uniformBufferView)
                .GetResult();

            commands.SetBindGroup(1, modelBindGroup);
            commands.SetBindGroup(2, constantBindGroups[drawID]);
            uint32_t zero = 0;
            commands.SetVertexBuffers(0, 1, &draw.vertexBuffer, &zero);
            commands.SetIndexBuffer(draw.indexBuffer, 0, nxt::IndexFormat::Uint32);
            commands.DrawElements(draw.count, 1, 0, 0);

            drawID++;
        }
    }
    commands.EndRenderSubpass();
    commands.EndRenderPass();

    for (Model* model : models) {
        for (const DrawInfo& draw : model->GetCommands()) {
            commands.TransitionBufferUsage(draw.indexBuffer, nxt::BufferUsageBit::Storage);
            commands.TransitionBufferUsage(draw.vertexBuffer, nxt::BufferUsageBit::Storage);
            commands.TransitionTextureUsage(gBufferTexture, nxt::TextureUsageBit::Sampled);
        }
    }

    drawID = 0;

    commands.BeginComputePass();
    commands.SetComputePipeline(shadingPipeline);
    commands.SetBindGroup(0, cameraBindGroup);

    for (Model* model : models) {
        for (const DrawInfo& draw : model->GetCommands()) {
            nxt::BufferView computeBufferViews[] = {
                computeOutputBufferView.Clone(),
                draw.indexBufferView.Clone(),
                draw.vertexBufferView.Clone(),
            };

            auto modelBindGroup = LOCK_AND_RELEASE(globalDevice, CreateBindGroupBuilder())
                .SetUsage(nxt::BindGroupUsage::Frozen)
                .SetLayout(modelBindGroupLayout)
                .SetBufferViews(0, 1, &draw.uniformBufferView)
                .SetSamplers(1, 3, &draw.diffuseSampler)
                .SetTextureViews(4, 3, &draw.diffuseTexture)
                .GetResult();

            auto computeBindGroup = LOCK_AND_RELEASE(globalDevice, CreateBindGroupBuilder())
                .SetUsage(nxt::BindGroupUsage::Frozen)
                .SetLayout(computeBindGroupLayout)
                .SetSamplers(0, 1, &default::defaultSampler)
                .SetTextureViews(1, 1, &gBufferView)
                .SetBufferViews(2, 3, computeBufferViews)
                .GetResult();

            commands.SetBindGroup(1, modelBindGroup);
            commands.SetBindGroup(2, computeBindGroup);
            commands.SetBindGroup(3, constantBindGroups[drawID]);
            commands.Dispatch(640, 480, 1);

            drawID++;
        }
    }

    commands.EndComputePass();

    commands.TransitionBufferUsage(computeOutputBuffer, nxt::BufferUsageBit::TransferSrc);
    commands.CopyBufferToTexture(computeOutputBuffer, 0, 0, texture, 0, 0, 0, 640, 480, 1, 0);

    auto cmds = commands.GetResult();
    LOCK_IN_SCOPE(globalDevice, [&](nxt::Device&) {
        globalQueue.Submit(1, &cmds);
    })
}

void Renderer::Quit() {
    shouldQuit = true;
    _thread.join();
}

bool Renderer::ShouldQuit() const {
    return shouldQuit;
}
