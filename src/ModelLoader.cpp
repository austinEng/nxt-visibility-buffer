
#include "ModelLoader.h"

#include <common/Constants.h>
#include <common/Math.h>
#include <glm/glm/mat4x4.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/glm/gtc/type_ptr.hpp>
#include <tinygltfloader/tiny_gltf_loader.h>
#include <utils/NXTHelpers.h>
#include "Binding.h"
#include "Layouts.h"

namespace {
    std::string getFilePathExtension(const std::string &filepath) {
        if (filepath.find_last_of(".") != std::string::npos) {
            return filepath.substr(filepath.find_last_of(".") + 1);
        }
        return "";
    }

    namespace gl {
        enum {
            Triangles = 0x0004,
            UnsignedShort = 0x1403,
            UnsignedInt = 0x1405,
            Float = 0x1406,
            RGBA = 0x1908,
            Nearest = 0x2600,
            Linear = 0x2601,
            NearestMipmapNearest = 0x2700,
            LinearMipmapNearest = 0x2701,
            NearestMipmapLinear = 0x2702,
            LinearMipmapLinear = 0x2703,
            ArrayBuffer = 0x8892,
            ElementArrayBuffer = 0x8893,
            FragmentShader = 0x8B30,
            VertexShader = 0x8B31,
            FloatVec2 = 0x8B50,
            FloatVec3 = 0x8B51,
            FloatVec4 = 0x8B52,
        };
    }

    std::map<uint32_t, std::string> slotSemantics = { { 0, "POSITION" },{ 1, "NORMAL" },{ 2, "TEXCOORD_0" } };
}

void ModelLoader::Start(const nxt::Device& device, const nxt::Queue& queue, std::string gltfPath, Model* model, const std::function<void(ModelLoader*, Model*)> &callback) {
    printf("Loading %s\n", gltfPath.c_str());

    tinygltf::TinyGLTFLoader tinyGLTFLoader;
    std::string ext = getFilePathExtension(gltfPath);

   std::string error;
    bool result = false;
    if (ext.compare("glb") == 0) {
        result = tinyGLTFLoader.LoadBinaryFromFile(&model->scene, &error, gltfPath.c_str());
    } else {
        result = tinyGLTFLoader.LoadASCIIFromFile(&model->scene, &error, gltfPath.c_str());
    }

    if (!error.empty()) {
        fprintf(stderr, "ERR: %s\n", error.c_str());
    }

    if (!result) {
        return callback(this, nullptr);
    }

    const tinygltf::Scene& scene = model->scene;

    {
        for (const auto& bv : scene.bufferViews) {
            const auto& iBufferViewID = bv.first;
            const auto& iBufferView = bv.second;

            nxt::BufferUsageBit usage = nxt::BufferUsageBit::Storage;
            switch (iBufferView.target) {
            case gl::ArrayBuffer:
                usage |= nxt::BufferUsageBit::Vertex;
                break;
            case gl::ElementArrayBuffer:
                usage |= nxt::BufferUsageBit::Index;
                break;
            case 0:
                fprintf(stderr, "TODO: buffer view has no target; skipping\n");
                continue;
            default:
                fprintf(stderr, "unsupported buffer view target %d\n", iBufferView.target);
                continue;
            }
            const auto& iBuffer = scene.buffers.at(iBufferView.buffer);

            size_t iBufferViewSize =
                iBufferView.byteLength ? iBufferView.byteLength :
                (iBuffer.data.size() - iBufferView.byteOffset);

            nxt::Buffer oBuffer = device.CreateBufferBuilder()
                .SetAllowedUsage(nxt::BufferUsageBit::TransferDst | usage)
                .SetInitialUsage(nxt::BufferUsageBit::TransferDst)
                .SetSize(static_cast<uint32_t>(iBufferViewSize))
                .GetResult();
            oBuffer.SetSubData(0, static_cast<uint32_t>(iBufferViewSize) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(&iBuffer.data.at(iBufferView.byteOffset)));
            model->buffers[iBufferViewID] = std::move(oBuffer);
        }
    }
    {
        for (const auto& s : scene.samplers) {
            const auto& iSamplerID = s.first;
            const auto& iSampler = s.second;

            auto magFilter = nxt::FilterMode::Nearest;
            auto minFilter = nxt::FilterMode::Nearest;
            auto mipmapFilter = nxt::FilterMode::Nearest;
            switch (iSampler.magFilter) {
            case gl::Nearest:
                magFilter = nxt::FilterMode::Nearest;
                break;
            case gl::Linear:
                magFilter = nxt::FilterMode::Linear;
                break;
            default:
                fprintf(stderr, "unsupported magFilter %d\n", iSampler.magFilter);
                break;
            }
            switch (iSampler.minFilter) {
            case gl::Nearest:
            case gl::NearestMipmapNearest:
            case gl::NearestMipmapLinear:
                minFilter = nxt::FilterMode::Nearest;
                break;
            case gl::Linear:
            case gl::LinearMipmapNearest:
            case gl::LinearMipmapLinear:
                minFilter = nxt::FilterMode::Linear;
                break;
            default:
                fprintf(stderr, "unsupported minFilter %d\n", iSampler.magFilter);
                break;
            }
            switch (iSampler.minFilter) {
            case gl::NearestMipmapNearest:
            case gl::LinearMipmapNearest:
                mipmapFilter = nxt::FilterMode::Nearest;
                break;
            case gl::NearestMipmapLinear:
            case gl::LinearMipmapLinear:
                mipmapFilter = nxt::FilterMode::Linear;
                break;
            }

            auto oSampler = device.CreateSamplerBuilder()
                .SetFilterMode(magFilter, minFilter, mipmapFilter)
                // TODO: wrap modes
                .GetResult();

            model->samplers[iSamplerID] = std::move(oSampler);
        }
    }
    {
        for (const auto& t : scene.textures) {
            const auto& iTextureID = t.first;
            const auto& iTexture = t.second;
            const auto& iImage = scene.images.at(iTexture.source);

            nxt::TextureFormat format = nxt::TextureFormat::R8G8B8A8Unorm;
            switch (iTexture.format) {
            case gl::RGBA:
                format = nxt::TextureFormat::R8G8B8A8Unorm;
                break;
            default:
                fprintf(stderr, "unsupported texture format %d\n", iTexture.format);
                continue;
            }

            auto oTexture = device.CreateTextureBuilder()
                .SetDimension(nxt::TextureDimension::e2D)
                .SetExtent(iImage.width, iImage.height, 1)
                .SetFormat(format)
                .SetMipLevels(1)
                .SetAllowedUsage(nxt::TextureUsageBit::TransferDst | nxt::TextureUsageBit::Sampled)
                .GetResult();
            // TODO: release this texture

            const uint8_t* origData = iImage.image.data();
            const uint8_t* data = nullptr;
            std::vector<uint8_t> newData;

            uint32_t width = static_cast<uint32_t>(iImage.width);
            uint32_t height = static_cast<uint32_t>(iImage.height);
            uint32_t rowSize = width * 4;
            uint32_t rowPitch = Align(rowSize, kTextureRowPitchAlignment);

            if (iImage.component == 3 || iImage.component == 4) {
                if (rowSize != rowPitch || iImage.component == 3) {
                    newData.resize(rowPitch * height);
                    uint32_t pixelsPerRow = rowPitch / 4;
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            size_t oldIndex = x + y * height;
                            size_t newIndex = x + y * pixelsPerRow;
                            if (iImage.component == 4) {
                                newData[4 * newIndex + 0] = origData[4 * oldIndex + 0];
                                newData[4 * newIndex + 1] = origData[4 * oldIndex + 1];
                                newData[4 * newIndex + 2] = origData[4 * oldIndex + 2];
                                newData[4 * newIndex + 3] = origData[4 * oldIndex + 3];
                            }
                            else if (iImage.component == 3) {
                                newData[4 * newIndex + 0] = origData[3 * oldIndex + 0];
                                newData[4 * newIndex + 1] = origData[3 * oldIndex + 1];
                                newData[4 * newIndex + 2] = origData[3 * oldIndex + 2];
                                newData[4 * newIndex + 3] = 255;
                            }
                        }
                    }
                    data = newData.data();
                }
                else {
                    data = origData;
                }
            }
            else {
                fprintf(stderr, "unsupported image.component %d\n", iImage.component);
            }

            nxt::Buffer staging = utils::CreateFrozenBufferFromData(device, data, rowPitch * iImage.height, nxt::BufferUsageBit::TransferSrc);
            auto cmdbuf = device.CreateCommandBufferBuilder()
                .TransitionTextureUsage(oTexture, nxt::TextureUsageBit::TransferDst)
                .CopyBufferToTexture(staging, 0, rowPitch, oTexture, 0, 0, 0, iImage.width, iImage.height, 1, 0)
                .GetResult();
            queue.Submit(1, &cmdbuf);
            oTexture.FreezeUsage(nxt::TextureUsageBit::Sampled);

            model->textureViews[iTextureID] = oTexture.CreateTextureViewBuilder().GetResult();
        }
    }

    for (const auto& it : scene.meshes) {
        const auto& mesh = it.second;
        for (const auto& prim : mesh.primitives) {

            const auto& positionAttribute = prim.attributes.at("POSITION");

            const auto& positionAccessor = scene.accessors.at(positionAttribute);
            if (positionAccessor.byteStride != 12) {
                fprintf(stderr, "Byte stride must be 12\n");
            }

            if (positionAccessor.componentType != gl::Float ||
                (positionAccessor.type != TINYGLTF_TYPE_VEC4 && positionAccessor.type != TINYGLTF_TYPE_VEC3 && positionAccessor.type != TINYGLTF_TYPE_VEC2)) {
                fprintf(stderr, "unsupported vertex accessor component type %d and type %d\n", positionAccessor.componentType, positionAccessor.type);
                continue;
            }

            if (prim.indices.empty()) {
                fprintf(stderr, "No indicies found\n");
                continue;
            }

            const auto& indexAccesor = scene.accessors.at(prim.indices);
            if (indexAccesor.componentType != gl::UnsignedShort || indexAccesor.type != TINYGLTF_TYPE_SCALAR) {
                fprintf(stderr, "unsupported index accessor component type %d and type %d\n", indexAccesor.componentType, indexAccesor.type);
                continue;
            }
        }
    }

    model->UpdateCommands(device, queue);

    callback(this, model);
}


void Model::UpdateCommands(const nxt::Device& device, const nxt::Queue& queue) {
    rasterCommands.clear();
    transforms.clear();

    std::vector<std::pair<const tinygltf::Node*, glm::mat4>> stack;
    const auto &defaultSceneNodes = scene.scenes[scene.defaultScene];
    for (const auto& nodeID : defaultSceneNodes) {
        const auto& node = scene.nodes.at(nodeID);
        stack.push_back(std::make_pair(&node, glm::mat4(1.0)));
    }
    while (stack.size() > 0) {
        const auto current = stack.back();
        stack.pop_back();
        const auto* node = current.first;
        const auto& transform = current.second;

        glm::mat4 model;
        if (node->matrix.size() == 16) {
            model = glm::make_mat4(node->matrix.data());
        } else {
            if (node->scale.size() == 3) {
                glm::vec3 scale = glm::make_vec3(node->scale.data());
                model = glm::scale(model, scale);
            }
            if (node->rotation.size() == 4) {
                glm::quat rotation = glm::make_quat(node->rotation.data());
                model = glm::mat4_cast(rotation) * model;
            }
            if (node->translation.size() == 3) {
                glm::vec3 translation = glm::make_vec3(node->translation.data());
                model = glm::translate(model, translation);
            }
        }
        model = transform * model;

        for (const auto& meshID : node->meshes) {
            const auto& mesh = scene.meshes.at(meshID);
            for (const auto& prim : mesh.primitives) {
                if (prim.mode != gl::Triangles) {
                    fprintf(stderr, "unsupported primitive mode %d\n", prim.mode);
                    continue;
                }

                {
                    auto it = prim.attributes.find("POSITION");
                    if (it == prim.attributes.end()) {
                        fprintf(stderr, "Missing POSITION attribute\n");
                        continue;
                    }
                    const auto& positionAccessor = it->second;
                    if (scene.accessors.at(positionAccessor).byteStride != 12) {
                        fprintf(stderr, "Byte stride must be 12\n");
                    }

                    const auto& accessor = scene.accessors.at(it->second);
                    if (accessor.componentType != gl::Float ||
                        (accessor.type != TINYGLTF_TYPE_VEC4 && accessor.type != TINYGLTF_TYPE_VEC3 && accessor.type != TINYGLTF_TYPE_VEC2)) {
                        fprintf(stderr, "unsupported vertex accessor component type %d and type %d\n", accessor.componentType, accessor.type);
                        continue;
                    }

                    if (prim.indices.empty()) {
                        fprintf(stderr, "No indicies found\n");
                        continue;
                    }

                    const auto& indices = scene.accessors.at(prim.indices);
                    if (indices.componentType != gl::UnsignedShort || indices.type != TINYGLTF_TYPE_SCALAR) {
                        fprintf(stderr, "unsupported index accessor component type %d and type %d\n", indices.componentType, indices.type);
                        continue;
                    }

                    rasterCommands.emplace_back(RasterCommand{
                        buffers.at(accessor.bufferView).Clone(),
                        buffers.at(indices.bufferView).Clone(),
                        static_cast<uint32_t>(accessor.byteOffset),
                        static_cast<uint32_t>(indices.byteOffset),
                        static_cast<uint32_t>(indices.count),
                    });

                    RasterCommand& rasterCommand = rasterCommands.back();

                    nxt::BufferView views[2] = {
                        rasterCommand.vertexBuffer.CreateBufferViewBuilder()
                            .SetExtent(accessor.byteOffset, scene.bufferViews.at(accessor.bufferView).byteLength)
                            .GetResult(),
                        rasterCommand.indexBuffer.CreateBufferViewBuilder()
                            .SetExtent(indices.byteOffset, scene.bufferViews.at(indices.bufferView).byteLength)
                            .GetResult(),
                    };

                    rasterCommand.storage = device.CreateBindGroupBuilder()
                        .SetLayout(layout::computeBufferLayout)
                        .SetBufferViews(0, 2, views)
                        .SetUsage(nxt::BindGroupUsage::Frozen)
                        .GetResult();

                    transforms.push_back(model);
                }
            }
        }

        for (const auto& childID : node->children) {
            const auto& child = scene.nodes.at(childID);
            stack.push_back(std::make_pair(&child, model));
        }
    }

    assert(rasterCommands.size() == transforms.size());

    if (!uniformBuffer) {
        uniformBuffer = device.CreateBufferBuilder()
            .SetAllowedUsage(nxt::BufferUsageBit::Uniform | nxt::BufferUsageBit::TransferDst)
            .SetInitialUsage(nxt::BufferUsageBit::TransferDst)
            .SetSize(sizeof(layout::model_block) * static_cast<uint32_t>(rasterCommands.size()))
            .GetResult();
    } else {
        uniformBuffer.TransitionUsage(nxt::BufferUsageBit::TransferDst);
    }

    uniformBuffer.SetSubData(0, sizeof(layout::model_block) * static_cast<uint32_t>(rasterCommands.size()) / sizeof(uint32_t), reinterpret_cast<const uint32_t*>(transforms.data()));

    for (unsigned int i = 0; i < rasterCommands.size(); ++i) {

        auto uniformBufferView = uniformBuffer.CreateBufferViewBuilder()
            .SetExtent(i * sizeof(layout::model_block), sizeof(layout::model_block))
            .GetResult();

        rasterCommands[i].uniforms = device.CreateBindGroupBuilder()
            .SetLayout(layout::modelLayout)
            .SetBufferViews(0, 1, &uniformBufferView)
            .SetUsage(nxt::BindGroupUsage::Frozen)
            .GetResult();

    }
}

const std::vector<RasterCommand>& Model::GetRasterCommands() const {
    return rasterCommands;
}

const nxt::Buffer& Model::GetUniformBuffer() const {
    return uniformBuffer;
}
