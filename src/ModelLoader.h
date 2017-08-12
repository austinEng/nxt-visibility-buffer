
#pragma once

#include <array>
#include <functional>
#include <glm/glm/mat4x4.hpp>
#include <map>
#include <nxt/nxtcpp.h>
#include <tinygltfloader/tiny_gltf_loader.h>
#include <thread>
#include "Commands.h"
#include "Globals.h"

class Model;
class ModelLoader {

    std::thread _thread;

    public:
        template<typename... Args>
        ModelLoader(Args&... args) : _thread(&ModelLoader::Start, this, std::forward<Args>(args)...) { }

        void Start(std::string gltfPath, Model* model, const std::function<void(ModelLoader*, Model*)> &callback);

        ModelLoader(const ModelLoader &other) = delete;
        ModelLoader& operator=(const ModelLoader &other) = delete;
        ModelLoader(ModelLoader&& other) : _thread(std::move(other._thread)) { }
        ModelLoader& operator=(ModelLoader&& other) {
            _thread = std::move(other._thread);
            return *this;
        }
        ~ModelLoader() {
            if (_thread.joinable()) {
                _thread.detach();
            }
        }
};


class Model {
    friend class ModelLoader;

public:

    struct Vertex {
        glm::vec3 position = glm::vec3(0,0,0);
        uint32_t normal = 0;
        uint32_t tangent = 0;
        uint32_t texCoord = 0;
        uint32_t materialID = 0;
        uint32_t pad;
    };

    void UpdateCommands();

    const std::vector<DrawInfo>& GetCommands() const;
    const nxt::Buffer& GetUniformBuffer() const;

private:

    tinygltf::Scene scene;
    std::map<std::string, nxt::Buffer> buffers;
    std::map<std::string, nxt::Sampler> samplers;
    std::map<std::string, nxt::TextureView> textureViews;

    std::vector<DrawInfo> rasterCommands;
    std::vector<layout::model_block> uniforms;
    nxt::Buffer uniformBuffer;
};
