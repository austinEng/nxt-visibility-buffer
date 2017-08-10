
#pragma once

#include <functional>
#include <glm/glm/mat4x4.hpp>
#include <map>
#include <nxt/nxtcpp.h>
#include <tinygltfloader/tiny_gltf_loader.h>
#include <thread>
#include "Commands.h"

class Model;
class ModelLoader {

    std::thread _thread;

    public:
        template<typename... Args>
        ModelLoader(Args&... args) : _thread(&ModelLoader::Start, this, std::forward<Args>(args)...) { }

        void Start(const nxt::Device& device, const nxt::Queue& queue, std::string gltfPath, Model* model, const std::function<void(ModelLoader*, Model*)> &callback);

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
        glm::vec3 position;
        unsigned int normal;
        unsigned int tangent;
        unsigned int texCoord;
        unsigned int materialID;
        unsigned int pad;
    };

    void UpdateCommands(const nxt::Device& device, const nxt::Queue& queue);

    const std::vector<RasterCommand>& GetCommands() const;
    const nxt::Buffer& GetUniformBuffer() const;

private:

    tinygltf::Scene scene;
    std::map<std::string, nxt::Buffer> buffers;
    std::map<std::string, nxt::Sampler> samplers;
    std::map<std::string, nxt::TextureView> textureViews;

    std::vector<RasterCommand> rasterCommands;
    std::vector<glm::mat4> transforms;
    nxt::Buffer uniformBuffer;
};
