
#include <utils/NXTHelpers.h>
#include <utils/SystemUtils.h>
#include <common/Constants.h>
#include <common/Math.h>
#include <common/Assert.h>
#include "Scene.h"
#include "Binding.h"
extern uint64_t updateSerial;

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

Scene::Scene(const nxt::Device& device, const nxt::Queue& queue) : device(device.Clone()), queue(queue.Clone()) {
}

void Scene::AddModel(std::string gltfPath) {
    Model* model = new Model();

    auto callback = [&, gltfPath, model](ModelLoader* loader, Model* model) {
        while (!modelLoaderMutex.try_lock()) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(16ms);
        }

        if (model != nullptr) {
            printf("Successfully loaded %s\n", gltfPath.c_str());
            models.push_back(model);
            updateSerial++;
        } else {
            fprintf(stderr, "Failed to load %s\n", gltfPath.c_str());
        }

        auto it = std::find(loaders.begin(), loaders.end(), loader);
        assert(it != loaders.end());
        std::swap(*it, loaders.back());
        loaders.pop_back(); 
        delete loader;
        modelLoaderMutex.unlock();
    };

    modelLoaderMutex.lock();
    loaders.push_back(new ModelLoader(device.Clone(), queue.Clone(), gltfPath, model, callback));
    printf("%d models loading...\n", loaders.size());
    modelLoaderMutex.unlock();
}

Scene::~Scene() {
    while (loaders.size() > 0) {
        printf("Waiting for %d loaders to complete...", loaders.size());
        utils::USleep(16000);
    }
}

const std::vector<Model*>& Scene::GetModels() const {
    return models;
}