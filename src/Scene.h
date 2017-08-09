
#pragma once

#include <mutex>
#include <map>
#include <string>
#include "ModelLoader.h"

class Scene {
    public:
        Scene(const nxt::Device&, const nxt::Queue& queue);
        void AddModel(std::string gltfPath);
        const std::vector<Model*>& GetModels() const;
        ~Scene();

    private:
        nxt::Device device;
        nxt::Queue queue;
        std::mutex modelLoaderMutex;

        std::vector<ModelLoader*> loaders;
        std::vector<Model*> models;
};
