
#pragma once

#include <mutex>
#include <map>
#include <string>
#include "ModelLoader.h"

class Scene {
    public:
        void AddModel(std::string gltfPath);
        const std::vector<Model*>& GetModels() const;
        ~Scene();

    private:
        std::mutex modelLoaderMutex;

        std::vector<ModelLoader*> loaders;
        std::vector<Model*> models;
};
