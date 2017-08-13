
#pragma once

#include <mutex>
#include <map>
#include <string>
#include "ModelLoader.h"

class Scene {
    public:
        void AddModel(std::string gltfPath);
        void UpdateModelList();
        const std::vector<Model*>& GetModels() const;
        ~Scene();

    private:
        uint64_t modelUploadSerial = 1;
        uint64_t currentModelUploadSerial = 0;

        std::mutex modelLoaderMutex;

        std::vector<ModelLoader*> loaders;
        std::vector<Model*> models;
        std::vector<Model*> currentModels;
};
