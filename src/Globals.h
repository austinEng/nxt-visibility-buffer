
#pragma once

#include <glm/glm/mat4x4.hpp>
#include <nxt/nxtcpp.h>

#include "LockedObject.h"

extern LockedObject<nxt::Device> globalDevice;
extern LockedObject<nxt::Queue> globalQueue;

namespace layout {

    struct alignas(256) camera_block {
        glm::mat4 viewProj;
        glm::vec3 eye;
        float pad;
    };

    struct alignas(256) model_block {
        glm::mat4 model;
        glm::mat4 modelInv;
    };

    extern nxt::BindGroupLayout cameraLayout;
    extern nxt::BindGroupLayout modelLayout;
}

namespace default {

    extern nxt::TextureView defaultDiffuse;
    extern nxt::TextureView defaultNormal;
    extern nxt::TextureView defaultSpecular;
    extern nxt::Sampler defaultSampler;

}
