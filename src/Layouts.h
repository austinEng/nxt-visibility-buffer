
#pragma once

#include <glm/glm/mat4x4.hpp>
#include <nxt/nxtcpp.h>

namespace layout {

    struct camera_block {
        glm::mat4 viewProj;
    };

    struct model_block {
        glm::mat4 model;
    };

    extern nxt::BindGroupLayout cameraLayout;
    extern nxt::BindGroupLayout modelLayout;
    extern nxt::BindGroupLayout computeBufferLayout;
}
