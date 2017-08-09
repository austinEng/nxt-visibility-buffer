
#pragma once

#include <glm/glm/mat4x4.hpp>

struct RasterCommand {
    nxt::Buffer vertexBuffer;
    uint32_t vertexBufferOffset;
    nxt::Buffer indexBuffer;
    uint32_t indexBufferOffset;
    uint32_t count;
    nxt::BindGroup uniforms;
};
