
#pragma once

#include <glm/glm/mat4x4.hpp>

struct RasterCommand {
    nxt::Buffer vertexBuffer;
    nxt::Buffer indexBuffer;
    nxt::BufferView vertexBufferView;
    nxt::BufferView indexBufferView;
    uint32_t vertexBufferOffset;
    uint32_t indexBufferOffset;
    uint32_t count;
    nxt::BufferView uniformBufferView;
};
