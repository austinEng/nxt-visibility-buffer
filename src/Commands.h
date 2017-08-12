
#pragma once

#include <glm/glm/mat4x4.hpp>

struct DrawInfo {
    nxt::Buffer indexBuffer;
    nxt::BufferView indexBufferView;
    nxt::Buffer vertexBuffer;
    nxt::BufferView vertexBufferView;
    uint32_t count;
    nxt::BufferView uniformBufferView;
    nxt::Sampler diffuseSampler;
    nxt::Sampler normalSampler;
    nxt::Sampler specularSampler;
    nxt::TextureView diffuseTexture;
    nxt::TextureView normalTexture;
    nxt::TextureView specularTexture;
};
