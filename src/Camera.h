#pragma once

#include <glm/glm/mat4x4.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>

class Camera {
    public:
        Camera();

        void Rotate(float dAzimuth, float dAltitude);
        void Pan(float dX, float dY);
        void Zoom(float factor);

        glm::mat4 GetView();
        glm::mat4 GetViewProj();
        glm::vec3 GetPosition() const;
        uint64_t GetSerial() const;

    private:
        void Recalculate();

        float _azimuth;
        float _altitude;
        float _radius;
        glm::vec3 _center;
        glm::vec3 _eyeDir;
        bool _dirty;
        glm::mat4 _view;
        glm::mat4 _projection = glm::perspective(glm::radians(60.f), 640.f / 480, 0.1f, 2000.f);
        uint64_t _serial;
};
