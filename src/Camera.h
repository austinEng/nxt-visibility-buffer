#pragma once

#include <glm/glm/mat4x4.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>

class Camera {
    public:
        Camera();

        void Rotate(float dAzimuth, float dAltitude);
        void Pan(float dX, float dY);
        void Zoom(float factor);

        glm::mat4 GetView() const;
        glm::mat4 GetViewProj() const;
        glm::vec3 GetPosition() const;

    private:
        void Recalculate() const;

        float _azimuth;
        float _altitude;
        float _radius;
        glm::vec3 _center;
        mutable glm::vec3 _eyeDir;
        mutable bool _dirty;
        mutable glm::mat4 _view;
        glm::mat4 _projection = glm::perspective(glm::radians(60.f), 1280.f / 960, 0.1f, 2000.f);
};
