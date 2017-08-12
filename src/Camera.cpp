
#include "Camera.h"

Camera::Camera()
    : _azimuth(glm::radians(45.f)),
    _altitude(glm::radians(30.f)),
    _radius(10.f),
    _center(0, 0, 0),
    _dirty(true) {
    Recalculate();
}

void Camera::Rotate(float dAzimuth, float dAltitude) {
    _dirty = true;
    _azimuth = glm::mod(_azimuth + dAzimuth, glm::radians(360.f));
    _altitude = glm::clamp(_altitude + dAltitude, glm::radians(-89.f), glm::radians(89.f));
}

void Camera::Pan(float dX, float dY) {
    Recalculate();
    glm::vec3 vX = glm::normalize(glm::cross(-_eyeDir, glm::vec3(0, 1, 0)));
    glm::vec3 vY = glm::normalize(glm::cross(_eyeDir, vX));
    _center += vX * dX * _radius + vY * dY * _radius;
}

void Camera::Zoom(float factor) {
    _dirty = true;
    _radius = _radius * glm::exp(-factor);
}

glm::mat4 Camera::GetView() const {
    if (_dirty) {
        Recalculate();
    }
    return _view;
}

glm::mat4 Camera::GetViewProj() const {
    return _projection * GetView();
}

glm::vec3 Camera::GetPosition() const {
    return _center + _eyeDir * _radius;
}

void Camera::Recalculate() const {
    glm::vec4 eye4 = glm::vec4(1, 0, 0, 1);
    eye4 = glm::rotate(glm::mat4(), _altitude, glm::vec3(0, 0, 1)) * eye4;
    eye4 = glm::rotate(glm::mat4(), _azimuth, glm::vec3(0, 1, 0)) * eye4;
    _eyeDir = glm::vec3(eye4);

    _view = glm::lookAt(_center + _eyeDir * _radius, _center, glm::vec3(0, 1, 0));
    _dirty = false;
}
