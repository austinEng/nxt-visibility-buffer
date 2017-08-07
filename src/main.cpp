
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>
#include <utils/SystemUtils.h>
#include "Binding.h"
#include "Camera.h"
#include "ViewportThread.h"

namespace {
    Camera camera;

    bool buttons[GLFW_MOUSE_BUTTON_LAST + 1] = { 0 };

    void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
        buttons[button] = (action == GLFW_PRESS);
    }

    void cursorPosCallback(GLFWwindow*, double mouseX, double mouseY) {
        static double oldX, oldY;
        float dX = static_cast<float>(mouseX - oldX);
        float dY = static_cast<float>(mouseY - oldY);
        oldX = mouseX;
        oldY = mouseY;

        if (buttons[2] || (buttons[0] && buttons[1])) {
            camera.Pan(-dX * 0.002f, dY * 0.002f);
        }
        else if (buttons[0]) {
            camera.Rotate(dX * -0.01f, dY * 0.01f);
        }
        else if (buttons[1]) {
            camera.Zoom(dY * -0.005f);
        }
    }

    void scrollCallback(GLFWwindow*, double, double yoffset) {
        camera.Zoom(static_cast<float>(yoffset) * 0.04f);
    }
}

void frame(const nxt::SwapChain& swapchain) {
    nxt::Texture backbuffer = swapchain.GetNextTexture();

    backbuffer.TransitionUsage(nxt::TextureUsageBit::Present);
    swapchain.Present(backbuffer);
    DoFlush();
}

int main(int argc, const char* argv[]) {
    if (!InitializeBackend(argc, argv)) {
        return 1;
    }

    nxt::Device device = CreateCppNXTDevice();
    nxt::Queue queue = device.CreateQueueBuilder().GetResult();

    GLFWwindow* window = GetGLFWWindow();
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    ViewportThread viewportThread(device, queue, camera);

    while (!ShouldQuit()) {
        glfwPollEvents();
        utils::USleep(16000);
    }

    viewportThread.Quit();
    viewportThread.join();
}
