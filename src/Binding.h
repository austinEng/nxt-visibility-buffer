
#include <nxt/nxtcpp.h>
#include <nxt/nxt_wsi.h>

bool InitializeBackend(int argc, const char** argv);
void DoFlush();
bool ShouldQuit();

struct GLFWwindow;
struct GLFWwindow* GetGLFWWindow();

nxt::Device CreateCppNXTDevice();
uint64_t GetSwapChainImplementation();
nxt::SwapChain GetSwapChain(const nxt::Device& device);