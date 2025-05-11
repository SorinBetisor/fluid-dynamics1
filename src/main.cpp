#include <vk_engine.h>

int main(int argc, char *argv[]) {
  VulkanEngine engine;

  engine.init();

  engine.run_simulation_loop();

  engine.cleanup();

  return 0;
}
