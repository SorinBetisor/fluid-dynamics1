#include "execPath.h"
#include <filesystem>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

std::string getExecutableDir() {
  namespace fs = std::filesystem;

  fs::path exePath;
#if defined(_WIN32)
  wchar_t buf[MAX_PATH];
  DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  if (len == 0 || len == MAX_PATH)
    throw std::runtime_error("GetModuleFileNameW failed");
  exePath = fs::path(buf, buf + len);

#elif defined(__linux__)
  std::string buf(1024, '\0');
  ssize_t len = readlink("/proc/self/exe", buf.data(), buf.size());
  if (len < 0)
    throw std::runtime_error("readlink(/proc/self/exe) failed");
  exePath = fs::canonical(buf.substr(0, len));

#else
#error Unsupported platform
#endif

  return exePath.parent_path().string();
}
