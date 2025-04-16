// Stub file for vulkan_video_codecs_common.h
// This is a minimal implementation to allow compilation without the actual video codec headers

#ifndef VULKAN_VIDEO_CODECS_COMMON_H_
#define VULKAN_VIDEO_CODECS_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

// Define minimal types needed for Vulkan video codec support
typedef enum StdVideoFormat {
    STD_VIDEO_FORMAT_UNDEFINED = 0,
    STD_VIDEO_FORMAT_420 = 1,
    STD_VIDEO_FORMAT_422 = 2,
    STD_VIDEO_FORMAT_444 = 3,
    STD_VIDEO_FORMAT_INVALID = 0x7FFFFFFF
} StdVideoFormat;

typedef struct StdVideoDecodeCapabilities {
    uint32_t flags;
    uint32_t maxLevel;
    uint32_t maxSlicesPerFrame;
} StdVideoDecodeCapabilities;

typedef struct StdVideoEncodeCapabilities {
    uint32_t flags;
    uint32_t maxSlicesPerFrame;
} StdVideoEncodeCapabilities;

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODECS_COMMON_H_ 