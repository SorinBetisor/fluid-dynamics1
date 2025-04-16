// Stub file for vulkan_video_codec_av1std.h
// This is a minimal implementation to allow compilation without the actual video codec headers

#ifndef VULKAN_VIDEO_CODEC_AV1STD_H_
#define VULKAN_VIDEO_CODEC_AV1STD_H_

#ifdef __cplusplus
extern "C" {
#endif

// Define minimal types needed
typedef enum StdVideoAV1ChromaFormatIdc {
    STD_VIDEO_AV1_CHROMA_FORMAT_IDC_MONOCHROME = 0,
    STD_VIDEO_AV1_CHROMA_FORMAT_IDC_420 = 1,
    STD_VIDEO_AV1_CHROMA_FORMAT_IDC_422 = 2,
    STD_VIDEO_AV1_CHROMA_FORMAT_IDC_444 = 3,
    STD_VIDEO_AV1_CHROMA_FORMAT_IDC_INVALID = 0x7FFFFFFF
} StdVideoAV1ChromaFormatIdc;

typedef enum StdVideoAV1ProfileIdc {
    STD_VIDEO_AV1_PROFILE_IDC_MAIN = 0,
    STD_VIDEO_AV1_PROFILE_IDC_HIGH = 1,
    STD_VIDEO_AV1_PROFILE_IDC_PROFESSIONAL = 2,
    STD_VIDEO_AV1_PROFILE_IDC_INVALID = 0x7FFFFFFF
} StdVideoAV1ProfileIdc;

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_AV1STD_H_ 