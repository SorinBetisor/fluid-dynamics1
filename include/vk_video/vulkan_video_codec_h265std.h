// Stub file for vulkan_video_codec_h265std.h
// This is a minimal implementation to allow compilation without the actual video codec headers

#ifndef VULKAN_VIDEO_CODEC_H265STD_H_
#define VULKAN_VIDEO_CODEC_H265STD_H_

#ifdef __cplusplus
extern "C" {
#endif

// Define minimal types needed
typedef enum StdVideoH265ChromaFormatIdc {
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_MONOCHROME = 0,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_420 = 1,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_422 = 2,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_444 = 3,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_INVALID = 0x7FFFFFFF
} StdVideoH265ChromaFormatIdc;

typedef enum StdVideoH265ProfileIdc {
    STD_VIDEO_H265_PROFILE_IDC_MAIN = 1,
    STD_VIDEO_H265_PROFILE_IDC_MAIN_10 = 2,
    STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE = 3,
    STD_VIDEO_H265_PROFILE_IDC_INVALID = 0x7FFFFFFF
} StdVideoH265ProfileIdc;

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_H265STD_H_ 