// Stub file for vulkan_video_codec_h264std.h
// This is a minimal implementation to allow compilation without the actual video codec headers

#ifndef VULKAN_VIDEO_CODEC_H264STD_H_
#define VULKAN_VIDEO_CODEC_H264STD_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Define minimal types needed
typedef enum StdVideoH264ChromaFormatIdc {
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME = 0,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_420 = 1,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_422 = 2,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_444 = 3,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_INVALID = 0x7FFFFFFF
} StdVideoH264ChromaFormatIdc;

typedef enum StdVideoH264ProfileIdc {
    STD_VIDEO_H264_PROFILE_IDC_BASELINE = 66,
    STD_VIDEO_H264_PROFILE_IDC_MAIN = 77,
    STD_VIDEO_H264_PROFILE_IDC_HIGH = 100,
    STD_VIDEO_H264_PROFILE_IDC_INVALID = 0x7FFFFFFF
} StdVideoH264ProfileIdc;

typedef enum StdVideoH264LevelIdc {
    STD_VIDEO_H264_LEVEL_IDC_1_0 = 0,
    STD_VIDEO_H264_LEVEL_IDC_1_1 = 1,
    STD_VIDEO_H264_LEVEL_IDC_1_2 = 2,
    STD_VIDEO_H264_LEVEL_IDC_1_3 = 3,
    STD_VIDEO_H264_LEVEL_IDC_2_0 = 4,
    STD_VIDEO_H264_LEVEL_IDC_2_1 = 5,
    STD_VIDEO_H264_LEVEL_IDC_2_2 = 6,
    STD_VIDEO_H264_LEVEL_IDC_3_0 = 7,
    STD_VIDEO_H264_LEVEL_IDC_3_1 = 8,
    STD_VIDEO_H264_LEVEL_IDC_3_2 = 9,
    STD_VIDEO_H264_LEVEL_IDC_4_0 = 10,
    STD_VIDEO_H264_LEVEL_IDC_4_1 = 11,
    STD_VIDEO_H264_LEVEL_IDC_4_2 = 12,
    STD_VIDEO_H264_LEVEL_IDC_5_0 = 13,
    STD_VIDEO_H264_LEVEL_IDC_5_1 = 14,
    STD_VIDEO_H264_LEVEL_IDC_5_2 = 15,
    STD_VIDEO_H264_LEVEL_IDC_6_0 = 16,
    STD_VIDEO_H264_LEVEL_IDC_6_1 = 17,
    STD_VIDEO_H264_LEVEL_IDC_6_2 = 18,
    STD_VIDEO_H264_LEVEL_IDC_INVALID = 0x7FFFFFFF
} StdVideoH264LevelIdc;

typedef struct StdVideoH264SequenceParameterSet {
    uint32_t flags;
    StdVideoH264ProfileIdc profile_idc;
    StdVideoH264LevelIdc level_idc;
    StdVideoH264ChromaFormatIdc chroma_format_idc;
    uint8_t seq_parameter_set_id;
    uint8_t bit_depth_luma_minus8;
    uint8_t bit_depth_chroma_minus8;
    uint8_t log2_max_frame_num_minus4;
    uint8_t pic_order_cnt_type;
    uint8_t log2_max_pic_order_cnt_lsb_minus4;
    uint8_t max_num_ref_frames;
    uint8_t reserved1;
} StdVideoH264SequenceParameterSet;

typedef struct StdVideoH264PictureParameterSet {
    uint32_t flags;
    uint8_t seq_parameter_set_id;
    uint8_t pic_parameter_set_id;
    uint8_t num_ref_idx_l0_default_active_minus1;
    uint8_t num_ref_idx_l1_default_active_minus1;
} StdVideoH264PictureParameterSet;

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_H264STD_H_ 