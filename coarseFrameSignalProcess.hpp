#pragma once
#include "parameters.hpp"
#include "kernel.cuh"
#include <chrono>
using Clock = std::chrono::steady_clock;
# define __TIC__(tag) auto __##tag##_start_time = Clock::now();

#define __TOC__(tag)                                                        \
auto __##tag##_end_time = Clock::now();                                     \
std::cout << #tag << ";"                                                    \
                  << std::chrono:duration_cast<std::chrono::microseconds>(  \
                        __##tag##_end_time - __##tag__start_time)           \
                        .count()                                            \
                  << "us" << std::endl;

void func_beam_union(std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, float& rangeVal, float& velVel, float& ConReginMatVal, uint16_t& tarcnt, uint8_t& CompareIdx, uint8_t& MaxValIdx);
void func_getConRegin(std::unique_ptr<float[]>& TOI_CFAR, std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, uint16_t TarNum_Detected, uint16* peak_R, uint16_t* peak_V, float* peak_Val, float* peak_SNR, ParaSys& para_sys, float* SpatialFFTVelSel_VeloNum_RangeNum);
void FFTD_SpatialFFT_CFAR_CoarseFrame(std::unique_ptr<float[]>& TOI_CFAR, std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, ParaSys& para_sys, VirtualArray& Virtual_array, std::unique_ptr<float[]>& compensate, float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device);
void func_signal_process_coarse(std::unique_ptr<float[]>& TOI_CFAR, std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, std::string filename, ParaSys& para_sys, VirtualArray& Virtual_array, std::unique_ptr<int16_t[]& input_data, std::unique_ptr<float[]>& compensate);
void func_signal_process_fine();
