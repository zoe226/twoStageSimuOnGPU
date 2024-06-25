#include "coarseFrameSignalProcess.hpp"
#include "kernel.cuh"
#include "parameters.hpp"
#include <string>
#include <fstream>

void func_beam_union(std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, float& rangeVal, float& velVel, float& ConReginMatVal, uint16_t& tarcnt, uint8_t& CompareIdx, uint8_t& MaxValIdx)
{

}
void func_getConRegin(std::unique_ptr<float[]>& TOI_CFAR, std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, uint16_t TarNum_Detected, uint16* peak_R, uint16_t* peak_V, float* peak_Val, float* peak_SNR, ParaSys& para_sys, float* SpatialFFTVelSel_VeloNum_RangeNum)
{

}
void FFTD_SpatialFFT_CFAR_CoarseFrame(std::unique_ptr<float[]>& TOI_CFAR, std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, ParaSys& para_sys, VirtualArray& Virtual_array, std::unique_ptr<float[]>& compensate, float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device)
{

}
void func_signal_process_coarse(std::unique_ptr<float[]>& TOI_CFAR, std::unique_ptr<float[]>& TOI_ConRegion, uint16_t& TOI_Count, std::string filename, ParaSys& para_sys, VirtualArray& Virtual_array, std::unique_ptr<int16_t[]& input_data, std::unique_ptr<float[]>& compensate)
{

}
void func_signal_process_fine()
{

}