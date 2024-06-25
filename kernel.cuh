#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parameters.hpp"
#include <cstdint>

__global__ void rectWin_kernel(float* winCoef, uint16_t winLen);
__global__ void hanningWin_kernel(float* winCoef, uint16_t winLen);
__global__ void hammingWin_kernel(float* winCoef, uint16_t winLen);
__global__ void addWin_kernel(float* WinRout_RangeSampleNum_ChirpNum_RxNum_device, int16_t* radarInputdata_device, float* win_Coef_device, uint16_t RangeSampleNum, uint16_t ChirpNum, uint16_t RxNum);
__global__ void dataReshape_afterFFTR_kernel(float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum, float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device, uint16_t* all_tx_seq_pos_device, uint16_t TxNum, uint16_t RxNum, uint16_t TxReuseNum, uint16_t CoarseRangeNum);
__global__ void addWinD_kernel(float2* WinDout_ChirpNum_RangeSampleNum_MIMONum_device, float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device, float* win_Coef_device, uint16_t VelocityNum, uint16_t CoarseRangeNum, uint16_t MimoNum);
__global__ void fftshift(float2* indata, uint16_t dim1, uint16_t dim2, uint16_t dim3);
__global__ void spatial_reorder_kernel(float2* result_xNum_yNum_VeloFFTNum_RangeNum, float2* FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, uint16_t* pos_in_mat, uint16_t MimoNum, uint16_t VelocityNum, uint16_t CoarseRangeNum, uint16_t VirtArrHorGridLen, uint16_t VirtArrVertGridLen);
__global__ void addWinA_kernel(float2* winAout_xNum_yNum_VeloFFTNum_RangeNum_device, float2* result_xNum_yNum_VeloFFTNum_RangeNum_device, float* win_Coef_device, uint16_t VirtArrHorGridLen, uint16_t VirtArrVertGridLen, uint16_t VelocityNum, uint16_t CoarseRangeNum, uint16_t AngleHorNum);
__global__ void fftshift_ffta(float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t AngleHorNum, uint16_t VirtArrVertGridLen, uint16_t VelocityNum, uint16_t CoarseRangeNum);
__global__ void ffta_abs_kernel(float* SpatialFFTA_ABS_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t AngleHorNum, uint16_t VirtArrVertGridLen, uint16_t VelocityNum, uint16_t CoarseRangeNum);
__global__ void cfar3d_cal_across_ArbitaryDim_kernel(float* SpatialFFTVelSel_VeloNum_RangeNum_device, float* SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t AngleHorNum, uint16_t VelocityNum, uint16_t CoarseRangeNum);
__global__ void peak_search_kernel(uint8_t* isPeak, float* SpatialFFTVelSel_VeloNum_RangeNum_device, uint16_t DetectCell_RIndex_Min_u10, uint16_t DetectCell_RIndex_Max_u10, uint16_t DetectCell_VIndex_Min_u11, uint16_t DetectCell_VIndex_Max_u11, uint16_t ChirpNum_u11, uint16_t RangeCellNum_u10);
__global__ void CFARChM_OS_1D_kernel(uint8_t* IsTarget_1D_R_u1, float* RSNR_u11, uint16_t RangeCellNum_u10, uint16_t ChirpNum_u11, uint8_t ProCellNum_R_u2, uint8_t RefCellNum_1D_u5, uint8_t Loc_OSCFAR_u5, uint8_t* isPeak, float* SpatialFFTVelSel_VelNum_RangeNum_device);
__global__ void CFARChm_OS_1D_V_kernel(uint8_t* IsTarget_1D_V_u1, float* VSNR_u11, uint8_t LogicTestFlag_u1, uint16_t RangeCellNum_u10, uint16_t ChirpNum_u11, uint8_t ProCellNum_V_u2, uint8_t RefCellNum_1D_u5, uint8_t Loc_OSCFAR_u5, uint8_t* IsTarget_1D_R_u1, float* SpatialFFTVelSel_VelNum_RangeNum_device);
__global__ postProcess_kernel(uint16_t ChirpNum_u11, uint16_t RangeCellNum_u10, uint16_t Index_Chirp_NotMove_OSCFAR_u11, uint16_t Threshold_RangeDim_For_2D_OSCFAR_u9, uint8_t* IsTarget_1D_R_u1, uint8_t* IsTarget_1D_V_u1, float* SpatialFFTVelSel_VeloNum_RangeNum_device);
void_winR_process(float* WinRout_RangeSampleNum_ChirpNum_RxNum_device, int16_t* radarInputdata_device, float* win_Coef_device, uint8_t& fft_win_type, uint16_t& RangeSampleNum, uint16_t& ChirpNum, uint16_t& RxNum);
void func_fftR_process(float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device, float* WinRout_RangeSampleNum_ChirpNum_RxNum_device, uint16_t& RangeSampleNum, uint16_t& VelocityNum, uint16_t& RxNum);
void func_dataReshape_afterFFTR(float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device, float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device, uint16_t* all_tx_seq_pos, uint16_t& TxNum, uint16_t& RxNum, uint16_t& TxReuseNum, uint16_t& CoarseRangeNum);
void func_winD_process(float2* WinDout_ChirpNum_RangeSampleNum_MIMONum_device, float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device, float* win_Coef_device, uint8_t& fft_win_type, uint16_t& VelocityNum, uint16_t& CoarseRangeNum, uint16_t& MimoNum);
void func_fftD_process(float2* FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, float2* WinDout_ChirpNum_RangeSampleNum_MIMONum_device, uint16_t& VelocityNum, uint16_t& CoarseRangeNum, uint16_t& MIMONum);
void func_Spatial_Reorder(float2* result_xNum_yNum_VeloFFTNum_RangeNum, float2* FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, uint8_t& Array_option, uint16_t& MimoNum, uint16_t* pos_in_mat, uint16_t& VirtArrHorGridLen, uint16_t& VirtArrVertGridLen, uint16_t& VelocityNum,uint16_t& CoarseRangeNum);
void func_winA_process(float2* winAout_xNum_yNum_VeloFFTNum_RangeNum_device, float2* result_xNum_yNum_VeloFFTNum_RangeNum_device, float* win_Coef_device, uint8_t& fft_win_type, uint16_t& VirtArrHorGridLen, uint16_t VirtArrVertGridLen, uint16_t& VelocityNum, uint16_t& CoarseRangeNum, uint16_t& AngleHorNum);
void func_fftA_process(float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, float2* winAout_xNum_yNum_VeloFFTNum_RangeNum_device, uint16_t& AngleHorNum, uint16_t& VirtArrVertGridLen, uint16_t& VelocityNum, uint16_t& CoarseRangeNum);
void func_abs_process(float* SpatialFFTA_ABS_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t& AngleHorNum, uint16_t& VirtArrVertGridLen, uint16_t& VelocityNum, uint16_t CoarseRangeNum);
void func_cfar3d_cal_across_ArbitaryDim(float* SpatialFFTVelSel_VeloNum_RangeNum_device, float* SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint8_t& SqueezeDim, uint8_t& cfar_include_order, uint8_t& cfar_exclude_order, float& snr_dB_different_dim, uint8_t& Switch3DMode, uint16_t& AngleHorNum, uint16_t& VelocityNum, uint16_t& CoarseRangeNum);
void func_PeakSearch_And_CFAR_2D_Cross(uint16_t& TarNum_Detected, uint16_t* peak_R, uint16_t* peak_V, float* peak_Val, float* peak_SNR, DetPara& det_para, float* SpatialFFTVelSel_VeloNum_RangeNum_device);