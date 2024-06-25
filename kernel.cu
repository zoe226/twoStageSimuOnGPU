#include "kernel.cuh"
#include <cmath>
#include <cufft.h>
#define M_PI 3.141592653589793

__global__ void rectWin_kernel(float* winCoef, uint16_t winLen)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if(tidx < winLen)
    {
        winCoef[tidx] = 1.0;
    }
}

__global__ void hanningWin_kernel(float* winCoef, uint16_t winLen)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if(tidx < winLen){
        winCoef[tidx] = 0.5*(1 - cos(2 * M_PI * tidx / (winLen - 1)));
    }
}

__global__ void hammingWin_kernel(float* winCoef, uint16_t winLen)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < winLen)
    {
        winCoef[tidx] = 0.54 - 0.46 * cos(2 * M_PI * tidx / (winLen - 1));
    }
}

__global__ void addWin_kernel(float* WinRout_RangeSampleNum_ChirpNum_RxNum_device, int16_t* radarInputdata_device, float* win_Coef_device, uint16_t RangeSampleNum, uint16_t ChirpNum, uint16_t RxNum)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tix = threadIdx.z + blockDim.z * blockIdx.z;

    if (tix < RangeSampleNum && tiy < RxNum && tiz < ChirpNum)
    {
        WinRout_RangeSampleNum_ChirpNum_RxNum_device[tix + tiz * RangeSampleNum + tiy * RangeSampleNum * ChirpNum] = radarInputdata_device[tix + tiy * RangeSampleNum + tiz * RangeSampleNum * RxNum] * win_Coef_device[tix];
    } 
}

__global__ void dataReshape_afterFFTR_kernel(float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum, float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device, uint16_t* all_tx_seq_pos_device, uint16_t TxNum, uint16_t RxNum, uint16_t TxReuseNum, uint16_t CoarseRangeNum)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tiz = threadIdx.z + blockDim.z * blockIdx.z;
    int velocityNum = TxNum * TxReuseNum;
    if(tix < CoarseRangeNum && tiy < velocityNum && tiz < RxNum)
    {
        int txIdx = tiy / TxReuseNum;
        int chirpLoc = tiy % TxReuseNum;
        int chirpIdx = all_tx_seq_pos_device[chirpLoc + txIdx * TxReuseNum] - 1;
        CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum[chirpIdx + tix * velocityNum + tiz * velocityNum * CoarseRangeNum] = CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device[tix + chirpIdx * (CoarseRangeNum + 1) + tiz * (CoarseRangeNum + 1) * VelocityNum];
    }
}

__global__ void addWinD_kernel(float2* WinDout_ChirpNum_RangeSampleNum_MIMONum_device, float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device, float* win_Coef_device, uint16_t VelocityNum, uint16_t CoarseRangeNum, uint16_t MimoNum)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tiz = threadIdx.z + blockDim.z * blockIdx.z;

    if (tix < VelocityNum && tiy < CoarseRangeNum && tiz < MimoNum)
    {
        WinDout_ChirpNum_RangeSampleNum_MIMONum_device[tix + tiy * VelocityNum + tiz * VelocityNum * CoarseRangeNum].x = CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device[tix + tiy * VelocityNum + tiz * VelocityNum * CoarseRangeNum].x * win_Coef_device[tix];
        WinDout_ChirpNum_RangeSampleNum_MIMONum_device[tix + tiy * VelocityNum + tiz * VelocityNum * CoarseRangeNum].y = CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device[tix + tiy * VelocityNum + tiz * VelocityNum * CoarseRangeNum].y * win_Coef_device[tix];
    } 
}

__global__ void fftshift(float2* indata, uint16_t dim1, uint16_t dim2, uint16_t dim3)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tiz = threadIdx.z + blockDim.z * blockIdx.z;
    float2 temp;
    if (tix < dim1/2 && tiy < dim2 && tiz < dim3)
    {
        temp.x = indata[tix + tiy * dim1 + tiz * dim1 * dim2].x;
        temp.y = indata[tix + tiy * dim1 + tiz * dim1 * dim2].y;

        indata[tix + tiy * dim1 + tiz * dim1 * dim2].x = indata[tix + dim1 / 2 + tiy * dim1 + tiz * dim1 * dim2].x;
        indata[tix + tiy * dim1 + tiz * dim1 * dim2].y = indata[tix + dim1 / 2 + tiy * dim1 + tiz * dim1 * dim2].y;

        indata[tix + dim1 / 2 + tiy * dim1 + tiz * dim1 * dim2].x = temp.x;
        indata[tix + dim1 / 2 + tiy * dim1 + tiz * dim1 * dim2].y = temp.y;
    }
}

__global__ void spatial_reorder_kernel(float2* result_xNum_yNum_VeloFFTNum_RangeNum, float2* FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, uint16_t* pos_in_mat, uint16_t MimoNum, uint16_t VelocityNum, uint16_t CoarseRangeNum, uint16_t VirtArrHorGridLen, uint16_t VirtArrVertGridLen)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tiz = threadIdx.z + blockDim.z * blockIdx.z;

    if (tix < VelocityNum && tiy < CoarseRangeNum && tiz < MimoNum)
    {
        int xIdx = pos_in_mat[2 * tiz];
        int yIdx = pos_in_mat[2 * tiz + 1];
        result_xNum_yNum_VeloFFTNum_RangeNum[xIdx + yIdx * VirtArrHorGridLen + tix * VirtArrHorGridLen * VirtArrVertGridLen + tiy * VirtArrHorGridLen * VirtArrVertGridLen * VelocityNum] = FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device[tix + tiy * VelocityNum + tiz * VelocityNum * CoarseRangeNum];
    }
}

__global__ void addWinA_kernel(float2* winAout_xNum_yNum_VeloFFTNum_RangeNum_device, float2* result_xNum_yNum_VeloFFTNum_RangeNum_device, float* win_Coef_device, uint16_t VirtArrHorGridLen, uint16_t VirtArrVertGridLen, uint16_t VelocityNum, uint16_t CoarseRangeNum, uint16_t AngleHorNum)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tiz = threadIdx.z + blockDim.z * blockIdx.z;
    if (tix < VirtArrHorGridLen && tiy < VelocityNum && tiz < CoarseRangeNum)
    {
        winAout_xNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x = result_xNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * VirtArrHorGridLen * VirtArrVertGridLen + tiz * VirtArrHorGridLen * VirtArrVertGridLen * VelocityNum].x * win_Coef_device[tix];
        winAout_xNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y = result_xNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * VirtArrHorGridLen * VirtArrVertGridLen + tiz * VirtArrHorGridLen * VirtArrVertGridLen * VelocityNum].y * win_Coef_device[tix];
        winAout_xNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum * (VirtArrVertGridLen - 1)  + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x = result_xNum_yNum_VeloFFTNum_RangeNum_device[tix + VirtArrHorGridLen * (VirtArrVertGridLen - 1) + tiy * VirtArrHorGridLen * VirtArrVertGridLen + tiz * VirtArrHorGridLen * VirtArrVertGridLen * VelocityNum].x * win_Coef_device[tix];
        winAout_xNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum * (VirtArrVertGridLen - 1)  + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y = result_xNum_yNum_VeloFFTNum_RangeNum_device[tix + VirtArrHorGridLen * (VirtArrVertGridLen - 1) + tiy * VirtArrHorGridLen * VirtArrVertGridLen + tiz * VirtArrHorGridLen * VirtArrVertGridLen * VelocityNum].y * win_Coef_device[tix];
    }
}

__global__ void fftshift_ffta(float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t AngleHorNum, uint16_t VirtArrVertGridLen, uint16_t VelocityNum, uint16_t CoarseRangeNum)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;
    int tiz = threadIdx.z + blockDim.z * blockIdx.z;

    float2 temp1;
    float2 temp2;
    if(tix < AngleHorNum/2 && tiy < VelocityNum && tiz < CoarseRangeNum)
    {
        temp1.x = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x;
        temp1.y = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y;
        
        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x;
        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y;        

        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x = temp1.x;
        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y = temp1.y;

        temp2.x = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x;
        temp2.y = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y;

        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x;
        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y = SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y;

        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].x = temp2.x;
        SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tix + AngleHorNum / 2 + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum].y = temp2.y;
    }
}

__global__ void ffta_abs_kernel(float* SpatialFFTA_ABS_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t AngleHorNum, uint16_t VirtArrVertGridLen, uint16_t VelocityNum, uint16_t CoarseRangeNum)
{
    int tix = threadIdx.x + blockDim.x * blockIdx.x;
    int tiy = threadIdx.y + blockDim.y * blockIdx.y;

    if (tix < AngleHorNum && tiy < VelocityNum && tiz < CoarseRangeNum)
    {
        int index1 = tix + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum;
        int index2 = tix + (VirtArrVertGridLen - 1) * AngleHorNum + tiy * AngleHorNum * VirtArrVertGridLen + tiz * AngleHorNum * VirtArrVertGridLen * VelocityNum;
        int index3 = tix + tiy * AngleHorNum + tiz * AngleHorNum * VelocityNum;
        float y1 = sqrt(powf(SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[index1].x,2) + powf(SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[index1].y,2));
        float y2 = sqrt(powf(SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[index2].x,2) + powf(SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[index2].y,2));
        SpatialFFTA_ABS_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[index3] = (y1 + y2) / VirtArrVertGridLen;
    }
}

__global__ void cfar3d_cal_across_ArbitaryDim_kernel(float* SpatialFFTVelSel_VeloNum_RangeNum_device, float* SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t AngleHorNum, uint16_t VelocityNum, uint16_t CoarseRangeNum)
{
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    int tiz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tiy < VelocityNum && tiz < CoarseRangeNum)
    {
        float temp = SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tiy * AngleHorNum + tiz * VelocityNum * AngleHorNum];
        for (size_t i = 0; i < AngleHorNum; i++)
        {
            if (temp < SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[i + tiy * AngleHorNum + tiz * VelocityNum * AngleHorNum])
            {
                temp = SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[i + tiy * AngleHorNum + tiz * VelocityNum * AngleHorNum];
            }
        }
        SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device[tiy + tiz * VelocityNum] = temp;
    }
}

__global__ void peak_search_kernel(uint8_t* isPeak, float* SpatialFFTVelSel_VeloNum_RangeNum_device, uint16_t DetectCell_RIndex_Min_u10, uint16_t DetectCell_RIndex_Max_u10, uint16_t DetectCell_VIndex_Min_u11, uint16_t DetectCell_VIndex_Max_u11, uint16_t ChirpNum_u11, uint16_t RangeCellNum_u10)
{
    // peak search enable and peakSearchWin = 1
    int tix = blockDim.x * blockIdx.x + threadIdx.x;
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tix < DetectCell_RIndex_Max_u10 && tix >= (DetectCell_RIndex_Min_u10 - 1) && tiy < DetectCell_VIndex_Max_u11 && tiy >= (DetectCell_VIndex_Min_u11-1))
    {
        int V_Upboundary_u11;
        int V_Backboundary_u11;
        float DataToDetect = SpatialFFTVelSel_VeloNum_RangeNum_device[tix + tiy * ChirpNum_u11];
        if (tix < (ChirpNum_u11 - 1))
        {
            V_Upboundary_u11 = tix + 1;
        }
        else
        {
            V_Upboundary_u11 = 0;
        }
        if (tix > 0)
        {
            V_Backboundary_u11 = tix - 1;
        }
        else
        {
            V_Backboundary_u11 = ChirpNum_u11 - 1;
        }
        if (tiy == 0)
        {
            isPeak[tix + tiy * ChirpNum_u11] = (DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[tix + (tiy + 1) * ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + (tiy + 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + (tiy + 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + tiy*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + tiy*ChirpNum_u11])  
        }
        else if(tiy == RangeCellNum_u10 - 1)
        {
            isPeak[tix + tiy * ChirpNum_u11] = (DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[tix + (tiy - 1) * ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + (tiy - 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + (tiy - 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + tiy*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + tiy*ChirpNum_u11]);
        }
        else
        {
            isPeak[tix + tiy * ChirpNum_u11] = (DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[tix + (tiy + 1) * ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + (tiy + 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + (tiy + 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[tix + (tiy - 1) * ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + (tiy - 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + (tiy - 1)*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Upboundary_u11 + tiy*ChirpNum_u11] && DataToDetect > SpatialFFTVelSel_VeloNum_RangeNum_device[V_Backboundary_u11 + tiy*ChirpNum_u11]);
        }
    } 
}

__global__ void CFARChM_OS_1D_kernel(uint8_t* IsTarget_1D_R_u1, float* RSNR_u11, uint16_t RangeCellNum_u10, uint16_t ChirpNum_u11, uint8_t ProCellNum_R_u2, uint8_t RefCellNum_1D_u5, uint8_t Loc_OSCFAR_u5, uint8_t* isPeak, float* SpatialFFTVelSel_VelNum_RangeNum_device)
{
    int tix = blockDim.x * blockIdx.x + threadIdx.x;
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tix < ChirpNum_u11 && tiy < RangeCellNum_u10)
    {
        if (isPeak[tix + tiy * ChirpNum_u11])
        {
            uint16_t LeftBoundary_u9 = ProCellNum_R_u2 + RefCellNum_1D_u5 - 1;
            uint16_t RightBoundary_u10 = RangeCellNum_u10 - ProCellNum_R_u2 - RefCellNum_1D_u5;
            uint8_t Logic1_u1 = 0;
            uint8_t Logic2_u1 = 0;
            uint8_t RefCellNum_u6 = 0;
            float DataSet_RefCell_u30[32] = {0.0};
            if (LeftBoundary_u9 >= RightBoundary_u10)
            {
                ProCellNum_R_u2 = 1;
                RefCellNum_1D_u5 = 1;
                LeftBoundary_u9 = ProCellNum_R_u2 + RefCellNum_1D_u5 - 1;
                RightBoundary_u10 = RangeCellNum_u10 - ProCellNum_R_u2 - RefCellNum_1D_u5;
            }

            if (tiy > LeftBoundary_u9)
            {
                Logic1_u1 = 1;
            }
            if (tiy < RightBoundary_u10)
            {
                Logic2_u1 = 1;
            }
            
            if (Logic1_u1 == 1 && Logic2_u1 == 1)
            {
                RefCellNum_u6 = 2 * RefCellNum_1D_u5;
                for (uint8_t i = 0; i < RefCellNum_u6; i++)
                {
                    if ((i%2) == 1)
                    {
                        uint16_t leftRefIndex = tiy - ProCellNum_R_u2 - RefCellNum_1D_u5 + i / 2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + leftRefIndex * ChirpNum_u11];
                    }
                    else
                    {
                        uint16_t rightRefIndex = tiy + ProCellNum_R_u2 + 1 + i / 2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + rightRefIndex * ChirpNum_u11];
                    }
                }
            }
            else if(Logic2_u1 == 1)
            {
                RefCellNum_u6 = RefCellNum_1D_u5 + tiy - ProCellNum_R_u2 - 1;
                if (RefCellNum_u6 < RefCellNum_1D_u5)
                {
                    RefCellNum_u6 = RefCellNum_1D_u5
                }
                for (uint8_t i = 0; i < RefCellNum_1D_u5; i++)
                {
                    uint16_t rightRefIndex = tiy + ProCellNum_R_u2 + 1 + i;
                    DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + rightRefIndex * ChirpNum_u11];
                }
                for (uint8_t i = RefCellNum_1D_u5; i < RefCellNum_u6; i++)
                {
                    uint16_t leftRefIndex = 1 + i - RefCellNum_1D_u5;
                    DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + leftRefIndex * ChirpNum_u11];
                }
            }
            else
            {
                RefCellNum_u6 = RefCellNum_1D_u5 + RangeCellNum_u10 - tiy - ProCellNum_R_u2 - 1;
                for (uint16_t i = 0; i < RefCellNum_1D_u5; i++)
                {
                    uint16_t leftRefIndex = tiy - ProCellNum_R_u2 - RefCellNum_1D_u5 + i;
                    DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + leftRefIndex * ChirpNum_u11];
                }
                for (uint16_t i = RefCellNum_1D_u5; i < RefCellNum_u6; i++)
                {
                    uint16_t rightRefIndex = tiy + ProCellNum_R_u2 + 1 + i - RefCellNum_1D_u5;
                    SpatialFFTVelSel_VelNum_RangeNum_device[tix + rightRefIndex * ChirpNum_u11];
                }
            }
            if (Loc_OSCFAR_u5 >= RefCellNum_u6)
            {
                Loc_OSCFAR_u5 = floor(double(RefCellNum_u6 / 2) + 1);
            }
            
            float Data_OSLoc_u30;
            Data_OSLoc_u30 = DataSet_RefCell_u30[RefCellNum_u6 - 1];
            uint8_t LeftLoc_u1 = 1;
            uint8_t LocComp_u5 = Loc_OSCFAR_u5 - 1;
            uint8_t middleLoc = floor(double(RefCellNum_u6 / 2));
            if (Loc_OSCFAR_u5 > middleLoc)
            {
                LeftLoc_u1 = 0;
                LocComp_u5 = RefCellNum_u6 - Loc_OSCFAR_u5;
            }
            for (uint8_t i = 0; i < RefCellNum_u6 - 1; i++)
            {
                float Cell_ToCompare_u30 = DataSet_RefCell_u30[i];
                uint8_t IsData_u1 = 0;
                uint8_t Count_u5 = 0;
                uint8_t Data_Equal_u5 = 0;
                uint8_t flag_Not_u1 = 0;
                for (uint8_t j = 0; j < RefCellNum_u6; j++)
                {
                    float Cell_Temp_u30 = DataSet_RefCell_u30[j];
                    if (LeftLoc_u1 == 1)
                    {
                        if (Cell_ToCompare_u30 > Cell_Temp_u30)
                        {
                            Count_u5 = Count_u5 + 1;
                        }
                        
                    }
                    else
                    {
                        if(Cell_ToCompare_u30 < Cell_Temp_u30)
                        {
                            Count_u5 = Count_u5 + 1;
                        }
                    }
                    if (Cell_ToCompare_u30 == Cell_Temp_u30)
                    {
                        Data_Equal_u5 = Data_Equal_u5 + 1;
                    }
                    if(Count_u5 > LocComp_u5)
                    {
                        flag_Not_u1 = 1;
                        break;
                    }
                }
                if (flag_Not_u1 == 0)
                {
                    if (Data_Equal_u5 == 1)
                    {
                        if (Count_u5 == LocComp_u5)
                        {
                            IsData_u1 = 1;
                        }
                        
                    }
                    else
                    {
                        uint8_t NumLess_u5 = Count_u5 + Data_Equal_u5;
                        if (NumLess_u5 > LocComp_u5)
                        {
                            IsData_u1 = 1;
                        }
                    }
                }
                if (IsData_u1 == 1)
                {
                    Data_OSLoc_u30 = Cell_ToCompare_u30;
                    break;
                }
            }
            uint8_t R_Threshold_Num_u6 = 32;
            uint8_t thIdx = floor(double(tiy / (RangeCellNum_u10/R_Threshold_Num_u6)))                    
            float Threshold2_u39 = 5 * Data_OSLoc_u30;
            float CellToDetect_u32 = SpatialFFTVelSel_VelNum_RangeNum_device[tix + tiy * ChirpNum_u11] * 2 * 2;
            if (CellToDetect_u32 > Threshold2_u39)
            {
                IsTarget_1D_R_u1[tix + tiy * ChirpNum_u11] = 1;
                RSNR_u11[tix + tiy * ChirpNum_u11] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + tiy * ChirpNum_u11] / Data_OSLoc_u30;
            } 
        }
    }
}

__global__ void CFARChm_OS_1D_V_kernel(uint8_t* IsTarget_1D_V_u1, float* VSNR_u11, uint8_t LogicTestFlag_u1, uint16_t RangeCellNum_u10, uint16_t ChirpNum_u11, uint8_t ProCellNum_V_u2, uint8_t RefCellNum_1D_u5, uint8_t Loc_OSCFAR_u5, uint8_t* IsTarget_1D_R_u1, float* SpatialFFTVelSel_VelNum_RangeNum_device)
{
    int tix = blockDim.x * blockIdx.x + threadIdx.x;
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tix < ChirpNum_u11 && tiy < RangeCellNum_u10)
    {
        if (IsTarget_1D_R_u1[tix + tiy * ChirpNum_u11] || LogicTestFlag_u1 == 1)
        {
            uint8_t Logic1_u1 = 0;
            uint8_t Logic2_u1 = 0;
            uint8_t Logic3_u1 = 0;
            uint8_t Logic4_u1 = 0;
            uint8_t Logic5_u1 = 0;
            uint16_t UpBoundary_u9 = 1 + ProCellNum_V_u2 + RefCellNum_1D_u5;
            uint16_t DownBoundary_u10 = ChirpNum_u11 - ProCellNum_V_u2 - RefCellNum_1D_u5;
            uint16_t RefCellNum_u6 = 2 * RefCellNum_1D_u5;
            float DataSet_RefCell_u30[32] = {0.0};
            if(UpBoundary_u9 >= DownBoundary_u10)
            {
                ProCellNum_V_u2 = 1;
                RefCellNum_1D_u5 = 1;
                UpBoundary_u9 = 1 + ProCellNum_V_u2 + RefCellNum_1D_u5;
                DownBoundary_u10 = ChirpNum_u11 - ProCellNum_V_u2 - RefCellNum_1D_u5;
            }
            if (tix >= 0 && tix <= 0 + ProCellNum_V_u2)
            {
                Logic1_u1 = 1;
            }
            if(tix > (0 + ProCellNum_V_u2) && tix < (0 + ProCellNum_V_u2 + RefCellNum_1D_u5))
            {
                Logic2_u1 = 1;
            }
            if (tix >= 0 + ProCellNum_V_u2 + RefCellNum_1D_u5 && tix <= ChirpNum_u11 - ProCellNum_V_u2 - RefCellNum_1D_u5 - 1)
            {
                Logic3_u1 = 1;
            }
            if (tix > ChirpNum_u11 - ProCellNum_V_u2 - RefCellNum_1D_u5 - 1 && tix < ChirpNum_u11 - ProCellNum_V_u2 - 1)
            {
                Logic4_u1 = 1;
            }
            if (tix >= ChirpNum_u11 - ProCellNum_V_u2 - 1)
            {
                Logic5_u1 = 1;
            }
            if (Logic1_u1 == 1)
            {
                for (uint8_t i = 0; i < RefCellNum_u6; i++)
                {
                    if (i % 2 == 1)
                    {
                        uint16_t upIndex = ChirpNum_u11 - ProCellNum_V_u2 - RefCellNum_1D_u5 + tix + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[upIndex + tiy * ChirpNum_u11];
                    }
                    else
                    {
                        uint16_t downIndex = tix + ProCellNum_V_u2 + 1 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[downIndex + tiy * ChirpNum_u11];
                    }
                }
                
            }
            else if(Logic2_u1 == 1)            
            {
                for (uint8_t i = 0; i < RefCellNum_u6; i++)
                {
                    if (i % 2 == 1)
                    {
                        uint16_t upIndex = (ChirpNum_u11 - ProCellNum_V_u2 - RefCellNum_1D_u5 + tix + i/2) % ChirpNum_u11;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[upIndex + tiy * ChirpNum_u11];
                    }
                    else
                    {
                        uint16_t downIndex = tix + ProCellNum_V_u2 + 1 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[downIndex + tiy * ChirpNum_u11];
                    }
                }
                
            }
            else if(Logic3_u1 == 1)
            {
                for (uint8_t i = 0; i < RefCellNum_u6; i++)
                {
                    if (i % 2 == 1)
                    {
                        uint16_t upIndex = tix - ProCellNum_V_u2 - RefCellNum_1D_u5 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[upIndex + tiy * ChirpNum_u11];
                    }
                    else
                    {
                        uint16_t downIndex = tix + ProCellNum_V_u2 + 1 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[downIndex + tiy * ChirpNum_u11];
                    }
                }
            }
            else if(Logic4_u1 == 1)
            {
                for (uint8_t i = 0; i < RefCellNum_u6; i++)
                {
                    if (i % 2 == 1)
                    {
                        uint16_t upIndex = tix - ProCellNum_V_u2 - RefCellNum_1D_u5 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[upIndex + tiy * ChirpNum_u11];
                    }
                    else
                    {
                        uint16_t downIndex = (tix + ProCellNum_V_u2 + 1 + i/2) % ChirpNum_u11;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[downIndex + tiy * ChirpNum_u11];
                    }
                }
            }
            else if(Logic5_u1 == 1)
            {
                for (uint8_t i = 0; i < RefCellNum_u6; i++)
                {
                    if (i % 2 == 1)
                    {
                        uint16_t upIndex = tix - ProCellNum_V_u2 - RefCellNum_1D_u5 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[upIndex + tiy * ChirpNum_u11];
                    }
                    else
                    {
                        uint16_t downIndex = tix + ProCellNum_V_u2 + 1 - ChirpNum_u11 + i/2;
                        DataSet_RefCell_u30[i] = SpatialFFTVelSel_VelNum_RangeNum_device[downIndex + tiy * ChirpNum_u11];
                    }
                }
            }
            if (Loc_OSCFAR_u5 >= RefCellNum_u6)
            {
                Loc_OSCFAR_u5 = floor(double(RefCellNum_u6 / 2) + 1);
            }
            
            float Data_OSLoc_u30;
            Data_OSLoc_u30 = DataSet_RefCell_u30[RefCellNum_u6 - 1];
            uint8_t LeftLoc_u1 = 1;
            uint8_t LocComp_u5 = Loc_OSCFAR_u5 - 1;
            uint8_t middleLoc = floor(double(RefCellNum_u6 / 2));
            if (Loc_OSCFAR_u5 > middleLoc)
            {
                LeftLoc_u1 = 0;
                LocComp_u5 = RefCellNum_u6 - Loc_OSCFAR_u5;
            }
            for (uint8_t i = 0; i < RefCellNum_u6 - 1; i++)
            {
                float Cell_ToCompare_u30 = DataSet_RefCell_u30[i];
                uint8_t IsData_u1 = 0;
                uint8_t Count_u5 = 0;
                uint8_t Data_Equal_u5 = 0;
                uint8_t flag_Not_u1 = 0;
                for (uint8_t j = 0; j < RefCellNum_u6; j++)
                {
                    float Cell_Temp_u30 = DataSet_RefCell_u30[j];
                    if (LeftLoc_u1 == 1)
                    {
                        if (Cell_ToCompare_u30 > Cell_Temp_u30)
                        {
                            Count_u5 = Count_u5 + 1;
                        }
                        
                    }
                    else
                    {
                        if(Cell_ToCompare_u30 < Cell_Temp_u30)
                        {
                            Count_u5 = Count_u5 + 1;
                        }
                    }
                    if (Cell_ToCompare_u30 == Cell_Temp_u30)
                    {
                        Data_Equal_u5 = Data_Equal_u5 + 1;
                    }
                    if(Count_u5 > LocComp_u5)
                    {
                        flag_Not_u1 = 1;
                        break;
                    }
                }
                if (flag_Not_u1 == 0)
                {
                    if (Data_Equal_u5 == 1)
                    {
                        if (Count_u5 == LocComp_u5)
                        {
                            IsData_u1 = 1;
                        }
                        
                    }
                    else
                    {
                        uint8_t NumLess_u5 = Count_u5 + Data_Equal_u5;
                        if (NumLess_u5 > LocComp_u5)
                        {
                            IsData_u1 = 1;
                        }
                    }
                }
                if (IsData_u1 == 1)
                {
                    Data_OSLoc_u30 = Cell_ToCompare_u30;
                    break;
                }
            }
            uint8_t thIdx = floor(double(tix / (ChirpNum_u11/16)))                    
            float Threshold2_u39 = 8 * Data_OSLoc_u30;
            float CellToDetect_u32 = SpatialFFTVelSel_VelNum_RangeNum_device[tix + tiy * ChirpNum_u11] * 2 * 2;
            if (CellToDetect_u32 > Threshold2_u39)
            {
                IsTarget_1D_R_u1[tix + tiy * ChirpNum_u11] = 1;
                VSNR_u11[tix + tiy * ChirpNum_u11] = SpatialFFTVelSel_VelNum_RangeNum_device[tix + tiy * ChirpNum_u11] / Data_OSLoc_u30;
            } 
        }
    }
}

__global__ postProcess_kernel(uint16_t ChirpNum_u11, uint16_t RangeCellNum_u10, uint16_t Index_Chirp_NotMove_OSCFAR_u11, uint16_t Threshold_RangeDim_For_2D_OSCFAR_u9, uint8_t* IsTarget_1D_R_u1, uint8_t* IsTarget_1D_V_u1, float* SpatialFFTVelSel_VeloNum_RangeNum_device)
{
    int tix = blockDim.x * blockIdx.x + threadIdx.x;
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tix < ChirpNum_u11 && tiy < RangeCellNum_u10)
    {
        /* code */
    }
}

void_winR_process(float* WinRout_RangeSampleNum_ChirpNum_RxNum_device, int16_t* radarInputdata_device, float* win_Coef_device, uint8_t& fft_win_type, uint16_t& RangeSampleNum, uint16_t& ChirpNum, uint16_t& RxNum)
{
    dim3 block_win(256, 1, 1);
    dim3 grid_win((RangeSampleNum + block_win.x - 1) / block_win.x,1,1);
    dim3 block_fftr(16,1,32);
    dim3 grid_fftr((RangeSampleNum + block_fftr.x - 1) / block_fftr.x, (RxNum + block_fftr.y - 1)/block_fftr.y, (ChirpNum + block_fftr.z - 1)/block_fftr.z);

    switch (fft_win_type)
    {
    case 0:
        rectWin_kernel<<<grid_win, block_win>>>(win_Coef_device, RangeSampleNum);
        break;
    case 1:
        hanningWin_kernel<<<grid_win, block_win>>>(win_Coef_device, RangeSampleNum);
        break;
    case 2:
        hammingWin_kernel<<<grid_win, block_win>>>(win_Coef_device, RangeSampleNum);
        break;
    default:
        break;
    }

    addWin_kernel<<<block_fftr, grid_fftr>>>(WinRout_RangeSampleNum_ChirpNum_RxNum_device,radarInputdata_device,win_Coef_device,RangeSampleNum,ChirpNum,RxNum);
}

void func_fftR_process(float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device, float* WinRout_RangeSampleNum_ChirpNum_RxNum_device, uint16_t& RangeSampleNum, uint16_t& VelocityNum, uint16_t& RxNum)
{
    cufftHandle plan;
    cufftPlan1d(&plan, RangeSampleNum, CUFFT_R2C, VelocityNum*RxNum)
    cufftExecR2C(plan, WinRout_RangeSampleNum_ChirpNum_RxNum_device, CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device);
    cufftDestroy(plan);
}

void func_dataReshape_afterFFTR(float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device, float2* CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device, uint16_t* all_tx_seq_pos, uint16_t& TxNum, uint16_t& RxNum, uint16_t& TxReuseNum, uint16_t& CoarseRangeNum)
{
    uint16_t velocityNum = TxNum * TxReuseNum;
    dim3 block(16,64,1);
    dim3 grid((CoarseRangeNum + block.x - 1) / block.x, (velocityNum + block.y - 1) / block.y, RxNum);
    uint16_t* all_tx_seq_pos_device;
    cudaMalloc(all_tx_seq_pos_device, velocityNum * sizeof(uint16_t));
    cudaMemcpy(all_tx_seq_pos_device, all_tx_seq_pos, velocityNum * sizeof(uint16_t), cudaMemcpyHostToDevice);
    dataReshape_afterFFTR_kernel<<<grid, block>>>(CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device,CoarseRangeFFT_ValidCoarseRangeBinNum_ChirpNum_RxNum_device,all_tx_seq_pos_device,TxNum,RxNum,TxReuseNum,CoarseRangeNum);
    cudaFree(all_tx_seq_pos_device);
}

void func_winD_process(float2* WinDout_ChirpNum_RangeSampleNum_MIMONum_device, float2* CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device, float* win_Coef_device, uint8_t& fft_win_type, uint16_t& VelocityNum, uint16_t& CoarseRangeNum, uint16_t& MimoNum)
{
    dim3 block_win(256,1,1);
    dim3 grid_win((VelocityNum + block_win.x - 1) / block_win.x,1,1);
    dim3 block_fftd(64,16,1);
    dim3 grid_fftd((VelocityNum + blockfftd.x - 1) / block_fftd.x, (CoarseRangeNum + blockfftd.y - 1)/blockfftd.y, MimoNum);

    switch (fft_win_type)
    {
    case 0:
        rectWin_kernel<<<grid_win,block_win>>>(win_Coef_device,VelocityNum);
        break;
    case 1:
        hanningWin_kernel<<<grid_win,block_win>>>(win_Coef_device,VelocityNum);
        break;
    case 2:
        hammingWin_kernel<<<grid_win,block_win>>>(win_Coef_device,VelocityNum);
        break;
    default:
        break;
    }

    addWinD_kernel<<<grid_fftd, block_fftd>>>(WinDout_ChirpNum_RangeSampleNum_MIMONum_device,CoarseRangeFFT_ChirpNum_RangeSampleNum_MIMONum_device,win_Coef_device,VelocityNum,CoarseRangeNum,MimoNum);
}

void func_fftD_process(float2* FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, float2* WinDout_ChirpNum_RangeSampleNum_MIMONum_device, uint16_t& VelocityNum, uint16_t& CoarseRangeNum, uint16_t& MIMONum)
{
    cufftHandle plan;
    int Batch = CoarseRangeNum * MIMONum;
    cufftPlan1d(&plan,VelocityNum,CUFFT_C2C,Batch);
    cufftExecC2C(plan, WinDout_ChirpNum_RangeSampleNum_MIMONum_device,FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device,CUFFT_FORWARD);
    cufftDestroy(plan);

    dim3 block(64, 16, 1);
    dim3 grid((VelocityNum/2 + block.x - 1) / block.x, (CoarseRangeNum + block.y - 1)/block.y, MIMONum);
    fftshift<<<grid,block>>>(FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, VelocityNum, CoarseRangeNum, MIMONum);
}

void func_Spatial_Reorder(float2* result_xNum_yNum_VeloFFTNum_RangeNum, float2* FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device, uint8_t& Array_option, uint16_t& MimoNum, uint16_t* pos_in_mat, uint16_t& VirtArrHorGridLen, uint16_t& VirtArrVertGridLen, uint16_t& VelocityNum,uint16_t& CoarseRangeNum)
{
    uint16_t* pos_in_mat_device;
    cudaMalloc(&pos_in_mat_device, MimoNum * 2 * sizeof(uint16_t));
    cudaMemcpy(pos_in_mat_device, pos_in_mat, MimoNum * 2 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    switch (Array_option)
    {
    case 0:
        ;
        break;
    case 1:
        ;
        break;
    case 2:
        ;
        break;
    case 3:
        dim3 block(64,16,1);
        dim3 grid((VelocityNum/2 + block.x - 1) / block.x, (CoarseRangeNum + block.y - 1)/block.y, MimoNum);
        spatial_reorder_kernel<<<grid,block>>>(result_xNum_yNum_VeloFFTNum_RangeNum,FFT2D_VeloFFTNum_CoarseRangeBinNum_MIMONum_device,pos_in_mat_device,MimoNum,VelocityNum,CoarseRangeNum,VirtArrHorGridLen,VirtArrVertGridLen);
        break;
    default:
        break;
    }

    cudaFree(pos_in_mat_device);
}


void func_winA_process(float2* winAout_xNum_yNum_VeloFFTNum_RangeNum_device, float2* result_xNum_yNum_VeloFFTNum_RangeNum_device, float* win_Coef_device, uint8_t& fft_win_type, uint16_t& VirtArrHorGridLen, uint16_t VirtArrVertGridLen, uint16_t& VelocityNum, uint16_t& CoarseRangeNum, uint16_t& AngleHorNum)
{
    dim3 block_win(27,1,1);
    dim3 grid_win((VirtArrHorGridLen + block_win.x - 1)/ block_win.x, 1,1);
    dim3 block_ffta(1,64,16);
    dim3 grid_ffta((VirtArrHorGridLen + block_ffta.x - 1)/block_ffta.x, (VelocityNum + block_ffta.y - 1)/block_ffta.y,(CoarseRangeNum+block_ffta.z-1)/block_ffta.z);
    switch (fft_win_type)
    {
    case 0:
        rectWin_kernel<<<grid_win,block_win>>>(win_Coef_device, VirtArrHorGridLen);
        break;
    case 1:
        hanningWin_kernel<<<grid_win,block_win>>>(win_Coef_device, VirtArrHorGridLen);
        break;
    case 2:
        hammingWin_kernel<<<grid_win,block_win>>>(win_Coef_device, VirtArrHorGridLen);
        break;
    default:
        break;
    }

    addWinA_kernel<<<grid_ffta,block_ffta>>>(winAout_xNum_yNum_VeloFFTNum_RangeNum_device,result_xNum_yNum_VeloFFTNum_RangeNum_device,win_Coef_device,VirtArrHorGridLen,VirtArrVertGridLen,VelocityNum,CoarseRangeNum,AngleHorNum);
}

void func_fftA_process(float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, float2* winAout_xNum_yNum_VeloFFTNum_RangeNum_device, uint16_t& AngleHorNum, uint16_t& VirtArrVertGridLen, uint16_t& VelocityNum, uint16_t& CoarseRangeNum)
{
    cufftHandle plan;
    int Batch = VirtArrVertGridLen * VelocityNum * CoarseRangeNum;
    cufftPlan1d(&plan, AngleHorNum, CUFFT_C2C, Batch);
    cufftExecC2C(plan, winAout_xNum_yNum_VeloFFTNum_RangeNum_device, SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, CUFFT_FORWARD);
    cufftDestroy(plan);

    dim3 block(1, 64, 16);
    dim3 grid((AngleHorNum + block.x - 1)/block.x, (VelocityNum + block.y - 1)/block.y, (CoarseRangeNum + block.z - 1)/block.z);
    fftshift_ffta<<<grid,block>>>(SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device,AngleHorNum,VirtArrVertGridLen,VelocityNum,CoarseRangeNum);
}

void func_abs_process(float* SpatialFFTA_ABS_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, float2* SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint16_t& AngleHorNum, uint16_t& VirtArrVertGridLen, uint16_t& VelocityNum, uint16_t CoarseRangeNum)
{
    dim3 block(1,64,16);
    dim3 grid((AngleHorNum + block.x - 1)/block.x, (VelocityNum + block.y - 1)/block.y, (CoarseRangeNum + block.z - 1)/block.z);
    ffta_abs_kernel<<<grid,block>>>(SpatialFFTA_ABS_AngleHorNum_yNum_VeloFFTNum_RangeNum_device,SpatialFFTA_AngleHorNum_yNum_VeloFFTNum_RangeNum_device,AngleHorNum,VirtArrVertGridLen,VelocityNum,CoarseRangeNum);
}

void func_cfar3d_cal_across_ArbitaryDim(float* SpatialFFTVelSel_VeloNum_RangeNum_device, float* SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device, uint8_t& SqueezeDim, uint8_t& cfar_include_order, uint8_t& cfar_exclude_order, float& snr_dB_different_dim, uint8_t& Switch3DMode, uint16_t& AngleHorNum, uint16_t& VelocityNum, uint16_t& CoarseRangeNum)
{
    dim3 block(128,4,2);
    dim3 grid((AngleHorNum + block.x - 1)/block.x, (VelocityNum + block.y - 1)/block.y, (CoarseRangeNum + block.z - 1)/block.z);
    cfar3d_cal_across_ArbitaryDim_kernel<<<grid,block>>>(SpatialFFTVelSel_VeloNum_RangeNum_device,SpatialFFTA_ABS_Mean_AngleHorNum_yNum_VeloFFTNum_RangeNum_device,AngleHorNum,VelocityNum,CoarseRangeNum);
}

void func_PeakSearch_And_CFAR_2D_Cross(uint16_t& TarNum_Detected, uint16_t* peak_R, uint16_t* peak_V, float* peak_Val, float* peak_SNR, DetPara& det_para, float* SpatialFFTVelSel_VeloNum_RangeNum_device)
{
    uint8_t *isPeak;
    cudaMalloc(&isPeak, det_para.ChirpNum_u11 * det_para.RangeCellNum_u10 * sizeof(uint8_t));
    dim3 block(64,16);
    dim3 grid((det_para.ChirpNum_u11 + block.x - 1)/block.x, (det_para.RangeCellNum_u10 + block.y - 1)/block.y);
    peak_search_kernel<<<grid,block>>>(isPeak,SpatialFFTVelSel_VeloNum_RangeNum_device,det_para.DetectCell_RIndex_Min_u10,det_para.DetectCell_RIndex_Max_u10, det_para.DetectCell_VIndex_Min_u11, det_para.DetectCell_VIndex_Max_u11,det_para.ChirpNum_u11,det_para.RangeCellNum_u10);
    cudaDeviceSynchronize();

    dim3 blockd(16,4);
    dim3 gridd((det_para.ChirpNum_u11 + blockd.x - 1)/blockd.x, (det_para.RangeCellNum_u10 + blockd.y - 1)/blockd.y);
    uint8_t* IsTarget_1D_R_u1;
    float* RSNR_u11;
    cudaMalloc(&IsTarget_1D_R_u1, det_para.ChirpNum_u11 * det_para.RangeCellNum_u10 * sizeof(uint8_t));
    cudaMalloc(&RSNR_u11, det_para.ChirpNum_u11 * det_para.RangeCellNum_u10 * sizeof(float));
    CFARChM_OS_1D_kernel<<<gridd,blockd>>>(IsTarget_1D_R_u1,RSNR_u11,det_para.RangeCellNum_u10,det_para.ChirpNum_u11,det_para.cfar_para.ProCellNum_R_u2,det_para.cfar_para.RefCellNum_1D_u5,det_para.Lod_OSCFAR_u5,isPeak,SpatialFFTVelSel_VeloNum_RangeNum_device)
    cudaDeviceSynchronize();

    uint8_t* IsTarget_1D_V_u1;
    float* VSNR_u11;
    cudaMalloc(&IsTarget_1D_V_u1, det_para.ChirpNum_u11 * det_para.RangeCellNum_u10 * sizeof(uint8_t));
    cudaMalloc(&VSNR_u11, det_para.ChirpNum_u11 * det_para.RangeCellNum_u10 * sizeof(float));
    CFARChM_OS_1D_V_kernel<<<gridd,blockd>>>(IsTarget_1D_V_u1,VSNR_u11,det_para.LogicTestFlag_u1,det_para.RangeCellNum,det_para.ChirpNum_u11,det_para.cfar_para.ProCellNum_V_u2,det_para.cfar_para.RefCellNum_1D_u5,det_para.Loc_OSCFAR_u5,IsTarget_1D_R_u1, SpatialFFTVelSel_VeloNum_RangeNum_device);

    postProcess_kernel<<<grid,block>>>(det_para.ChirpNum_u11,det_para.RangeCellNum_u10,det_para.Index_Chirp_NotMove_OSCFAR_u11, det_para.Threshold_RangeDim_For_2D_OSCFAR_u9,IsTarget_1D_R_u1,IsTarget_1D_V_u1,SpatialFFTVelSel_VeloNum_RangeNum_device);
    cudaDeviceSynchronize();

    // 将结果memcpy到主机端，赋给输出参数

    // free掉内存
}