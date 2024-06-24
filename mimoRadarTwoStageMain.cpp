#include <iostream>
#include <stdexcept>
#include "parameters.hpp"
#include "coarseFrameSignalProcess.hpp"
#include "cuda_runtime_api.h"

auto TOI_CFAR = std::make_unique<float[]>(15000);
auto TOI_ConRegion = std::make_unique<float[]>(15000);
uint16_t TOI_Count = 0;

int main(){
    std::string filename = "ADCData_Frame_000000_20240314_17_22_19_2876_ShareMemory.bin";
    
    BinFile bin_file;
    VirtualArray virtual_array;

    get_info(filename, bin_file, virtual_array);
    cudaSetDevice(3);

    if (bin_file.para_sys.frame_type == 0)
    {
        for (size_t i = 0; i < 5; i++)
        {
            func_signal_process_coarse(TOI_CFAR, TOI_ConRegion, TOI_Count, filename, bin_file.para_sys, virtual_array, bin_file.input_data, bin_file.compensate_mat);
        }
        for (size_t i = 0; i < 10; i++)
        {
            __TOI__(PCOARSE);
            func_signal_process_coarse(TOI_CFAR, TOI_ConRegion, TOI_Count, filename, bin_file.para_sys, virtual_array, bin_file.input_data, bin_file.compensate_mat);
            __TOC__(PCOARSE);
        }
    }
    else if(bin_file.para_sys.frame_type == 1)
    {
        if(TOI_Count > 0)
        {
            func_signal_process_fine();
        }
        else
        {
            printf("s%\n", "TOI is empty");
        }
    }
    else
    {
        throw std::invalid_argument("Input value is illegal.");
    }
    return 0;
}