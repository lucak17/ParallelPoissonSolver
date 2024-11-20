#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once

#include "blockGrid.hpp"

template<int DIM, typename T_data>
class MatrixFreeOperatorA{

    public:

    MatrixFreeOperatorA(const BlockGrid<DIM,T_data>& blockGrid):
        blockGrid_(blockGrid),
        ds_(blockGrid.getDs()),
        stride_j_(blockGrid.getNlocalGuards()[0]),
        stride_k_(blockGrid.getNlocalGuards()[0]*blockGrid.getNlocalGuards()[1])
    {} 

    inline T_data operator()(const int i, const int j, const int k, const T_data data[]) const
    {
        if constexpr(DIM == 1)
        {
            return (data[i-1] - 2*data[i] + data[i+1])/(ds_[0]*ds_[0]);
        }
        else if constexpr(DIM == 2)
        {
            return (data[i-1 + stride_j_*j] - 2*data[i + stride_j_*j] + data[i+1 + stride_j_*j])/(ds_[0]*ds_[0])
                + (data[i + stride_j_*(j-1)] - 2*data[i + stride_j_*j] + data[i + stride_j_*(j+1)])/(ds_[1]*ds_[1]);  
        }
        else if constexpr(DIM == 3)
        {
            return (data[i-1 + stride_j_*j + stride_k_*k] - 2*data[i + stride_j_*j + stride_k_*k] + data[i+1 + stride_j_*j + stride_k_*k])/(ds_[0]*ds_[0]) 
                + (data[i + stride_j_*(j-1) + stride_k_*k] - 2*data[i + stride_j_*j + stride_k_*k] + data[i + stride_j_*(j+1) + stride_k_*k ])/(ds_[1]*ds_[1]) 
                + (data[i + stride_j_*j + stride_k_*(k-1)] - 2*data[i + stride_j_*j + stride_k_*k] + data[i + stride_j_*j + stride_k_*(k+1)])/(ds_[2]*ds_[2]);   
        }
    }


    private:

    const BlockGrid<DIM,T_data>& blockGrid_;
    const std::array<T_data,3> ds_;
    const int stride_j_;
    const int stride_k_;
};