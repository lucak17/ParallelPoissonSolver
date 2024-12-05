#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdio>




template<int DIM, typename T_data, bool fieldData> 
struct ResetNeumanBCsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, const ExactSolutionAndBCs<DIM,T_data> exactSolutionAndBCs, const Idx dir, const T_data normFieldB, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge, const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsData, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, const alpaka::Vec<alpaka::DimInt<3>, T_data> origin, 
                                const alpaka::Vec<alpaka::DimInt<3>, T_data> globalLocation, const alpaka::Vec<alpaka::DimInt<3>, Idx> nlocal_noguards,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const indxData = gridThreadIdxShifted + adjustIdx;
        auto const indxGuard = gridThreadIdxShifted + adjustIdx;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];

        const T_data x=origin[2] + (iGrid-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
        const T_data y=origin[1] + (jGrid-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
        const T_data z=origin[0] + (kGrid-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];


        if( iGrid>=indexLimitsEdge[4] && iGrid<indexLimitsEdge[5] && jGrid>=indexLimitsEdge[2] && jGrid<indexLimitsEdge[3] && kGrid>=indexLimitsEdge[0] && kGrid<indexLimitsEdge[1])
        {
            if constexpr (fieldData)
            {
                bufData(indxGuard[0],indxGuard[1],indxGuard[2]) = bufData(indxData[0],indxData[1],indxData[2]) - 2 * ds[dir] * exactSolutionAndBCs.trueSolutionDdir(x,y,z,DIM - 1 - dir) / normFieldB;
            }
            else
            {
                bufData(indxGuard[0],indxGuard[1],indxGuard[2]) = bufData(indxData[0],indxData[1],indxData[2]);
            }
        }

    }
};


