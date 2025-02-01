#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdio>


template<int DIM, typename T_data_in, typename T_data_out> 
struct CastPrecisionFieldKernel
{
    template<typename TAcc, typename TMdSpanIn,typename TMdSpanOut>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpanIn bufDataA, TMdSpanOut bufDataB) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        // Z Y X
        bufDataB(gridThreadIdx[0],gridThreadIdx[1],gridThreadIdx[2]) =  static_cast<T_data_out>(bufDataA(gridThreadIdx[0],gridThreadIdx[1],gridThreadIdx[2]));
    }
};




template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct AssignFieldWith1FieldKernel
{
    template<typename TAcc, typename TMdSpanBase,typename TMdSpanChebyshev>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpanBase bufDataA, TMdSpanChebyshev bufDataB, const T_data_chebyshev constB,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        /*
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        */
        // Z Y X
        if( gridThreadIdxShifted[2]>=indexLimitsSolver[4] && gridThreadIdxShifted[2]<indexLimitsSolver[5] && gridThreadIdxShifted[1]>=indexLimitsSolver[2] && gridThreadIdxShifted[1]<indexLimitsSolver[3] && gridThreadIdxShifted[0]>=indexLimitsSolver[0] && gridThreadIdxShifted[0]<indexLimitsSolver[1] )
        {
            bufDataA(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) =  constB * bufDataB(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]);
        }
    }
};



template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct AssignFieldWith1FieldKernelSolver
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, const T_data_chebyshev constB,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, Idx> offset) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        // Z Y X
        bufDataA(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) =  constB * bufDataB(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]);
    }
};



template<int DIM, typename T_data> 
struct AssignFieldWith2FieldsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, TMdSpan bufDataC, const T_data constB, const T_data constC,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[4] && i<indexLimitsSolver[5] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[0] && k<indexLimitsSolver[1] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i) + constC * bufDataC(k,j,i);
        }
    }
};


template<int DIM, typename T_data> 
struct AssignFieldWith4FieldsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, TMdSpan bufDataC, TMdSpan bufDataD, TMdSpan bufDataE, 
                                    const T_data constB, const T_data constC, const T_data constD, const T_data constE,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[4] && i<indexLimitsSolver[5] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[0] && k<indexLimitsSolver[1] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i) + constC * bufDataC(k,j,i) + constD * bufDataD(k,j,i) + constE * bufDataE(k,j,i);
        }
    }
};









template<int DIM, typename T_data_base, typename T_data_chebyshev > 
struct Chebyshev1Kernel
{

    // apply s0 and s1
    /*
    for(k=kmin; k<kmax; k++)
    {
        for(j=jmin; j<jmax; j++)
        {
            for(i=imin; i<imax; i++)
            {
                indx=i + this->stride_j_*j + this->stride_k_*k;
                fieldZ[indx] = fieldB[indx]/theta_;
                fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
            }
        }
    }
    */

    template<typename TAcc, typename TMdSpanBase, typename TMdSpanChebyshev>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpanBase bufDataB, TMdSpanChebyshev bufDataY, TMdSpanChebyshev bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev theta, const T_data_chebyshev rhoCurr, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const i = gridThreadIdxShifted[2];
        auto const j = gridThreadIdxShifted[1];
        auto const k = gridThreadIdxShifted[0];
        
        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = 2 * rhoCurr / delta * (  1/r0 /theta );
        const T_data_chebyshev f1 = 2 * rhoCurr / delta * (  1/r1 /theta );
        const T_data_chebyshev f2 = 2 * rhoCurr / delta * (  1/r2 /theta );
        T_data_chebyshev tmp;
                
        if( i>=indexLimitsSolver[4] && i<indexLimitsSolver[5] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[0] && k<indexLimitsSolver[1] )
        {
            bufDataZ(k,j,i) = bufDataB(k,j,i)/theta;
        
            if constexpr (DIM==3)
            {
                const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1 + 1/r2)/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                tmp = bufDataB(k,j,i) * fc0 + ( bufDataB(k,j,i-1) + bufDataB(k,j,i + 1) ) * f0;
                tmp +=              ( bufDataB(k,j-1,i) + bufDataB(k,j+1,i)) * f1 +
                                    ( bufDataB(k-1,j,i) + bufDataB(k+1,j,i) ) * f2;  
             }
            else if constexpr (DIM==2)
            {
                const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1)/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                tmp = bufDataB(k,j,i) * fc0 + ( bufDataB(k,j,i-1) + bufDataB(k,j,i + 1) ) * f0;
                tmp +=              ( bufDataB(k,j-1,i) + bufDataB(k,j+1,i)) * f1;
            }
            else
            {
                const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 )/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                tmp = bufDataB(k,j,i) * fc0 + ( bufDataB(k,j,i-1) + bufDataB(k,j,i + 1) ) * f0;
            }
            bufDataY(k,j,i) = tmp;

        }   
    }
};






template<int DIM, typename T_data_base,typename T_data_chebyshev> 
struct Chebyshev2Kernel
{

    // apply si
    /*
    for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
    {   
        rhoOld=rhoCurr;
        rhoCurr=1/(2*sigma_ - rhoOld);
        if constexpr (communicationON)
        {
            this->communicatorMPI_(fieldY);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
        this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                }
            }
        }
        std::swap(fieldZ,fieldY);
        std::swap(fieldW,fieldY);
    }
    */
    template<typename TAcc, typename TMdSpanBase, typename TMdSpanChebyshev>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpanBase bufDataB, TMdSpanChebyshev bufDataY, TMdSpanChebyshev bufDataW, TMdSpanChebyshev bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev sigma, const T_data_chebyshev rhoCurr, const T_data_chebyshev rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> offset ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const i = gridThreadIdxShifted[2];
        auto const j = gridThreadIdxShifted[1];
        auto const k = gridThreadIdxShifted[0];
        
        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data_chebyshev f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data_chebyshev f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data_chebyshev fB = rhoCurr * 2 / delta;
        const T_data_chebyshev fZ = - rhoCurr * rhoOld;
        T_data_chebyshev tmp;

        //if( i>=indexLimitsSolver[4] && i<indexLimitsSolver[5] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[0] && k<indexLimitsSolver[1] )
        //if(i<indexLimitsSolver[5] && j<indexLimitsSolver[3] && k<indexLimitsSolver[1] )
        {            

            if constexpr (DIM==3)
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                tmp = bufDataY(k,j,i) * fc0 + ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0;
                tmp += ( bufDataY(k,j-1,i) + bufDataY(k,j+1,i)) * f1 + ( bufDataY(k-1,j,i) + bufDataY(k+1,j,i) ) * f2;  
             }
            else if constexpr (DIM==2)
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                tmp = bufDataY(k,j,i) * fc0 + ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0;
                tmp += ( bufDataY(k,j-1,i) + bufDataY(k,j+1,i)) * f1; 
            }
            else
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                tmp = bufDataY(k,j,i) * fc0 + ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0;
            }
            tmp += bufDataB(k,j,i) * fB;
            tmp += bufDataZ(k,j,i) * fZ;
            bufDataW(k,j,i) = tmp;
        }   
    }
};









template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct Chebyshev1KernelSharedMem
{

    // apply s0 and s1
    /*
    for(k=kmin; k<kmax; k++)
    {
        for(j=jmin; j<jmax; j++)
        {
            for(i=imin; i<imax; i++)
            {
                indx=i + this->stride_j_*j + this->stride_k_*k;
                fieldZ[indx] = fieldB[indx]/theta_;
                fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
            }
        }
    }
    */

    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev theta, const T_data_chebyshev rhoCurr, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        /*
        auto const iGrid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const jGrid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const kGrid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        */
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        /*
        auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        */

        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = 2 * rhoCurr / delta * (  1/r0 /theta );
        const T_data_chebyshev f1 = 2 * rhoCurr / delta * (  1/r1 /theta );
        const T_data_chebyshev f2 = 2 * rhoCurr / delta * (  1/r2 /theta );
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        //auto& sdata = alpaka::declareSharedVar<T_data[4096], __COUNTER__>(acc);
        auto* sdata = alpaka::getDynSharedMem<T_data_chebyshev>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data_chebyshev, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

        //auto globalIdx = gridThreadIdx;
            
        // fill shared memory
        //if( iGrid>0 && iGrid<(gridThreadExtent[2]-1) && jGrid>0 && jGrid<(gridThreadExtent[1]-1) && kGrid>0 && kGrid<(gridThreadExtent[0]-1) )
        {
            for(auto k = blockThreadIdx[0]; k < chunck[0]; k += blockThreadExtent[0])
            {
                for(auto j = blockThreadIdx[1]; j < chunck[1]; j += blockThreadExtent[1])
                {
                    for(auto i = blockThreadIdx[2]; i < chunck[2]; i += blockThreadExtent[2])
                    {
                        auto localIdx3D = alpaka::Vec<alpaka::DimInt<3>,Idx>(k,j,i);
                        auto globalIdx = localIdx3D + blockStartThreadIdx;
                        //if( globalIdx[2]>=0 && globalIdx[2]<(gridThreadExtent[2]) && globalIdx[1]>=0 && globalIdx[1]<(gridThreadExtent[1]) && globalIdx[0]>=0 && globalIdx[0]<(gridThreadExtent[0]) )
                        {
                            sdataMdSpan(localIdx3D[0], localIdx3D[1],localIdx3D[2]) = bufDataB(globalIdx[0], globalIdx[1],globalIdx[2]);
                        }
                    }
                }
            }   
            alpaka::syncBlockThreads(acc);
        
            if( iGrid>=indexLimitsSolver[4] && iGrid<indexLimitsSolver[5] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[0] && kGrid<indexLimitsSolver[5] )
            {
                
                //localIdx = blockThreadIdx + haloSize;

                // offset for halo, as we only want to go over core cells
                //auto localIdx = blockThreadIdx + haloSize;
                const auto localIdx = blockThreadIdx + haloSize;
                bufDataZ(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2])/theta;
                if constexpr (DIM==3)
                {
                    const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1 + 1/r2)/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                                        ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                                        ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2;  
                }
                else if constexpr (DIM==2)
                {
                    const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1)/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1; 
                }
                else
                {
                    const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 )/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0; 
                }
            } 
        }  
    }
};



template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct Chebyshev1KernelSharedMemSolver
{

    // apply s0 and s1
    /*
    for(k=kmin; k<kmax; k++)
    {
        for(j=jmin; j<jmax; j++)
        {
            for(i=imin; i<imax; i++)
            {
                indx=i + this->stride_j_*j + this->stride_k_*k;
                fieldZ[indx] = fieldB[indx]/theta_;
                fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
            }
        }
    }
    */

    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev theta, const T_data_chebyshev rhoCurr, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize, const alpaka::Vec<alpaka::DimInt<3>, Idx> offset ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        /*
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        */
        /*
        auto const iGrid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const jGrid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const kGrid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        */
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        /*
        auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        */

        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = 2 * rhoCurr / delta * (  1/r0 /theta );
        const T_data_chebyshev f1 = 2 * rhoCurr / delta * (  1/r1 /theta );
        const T_data_chebyshev f2 = 2 * rhoCurr / delta * (  1/r2 /theta );
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        //auto& sdata = alpaka::declareSharedVar<T_data_chebyshev[4096], __COUNTER__>(acc);
        auto* sdata = alpaka::getDynSharedMem<T_data_chebyshev>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data_chebyshev, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

        //auto globalIdx = gridThreadIdx;
            
        // fill shared memory
        //if( iGrid>0 && iGrid<(gridThreadExtent[2]-1) && jGrid>0 && jGrid<(gridThreadExtent[1]-1) && kGrid>0 && kGrid<(gridThreadExtent[0]-1) )
        {
            for(auto k = blockThreadIdx[0]; k < chunck[0]; k += blockThreadExtent[0])
            {
                for(auto j = blockThreadIdx[1]; j < chunck[1]; j += blockThreadExtent[1])
                {
                    for(auto i = blockThreadIdx[2]; i < chunck[2]; i += blockThreadExtent[2])
                    {
                        auto localIdx3D = alpaka::Vec<alpaka::DimInt<3>,Idx>(k,j,i);
                        auto globalIdx = localIdx3D + blockStartThreadIdx + offset - haloSize;
                        //if( globalIdx[2]>=0 && globalIdx[2]<(gridThreadExtent[2]) && globalIdx[1]>=0 && globalIdx[1]<(gridThreadExtent[1]) && globalIdx[0]>=0 && globalIdx[0]<(gridThreadExtent[0]) )
                        {
                            sdataMdSpan(localIdx3D[0], localIdx3D[1],localIdx3D[2]) = bufDataB(globalIdx[0], globalIdx[1],globalIdx[2]);
                        }
                    }
                }
            }   
            alpaka::syncBlockThreads(acc);
        
            //if( iGrid>=indexLimitsSolver[4] && iGrid<indexLimitsSolver[5] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[0] && kGrid<indexLimitsSolver[5] )
            {
                
                //localIdx = blockThreadIdx + haloSize;

                // offset for halo, as we only want to go over core cells
                //auto localIdx = blockThreadIdx + haloSize;
                auto const gridThreadIdxShifted = gridThreadIdx + offset;
                const auto localIdx = blockThreadIdx + haloSize;
                
                bufDataZ(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2])/theta;

                if constexpr (DIM==3)
                {
                    const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1 + 1/r2)/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                                        ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                                        ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2;  
                }
                else if constexpr (DIM==2)
                {
                    const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1)/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1; 
                }
                else
                {
                    const T_data_chebyshev fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 )/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0; 
                }
            } 
        }  
    }
};







template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct Chebyshev2KernelSharedMem
{

    // apply si
    /*
    for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
    {   
        rhoOld=rhoCurr;
        rhoCurr=1/(2*sigma_ - rhoOld);
        if constexpr (communicationON)
        {
            this->communicatorMPI_(fieldY);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
        this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                }
            }
        }
        std::swap(fieldZ,fieldY);
        std::swap(fieldW,fieldY);
    }
    */
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataW, TMdSpan bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev sigma, const T_data_chebyshev rhoCurr, const T_data_chebyshev rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        //auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        //auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        //auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data_chebyshev f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data_chebyshev f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data_chebyshev fB = rhoCurr * 2 / delta;
        const T_data_chebyshev fZ = - rhoCurr * rhoOld;
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        auto* sdata = alpaka::getDynSharedMem<T_data_chebyshev>(acc);
        //auto& sdata = alpaka::declareSharedVar<T_data_chebyshev[sMemSizeFixed], __COUNTER__>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data_chebyshev, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

        for(auto k = blockThreadIdx[0]; k < chunck[0]; k += blockThreadExtent[0])
        {
            for(auto j = blockThreadIdx[1]; j < chunck[1]; j += blockThreadExtent[1])
            {
                for(auto i = blockThreadIdx[2]; i < chunck[2]; i += blockThreadExtent[2])
                {
                    auto localIdx3D = alpaka::Vec<alpaka::DimInt<3>,Idx>(k,j,i);
                    auto globalIdx = localIdx3D + blockStartThreadIdx;
                    //if( globalIdx[2]>=0 && globalIdx[2]<(gridThreadExtent[2]) && globalIdx[1]>=0 && globalIdx[1]<(gridThreadExtent[1]) && globalIdx[0]>=0 && globalIdx[0]<(gridThreadExtent[0]) )
                    {
                        sdataMdSpan(localIdx3D[0], localIdx3D[1],localIdx3D[2]) = bufDataY(globalIdx[0], globalIdx[1],globalIdx[2]);
                    }
                }
            }
        }   
        alpaka::syncBlockThreads(acc);
        
    
        if( iGrid>=indexLimitsSolver[4] && iGrid<indexLimitsSolver[5] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[0] && kGrid<indexLimitsSolver[1] )
        {
            
            //localIdx = blockThreadIdx + haloSize;

            // offset for halo, as we only want to go over core cells
            //auto localIdx = blockThreadIdx + haloSize;
            auto const globalIdx = gridThreadIdx + haloSize;
            auto const localIdx = blockThreadIdx + haloSize; 
            if constexpr (DIM==3)
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0
                                                                + ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0; 
                //bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0; 
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1;
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2; 
                //bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                //                ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                //                ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                //                ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2 ;
            }
            else if constexpr (DIM==2)
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1;
                                    //bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
            else
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0;
                                    //bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
            bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ;
        }  
    }
};





template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct Chebyshev2KernelSharedMemLoop1D
{

    // apply si
    /*
    for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
    {   
        rhoOld=rhoCurr;
        rhoCurr=1/(2*sigma_ - rhoOld);
        if constexpr (communicationON)
        {
            this->communicatorMPI_(fieldY);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
        this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                }
            }
        }
        std::swap(fieldZ,fieldY);
        std::swap(fieldW,fieldY);
    }
    */
    template<typename TAcc, typename TMdSpanBase, typename TMdSpanChebyshev>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpanBase bufDataB, TMdSpanChebyshev bufDataY, TMdSpanChebyshev bufDataW, TMdSpanChebyshev bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev sigma, const T_data_chebyshev rhoCurr, const T_data_chebyshev rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize, const alpaka::Vec<alpaka::DimInt<3>, Idx> offset ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data_chebyshev f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data_chebyshev f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data_chebyshev fB = rhoCurr * 2 / delta;
        const T_data_chebyshev fZ = - rhoCurr * rhoOld;
        T_data_chebyshev fc0;
        T_data_chebyshev tmp;
        if constexpr (DIM ==3){
            fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
        }
        else if constexpr(DIM ==2){
            fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1) );
        }
        else{
            fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 ) );
        }  

        bufDataW(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = bufDataB(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) * fB 
                                                                                            + bufDataZ(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) * fZ;
        
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const numThreadsPerBlock = blockThreadExtent.prod();
        auto const blockThreadIdx1D = alpaka::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        auto const chunkDim = chunck.prod();
        auto* sdata = alpaka::getDynSharedMem<T_data_chebyshev>(acc);
        //auto& sdata = alpaka::declareSharedVar<T_data_chebyshev[sMemSizeFixed], __COUNTER__>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data_chebyshev, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);
        
        for(auto i = blockThreadIdx1D; i < chunkDim; i += numThreadsPerBlock)
        {
            auto globalIdx = alpaka::mapIdx<3u>(alpaka::Vec<alpaka::DimInt<1u>, Idx>(i), chunck) + blockStartThreadIdx + offset - haloSize;
            //if( globalIdx[2]>=0 && globalIdx[2]<(gridThreadExtent[2]) && globalIdx[1]>=0 && globalIdx[1]<(gridThreadExtent[1]) && globalIdx[0]>=0 && globalIdx[0]<(gridThreadExtent[0]) )
            sdata[i] =  bufDataY(globalIdx[0], globalIdx[1],globalIdx[2]);
        }
        alpaka::syncBlockThreads(acc);
        
        constexpr Idx unit=1u;
                
        auto localIdx = haloSize;
        auto globalIdx =  blockStartThreadIdx + offset;
        const auto imin = jumpI*blockThreadIdx[2];
        const auto imax = jumpI*(blockThreadIdx[2]+1);
        const auto jmin = jumpJ*blockThreadIdx[1];
        const auto jmax = jumpJ*(blockThreadIdx[1]+1);
        const auto kmin = jumpK*blockThreadIdx[0];
        const auto kmax = jumpK*(blockThreadIdx[0]+1);
        //if(jumpK*blockThreadIdx[0]<blockThreadExtent[0] && jumpJ*blockThreadIdx[1]<blockThreadExtent[1] && jumpI*blockThreadIdx[2]<blockThreadExtent[2] )
        for(auto k =  kmin; k < kmax && k < blockThreadExtent[0]; k=k+unit)
        {
            for(auto j = jmin; j < jmax && j < blockThreadExtent[1]; j=j+unit)
            {
                for(auto i = imin ; i < imax && i < blockThreadExtent[2]; i=i+unit)
                {   

                    /*
                    printf("Cheb2 Thread %ld %ld %ld --> zyx %ld, %ld, %ld - localIdx %ld, %ld, %ld - global %ld, %ld, %ld - bufY %f - sdata %f\n", 
                                                blockThreadIdx[0],blockThreadIdx[1],blockThreadIdx[2], 
                                                k, j, i,
                                                localIdx[0],localIdx[1],localIdx[2],
                                                globalIdx[0],globalIdx[1],globalIdx[2], bufDataY(globalIdx[0],globalIdx[1],globalIdx[2]), sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) );
                    */
                    
                    // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                    localIdx = alpaka::Vec<alpaka::DimInt<3>,Idx>(k,j,i) + haloSize;
                    globalIdx = alpaka::Vec<alpaka::DimInt<3>,Idx>(k,j,i) + blockStartThreadIdx + offset;
                    if constexpr (DIM==3)
                    {
                        const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                        // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                        bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2 ;
                    }
                    else if constexpr (DIM==2)
                    {
                        const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                        // Z Y X
                        bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1;
                    }
                    else
                    {
                        const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                        // Z Y X
                        bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0;
                    }

                    
                }
            }
        } 
    }
};




template<int DIM, typename T_data_base, typename T_data_chebyshev> 
struct Chebyshev2KernelSharedMemSolver
{

    // apply si
    /*
    for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
    {   
        rhoOld=rhoCurr;
        rhoCurr=1/(2*sigma_ - rhoOld);
        if constexpr (communicationON)
        {
            this->communicatorMPI_(fieldY);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
        this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                }
            }
        }
        std::swap(fieldZ,fieldY);
        std::swap(fieldW,fieldY);
    }
    */
    template<typename TAcc, typename TMdSpanBase, typename TMdSpanChebyshev>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpanBase bufDataB, TMdSpanChebyshev bufDataY, TMdSpanChebyshev bufDataW, TMdSpanChebyshev bufDataZ, const T_data_chebyshev delta, const T_data_chebyshev sigma, const T_data_chebyshev rhoCurr, const T_data_chebyshev rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize, const alpaka::Vec<alpaka::DimInt<3>, Idx> offset ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        /*
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        */
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        //auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        //auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        //auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        const T_data_chebyshev r0 = ds[2]*ds[2];
        const T_data_chebyshev r1 = ds[1]*ds[1];
        const T_data_chebyshev r2 = ds[0]*ds[0];
        const T_data_chebyshev f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data_chebyshev f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data_chebyshev f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data_chebyshev fB = rhoCurr * 2 / delta;
        const T_data_chebyshev fZ = - rhoCurr * rhoOld;
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        auto* sdata = alpaka::getDynSharedMem<T_data_chebyshev>(acc);
        //auto& sdata = alpaka::declareSharedVar<T_data_chebyshev[sMemSizeFixed], __COUNTER__>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data_chebyshev, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);
        //auto* sdata2 = alpaka::getDynSharedMem<T_data_chebyshev>(acc);
        //auto& sdata = alpaka::declareSharedVar<T_data_chebyshev[sMemSizeFixed], __COUNTER__>(acc);
        //auto sdataMdSpan2 = std::experimental::mdspan<T_data_chebyshev, std::experimental::dextents<Idx, 3>>(sdata2, chunck[0], chunck[1], chunck[2]);

        for(auto k = blockThreadIdx[0]; k < chunck[0]; k += blockThreadExtent[0])
        {
            for(auto j = blockThreadIdx[1]; j < chunck[1]; j += blockThreadExtent[1])
            {
                for(auto i = blockThreadIdx[2]; i < chunck[2]; i += blockThreadExtent[2])
                {
                    auto localIdx3D = alpaka::Vec<alpaka::DimInt<3>,Idx>(k,j,i);
                    auto globalIdx = localIdx3D + blockStartThreadIdx + offset - haloSize;
                    //if( globalIdx[2]>=0 && globalIdx[2]<(gridThreadExtent[2]) && globalIdx[1]>=0 && globalIdx[1]<(gridThreadExtent[1]) && globalIdx[0]>=0 && globalIdx[0]<(gridThreadExtent[0]) )
                    {
                        sdataMdSpan(localIdx3D[0], localIdx3D[1],localIdx3D[2]) = bufDataY(globalIdx[0], globalIdx[1],globalIdx[2]);
                    }
                }
            }
        }

        alpaka::syncBlockThreads(acc);
        auto const globalIdx = gridThreadIdx + offset;
        auto const localIdx = blockThreadIdx + haloSize;
        //sdataMdSpan2(localIdx[0], localIdx[1],localIdx[2]) 
        bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ;
             
        //if( iGrid>=indexLimitsSolver[4] && iGrid<indexLimitsSolver[5] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[0] && kGrid<indexLimitsSolver[5] )
        {
            
            //localIdx = blockThreadIdx + haloSize;

            // offset for halo, as we only want to go over core cells
            //auto localIdx = blockThreadIdx + haloSize;
            if constexpr (DIM==3)
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) += sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2 ;
                                //bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ;  
            }
            else if constexpr (DIM==2)
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1;
                                    //bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
            else
            {
                const T_data_chebyshev fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 ;
                                    //bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
        }  
    }
};





// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev1KernelSharedMem<DIM,T_data_base,T_data_chebyshev>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data_base, typename T_data_chebyshev, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev1KernelSharedMem<DIM,T_data_base,T_data_chebyshev> const&  kernel ,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent, // dimensions of elements per thread
            TMdSpan bufDataB, 
            TMdSpan bufDataY, 
            TMdSpan bufDataZ, 
            const T_data_chebyshev delta, 
            const T_data_chebyshev theta, 
            const T_data_chebyshev rhoCurr,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize) // allocated filter width
        {
            // Reserve the buffer, buffers size is the number of elements in a block (tile)
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data_chebyshev);
        }
    };
} // namespace alpaka::trait


// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev2KernelSharedMem<DIM,T_data_base,T_data_chebyshev>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data_base, typename T_data_chebyshev, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev2KernelSharedMem<DIM,T_data_base,T_data_chebyshev> const&,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent,
            TMdSpan bufDataB, 
            TMdSpan bufDataY,
            TMdSpan bufDataW, 
            TMdSpan bufDataZ, 
            const T_data_chebyshev delta, 
            const T_data_chebyshev sigma, 
            const T_data_chebyshev rhoCurr,
            const T_data_chebyshev rhoOld,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize)  -> std::size_t
        {
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data_chebyshev);
        }
    };
} // namespace alpaka::trait


// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev2KernelSharedMemLoop1D<DIM,T_data_base,T_data_chebyshev>, TAcc>
    {
        template<typename TVec, typename TMdSpanBase, typename TMdSpanChebyshev, typename T_data_base, typename T_data_chebyshev, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev2KernelSharedMemLoop1D<DIM,T_data_base,T_data_chebyshev> const&,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent,
            TMdSpanBase bufDataB, 
            TMdSpanChebyshev bufDataY,
            TMdSpanChebyshev bufDataW, 
            TMdSpanChebyshev bufDataZ, 
            const T_data_chebyshev delta, 
            const T_data_chebyshev sigma, 
            const T_data_chebyshev rhoCurr,
            const T_data_chebyshev rhoOld,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize,
            const alpaka::Vec<alpaka::DimInt<3>, Idx> offset)  -> std::size_t
        {
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data);
        }
    };
} // namespace alpaka::trait



// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev1KernelSharedMemSolver<DIM,T_data_base,T_data_chebyshev>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data_base, typename T_data_chebyshev, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev1KernelSharedMemSolver<DIM,T_data_base,T_data_chebyshev> const&  kernel ,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent, // dimensions of elements per thread
            TMdSpan bufDataB, 
            TMdSpan bufDataY, 
            TMdSpan bufDataZ, 
            const T_data_chebyshev delta, 
            const T_data_chebyshev theta, 
            const T_data_chebyshev rhoCurr,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize,
            const alpaka::Vec<alpaka::DimInt<3>, Idx> offset) // allocated filter width
        {
            // Reserve the buffer, buffers size is the number of elements in a block (tile)
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data_chebyshev);
        }
    };
}



// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev2KernelSharedMemSolver<DIM,T_data_base,T_data_chebyshev>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data_base, typename T_data_chebyshev, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev2KernelSharedMemSolver<DIM,T_data_base,T_data_chebyshev> const&,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent,
            TMdSpan bufDataB, 
            TMdSpan bufDataY,
            TMdSpan bufDataW, 
            TMdSpan bufDataZ, 
            const T_data_chebyshev delta, 
            const T_data_chebyshev sigma, 
            const T_data_chebyshev rhoCurr,
            const T_data_chebyshev rhoOld,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data_base> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize,
            const alpaka::Vec<alpaka::DimInt<3>, Idx> offset)  -> std::size_t
        {
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data_chebyshev);
        }
    };
} // namespace alpaka::trait
