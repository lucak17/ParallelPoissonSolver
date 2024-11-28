#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdio>

template<int DIM, typename T_data> 
struct AssignFieldWith1FieldKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, const T_data constB,
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
        if( gridThreadIdxShifted[2]>=indexLimitsSolver[0] && gridThreadIdxShifted[2]<indexLimitsSolver[1] && gridThreadIdxShifted[1]>=indexLimitsSolver[2] && gridThreadIdxShifted[1]<indexLimitsSolver[3] && gridThreadIdxShifted[0]>=indexLimitsSolver[4] && gridThreadIdxShifted[0]<indexLimitsSolver[5] )
        {
            bufDataA(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) =  constB * bufDataB(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]);
        }
    }
};



template<int DIM, typename T_data> 
struct AssignFieldWith1FieldKernelSolver
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, const T_data constB,
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
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
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
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i) + constC * bufDataC(k,j,i) + constD * bufDataD(k,j,i) + constE * bufDataE(k,j,i);
        }
    }
};









template<int DIM, typename T_data> 
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

    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataZ, const T_data delta, const T_data theta, const T_data rhoCurr, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        T_data r0 = ds[0]*ds[0];
        T_data r1 = ds[1]*ds[1];
        T_data r2 = ds[2]*ds[2];
        T_data fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1 + 1/r2)/theta );
        T_data f0 = 2 * rhoCurr / delta * (  1/r0 /theta );
        T_data f1 = 2 * rhoCurr / delta * (  1/r1 /theta );
        T_data f2 = 2 * rhoCurr / delta * (  1/r2 /theta );
                
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataZ(k,j,i) = bufDataB(k,j,i)/theta;
        
            if constexpr (DIM==3)
            {
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = bufDataB(k,j,i) * fc0 + 
                                    ( bufDataB(k,j,i-1) + bufDataB(k,j,i + 1) ) * f0 + 
                                    ( bufDataB(k,j-1,i) + bufDataB(k,j+1,i)) * f1 +
                                    ( bufDataB(k-1,j,i) + bufDataB(k+1,j,i) ) * f2;  
             }
            else if constexpr (DIM==2)
            {
                fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1)/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = bufDataB(k,j,i) * fc0 + 
                                    ( bufDataB(k,j,i-1) + bufDataB(k,j,i + 1) ) * f0 + 
                                    ( bufDataB(k,j-1,i) + bufDataB(k,j+1,i)) * f1;
            }
            else
            {
                fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 )/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = bufDataB(k,j,i) * fc0 + 
                                    ( bufDataB(k,j,i-1) + bufDataB(k,j,i + 1) ) * f0;
            }

        }   
    }
};






template<int DIM, typename T_data> 
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
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataW, TMdSpan bufDataZ, const T_data delta, const T_data sigma, const T_data rhoCurr, const T_data rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> offset ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const i = gridThreadIdxShifted[2];
        auto const j = gridThreadIdxShifted[1];
        auto const k = gridThreadIdxShifted[0];
        
        const T_data r0 = ds[0]*ds[0];
        const T_data r1 = ds[1]*ds[1];
        const T_data r2 = ds[2]*ds[2];
        const T_data f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data fB = rhoCurr * 2 / delta;
        const T_data fZ = - rhoCurr * rhoOld;

//        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {            

            if constexpr (DIM==3)
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(k,j,i) = bufDataY(k,j,i) * fc0 + 
                                    ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0 + 
                                    ( bufDataY(k,j-1,i) + bufDataY(k,j+1,i)) * f1 +
                                    ( bufDataY(k-1,j,i) + bufDataY(k+1,j,i) ) * f2 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ;  
             }
            else if constexpr (DIM==2)
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(k,j,i) = bufDataY(k,j,i) * fc0 + 
                                    ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0 + 
                                    ( bufDataY(k,j-1,i) + bufDataY(k,j+1,i)) * f1 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ; 
            }
            else
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                bufDataW(k,j,i) = bufDataY(k,j,i) * fc0 + 
                                    ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0 + 
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ;

            }
        }   
    }
};









template<int DIM, typename T_data> 
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
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataZ, const T_data delta, const T_data theta, const T_data rhoCurr, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
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

        const T_data r0 = ds[0]*ds[0];
        const T_data r1 = ds[1]*ds[1];
        const T_data r2 = ds[2]*ds[2];
        const T_data f0 = 2 * rhoCurr / delta * (  1/r0 /theta );
        const T_data f1 = 2 * rhoCurr / delta * (  1/r1 /theta );
        const T_data f2 = 2 * rhoCurr / delta * (  1/r2 /theta );
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        //auto& sdata = alpaka::declareSharedVar<T_data[4096], __COUNTER__>(acc);
        auto* sdata = alpaka::getDynSharedMem<T_data>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

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
        
            if( iGrid>=indexLimitsSolver[0] && iGrid<indexLimitsSolver[1] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[4] && kGrid<indexLimitsSolver[5] )
            {
                
                //localIdx = blockThreadIdx + haloSize;

                // offset for halo, as we only want to go over core cells
                //auto localIdx = blockThreadIdx + haloSize;
                const auto localIdx = blockThreadIdx + haloSize;
                bufDataZ(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2])/theta;
                if constexpr (DIM==3)
                {
                    const T_data fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1 + 1/r2)/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                                        ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                                        ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2;  
                }
                else if constexpr (DIM==2)
                {
                    const T_data fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1)/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1; 
                }
                else
                {
                    const T_data fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 )/theta );
                    // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    bufDataY(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                        ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0; 
                }
            } 
        }  
    }
};







template<int DIM, typename T_data> 
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
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataW, TMdSpan bufDataZ, const T_data delta, const T_data sigma, const T_data rhoCurr, const T_data rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
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

        const T_data r0 = ds[0]*ds[0];
        const T_data r1 = ds[1]*ds[1];
        const T_data r2 = ds[2]*ds[2];
        const T_data f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data fB = rhoCurr * 2 / delta;
        const T_data fZ = - rhoCurr * rhoOld;
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        auto* sdata = alpaka::getDynSharedMem<T_data>(acc);
        //auto& sdata = alpaka::declareSharedVar<T_data[sMemSizeFixed], __COUNTER__>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

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
    
        if( iGrid>=indexLimitsSolver[0] && iGrid<indexLimitsSolver[1] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[4] && kGrid<indexLimitsSolver[5] )
        {
            
            //localIdx = blockThreadIdx + haloSize;

            // offset for halo, as we only want to go over core cells
            //auto localIdx = blockThreadIdx + haloSize;
            auto const globalIdx = gridThreadIdx + haloSize;
            auto localIdx = blockThreadIdx + haloSize;
            if constexpr (DIM==3)
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2 +
                                bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ;  
            }
            else if constexpr (DIM==2)
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
            else
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 +
                                    bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
        }  
    }
};






template<int DIM, typename T_data> 
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
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataB, TMdSpan bufDataY, TMdSpan bufDataW, TMdSpan bufDataZ, const T_data delta, const T_data sigma, const T_data rhoCurr, const T_data rhoOld, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize, const alpaka::Vec<alpaka::DimInt<3>, Idx> offset ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        //auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        //auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        //auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        const T_data r0 = ds[0]*ds[0];
        const T_data r1 = ds[1]*ds[1];
        const T_data r2 = ds[2]*ds[2];
        const T_data f0 = rhoCurr * 2 / delta * (  1/r0  );
        const T_data f1 = rhoCurr * 2 / delta * (  1/r1  );
        const T_data f2 = rhoCurr * 2 / delta * (  1/r2  );
        const T_data fB = rhoCurr * 2 / delta;
        const T_data fZ = - rhoCurr * rhoOld;
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        //auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockStartThreadIdx = gridBlockIdx * blockThreadExtent;
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        auto* sdata = alpaka::getDynSharedMem<T_data>(acc);
        //auto& sdata = alpaka::declareSharedVar<T_data[sMemSizeFixed], __COUNTER__>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

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
    
        //if( iGrid>=indexLimitsSolver[0] && iGrid<indexLimitsSolver[1] && jGrid>=indexLimitsSolver[2] && jGrid<indexLimitsSolver[3] && kGrid>=indexLimitsSolver[4] && kGrid<indexLimitsSolver[5] )
        {
            
            //localIdx = blockThreadIdx + haloSize;

            // offset for halo, as we only want to go over core cells
            //auto localIdx = blockThreadIdx + haloSize;
            auto const globalIdx = gridThreadIdx + offset;
            auto localIdx = blockThreadIdx + haloSize;
            if constexpr (DIM==3)
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2 +
                                bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ;  
            }
            else if constexpr (DIM==2)
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
            else
            {
                const T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                bufDataW(globalIdx[0],globalIdx[1],globalIdx[2]) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 +
                                    bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]) * fB + bufDataZ(globalIdx[0],globalIdx[1],globalIdx[2]) * fZ ; 
            }
        }  
    }
};





// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev1KernelSharedMem<DIM,T_data>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev1KernelSharedMem<DIM,T_data> const&  kernel ,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent, // dimensions of elements per thread
            TMdSpan bufDataB, 
            TMdSpan bufDataY, 
            TMdSpan bufDataZ, 
            const T_data delta, 
            const T_data theta, 
            const T_data rhoCurr,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize) // allocated filter width
        {
            // Reserve the buffer, buffers size is the number of elements in a block (tile)
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
    struct BlockSharedMemDynSizeBytes<Chebyshev2KernelSharedMem<DIM,T_data>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev2KernelSharedMem<DIM,T_data> const&,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent,
            TMdSpan bufDataB, 
            TMdSpan bufDataY,
            TMdSpan bufDataW, 
            TMdSpan bufDataZ, 
            const T_data delta, 
            const T_data sigma, 
            const T_data rhoCurr,
            const T_data rhoOld,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize)  -> std::size_t
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
    struct BlockSharedMemDynSizeBytes<Chebyshev2KernelSharedMemSolver<DIM,T_data>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev2KernelSharedMemSolver<DIM,T_data> const&,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent,
            TMdSpan bufDataB, 
            TMdSpan bufDataY,
            TMdSpan bufDataW, 
            TMdSpan bufDataZ, 
            const T_data delta, 
            const T_data sigma, 
            const T_data rhoCurr,
            const T_data rhoOld,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize,
            const alpaka::Vec<alpaka::DimInt<3>, Idx> offset)  -> std::size_t
        {
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data);
        }
    };
} // namespace alpaka::trait
