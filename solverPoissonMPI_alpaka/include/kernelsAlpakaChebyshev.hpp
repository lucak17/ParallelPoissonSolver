#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdio>

template<int DIM, typename T_data> 
struct AssignFieldWith1FieldKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, const T_data constB,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i);
        }
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
        T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
        T_data f0 = rhoCurr * 2 / delta * (  1/r0  );
        T_data f1 = rhoCurr * 2 / delta * (  1/r1  );
        T_data f2 = rhoCurr * 2 / delta * (  1/r2  );
        T_data fB = rhoCurr * 2 / delta;
        T_data fZ = - rhoCurr * rhoOld;

        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {            

            if constexpr (DIM==3)
            {
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(k,j,i) = bufDataY(k,j,i) * fc0 + 
                                    ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0 + 
                                    ( bufDataY(k,j-1,i) + bufDataY(k,j+1,i)) * f1 +
                                    ( bufDataY(k-1,j,i) + bufDataY(k+1,j,i) ) * f2 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ;  
             }
            else if constexpr (DIM==2)
            {
                T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(k,j,i) = bufDataY(k,j,i) * fc0 + 
                                    ( bufDataY(k,j,i-1) + bufDataY(k,j,i + 1) ) * f0 + 
                                    ( bufDataY(k,j-1,i) + bufDataY(k,j+1,i)) * f1 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ; 
            }
            else
            {
                T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
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
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        //auto& sdata = alpaka::declareSharedVar<T_data[4096], __COUNTER__>(acc);
        auto* sdata = alpaka::getDynSharedMem<T_data>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);
 
        auto localIdx = blockThreadIdx + haloSize;
        auto globalIdx = gridThreadIdx;
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            
            sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataB(k,j,i);

            alpaka::Vec<alpaka::DimInt<3>,Idx> shift = {0,0,0}; 
            
            for(int dir=2; dir >= 0; dir=dir-1)
            {
                if(blockThreadIdx[dir] == 0 || gridThreadIdx[dir] == indexLimitsSolver[2*(2-dir)] )
                {                    
                    shift = {0, 0, 0};
                    shift[dir] -= haloSize[dir]; 
                    localIdx = blockThreadIdx + haloSize + shift;
                    globalIdx = gridThreadIdx + shift;
                    //printf("hello1 dir %lu, blockThreadIdx %lu %lu %lu - gridThreadIdx %lu %lu %lu - globalIdx %lu %lu %lu\n",
                     //       dir,blockThreadIdx[0],blockThreadIdx[1],blockThreadIdx[2], gridThreadIdx[0],gridThreadIdx[1],gridThreadIdx[2],globalIdx[0],globalIdx[1],globalIdx[2]);
                    sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]);
                }
                if(blockThreadIdx[dir] == ( blockThreadExtent[dir] -1 ) || (gridThreadIdx[dir] == indexLimitsSolver[2*(2-dir)+1]-1))
                {
                    shift = {0,0,0};
                    shift[dir] += haloSize[dir];
                    localIdx = blockThreadIdx + haloSize + shift;
                    globalIdx = gridThreadIdx + shift;
                    //printf("hello2 dir %d, blockThreadIdx %lu %lu %lu - gridThreadIdx %lu %lu %lu - globalIdx %lu %lu %lu\n",
                     //       dir,blockThreadIdx[0],blockThreadIdx[1],blockThreadIdx[2], gridThreadIdx[0],gridThreadIdx[1],gridThreadIdx[2],globalIdx[0],globalIdx[1],globalIdx[2]);
                    sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]);
                }
            }
        
            
            alpaka::syncBlockThreads(acc);
            
            localIdx = blockThreadIdx + haloSize;
            bufDataZ(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2])/theta;
        
            if constexpr (DIM==3)
            {
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2;  
             }
            else if constexpr (DIM==2)
            {
                fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1)/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1; 
            }
            else
            {
                fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 )/theta );
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0; 
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
        T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 + 1/r2) );
        T_data f0 = rhoCurr * 2 / delta * (  1/r0  );
        T_data f1 = rhoCurr * 2 / delta * (  1/r1  );
        T_data f2 = rhoCurr * 2 / delta * (  1/r2  );
        T_data fB = rhoCurr * 2 / delta;
        T_data fZ = - rhoCurr * rhoOld;
                
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        auto* sdata = alpaka::getDynSharedMem<T_data>(acc);
        auto sdataMdSpan = std::experimental::mdspan<T_data, std::experimental::dextents<Idx, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

        auto localIdx = blockThreadIdx + haloSize;
        auto globalIdx = gridThreadIdx;

        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {            
            sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataY(k,j,i);

            alpaka::Vec<alpaka::DimInt<3>,Idx> shift = {0,0,0}; 
            
            for(int dir=2; dir >= 0; dir=dir-1)
            {
                if(blockThreadIdx[dir] == 0 || gridThreadIdx[dir] == indexLimitsSolver[2*(2-dir)] )
                {                    
                    shift = {0, 0, 0};
                    shift[dir] -= haloSize[dir]; 
                    localIdx = blockThreadIdx + haloSize + shift;
                    globalIdx = gridThreadIdx + shift;
                    sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataY(globalIdx[0],globalIdx[1],globalIdx[2]);
                }
                if(blockThreadIdx[dir] == ( blockThreadExtent[dir] -1 ) || (gridThreadIdx[dir] == indexLimitsSolver[2*(2-dir)+1]-1))
                {
                    shift = {0,0,0};
                    shift[dir] += haloSize[dir];
                    localIdx = blockThreadIdx + haloSize + shift;
                    globalIdx = gridThreadIdx + shift;
                    sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataY(globalIdx[0],globalIdx[1],globalIdx[2]);
                }
            }
        
            alpaka::syncBlockThreads(acc);
            
            localIdx = blockThreadIdx + haloSize;        
            if constexpr (DIM==3)
            {
                // Z Y X fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                bufDataW(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ;  
             }
            else if constexpr (DIM==2)
            {
                T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0 + 1/r1 ) );
                // Z Y X
                bufDataW(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ; 
            }
            else
            {
                T_data fc0 =  rhoCurr * ( 2 * sigma  - 4 / delta * ( 1/r0  ) );
                // Z Y X
                bufDataW(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 +
                                    bufDataB(k,j,i) * fB + bufDataZ(k,j,i) * fZ ;

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

