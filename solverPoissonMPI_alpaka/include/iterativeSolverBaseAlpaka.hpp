#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once
#include <alpaka/alpaka.hpp>
#include "solverSetup.hpp"
#include "blockGrid.hpp"
#include "matrixFreeOperatorA.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"
#include "kernelsAlpakaIterativeSolverBase.hpp"

template <int DIM, typename T_data, int maxIteration>
class IterativeSolverBaseAlpaka{
    public:

    IterativeSolverBaseAlpaka(const BlockGrid<DIM,T_data>& blockGrid,const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs,
                            CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
        blockGrid_(blockGrid),
        exactSolutionAndBCs_(exactSolutionAndBCs),
        communicatorMPI_(communicatorMPI),
        alpakaHelper_(alpakaHelper),
        queueSolver_({alpakaHelper.devAcc_}),
        queueSolverNonBlocking1_({alpakaHelper.devAcc_}),
        queueSolverNonBlocking2_({alpakaHelper.devAcc_}),
        my_rank_(blockGrid.getMyrank()),
        ntot_ranks_(blockGrid.getNranks()[0]*blockGrid.getNranks()[1]*blockGrid.getNranks()[2]),
        globalLocation_(blockGrid.getGlobalLocation()),
        origin_(blockGrid.getOrigin()),
        ds_(blockGrid.getDs()),
        guards_(blockGrid.getGuards()),
        indexLimitsData_(blockGrid.getIndexLimitsData()),
        indexLimitsSolver_(blockGrid.getIndexLimitsSolver()),
        nlocal_noguards_(blockGrid.getNlocalNoGuards()),
        nlocal_guards_(blockGrid.getNlocalGuards()),
        ntotlocal_guards_(blockGrid.getNtotLocalGuards()),
        hasBoundary_(blockGrid.getHasBoundary()),
        bcsType_(blockGrid.getBcsType()),
        stride_j_(blockGrid.getNlocalGuards()[0]),
        stride_k_(blockGrid.getNlocalGuards()[0]*blockGrid.getNlocalGuards()[1]),
        eigenValuesLocal_(blockGrid.getEigenValuesLocal()),
        eigenValuesGlobal_(blockGrid.getEigenValuesGlobal()),
        normFieldB_(1.0),
        errorFromIteration_(-1.0),
        errorComputeOperator_(-1.0),
        numIterationFinal_(0),
        bufTmp(alpaka::allocBuf<T_data, Idx>(alpakaHelper.devAcc_, alpakaHelper.extent_)),
        workDivExtentResetNeumanBCsKernel_(alpaka::getValidWorkDiv(alpakaHelper.cfgExtentNoHaloHelper_, alpakaHelper.devAcc_, ResetNeumanBCsKernel<DIM, T_data, false, true>{}, 
                                                    alpaka::experimental::getMdSpan(this->bufTmp), this->exactSolutionAndBCs_, 1, 1.0, 
                                                    alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.haloSize_, alpakaHelper.ds_, alpakaHelper.origin_, 
                                                    alpakaHelper.globalLocation_, alpakaHelper.nlocal_noguards_, alpakaHelper.haloSize_ ) ),
        workDivExtentNormalizeProblem_(alpaka::getValidWorkDiv(alpakaHelper.cfgExtentSolverHelper_, alpakaHelper.devAcc_, DotProductKernel<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->bufTmp), alpaka::experimental::getMdSpan(this->bufTmp), 
                                                    ptrTmp, alpakaHelper.offsetSolver_)),
        workDivExtentApplyNeumanBCsFieldBKernel_(alpaka::getValidWorkDiv(alpakaHelper.cfgExtentNoHaloHelper_, alpakaHelper.devAcc_, ApplyNeumanBCsFieldBKernel<DIM, T_data, true>{}, 
                                                    alpaka::experimental::getMdSpan(this->bufTmp), this->exactSolutionAndBCs_, 1, 
                                                    alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.ds_, alpakaHelper.origin_, 
                                                    alpakaHelper.globalLocation_, alpakaHelper.nlocal_noguards_, alpakaHelper.haloSize_ )),
        workDivExtentApplyDirichletBCsFieldBKernel_(alpaka::getValidWorkDiv(alpakaHelper.cfgExtentNoHaloHelper_, alpakaHelper.devAcc_, ApplyDirichletBCsFieldBKernel<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->bufTmp),alpaka::experimental::getMdSpan(this->bufTmp), 1, 
                                                    alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.haloSize_, alpakaHelper.ds_, alpakaHelper.haloSize_ )),
        workDivExtentApplyDirichletBCsFromFunctionKernel_(alpaka::getValidWorkDiv(alpakaHelper.cfgExtentNoHaloHelper_, alpakaHelper.devAcc_, ApplyDirichletBCsFromFunctionKernel<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->bufTmp), this->exactSolutionAndBCs_, 
                                                    alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.ds_, alpakaHelper.origin_, 
                                                    alpakaHelper.globalLocation_, alpakaHelper.nlocal_noguards_, alpakaHelper.haloSize_ )),
        workDivExtentSetFieldValuefromFunctionKernel_(alpaka::getValidWorkDiv(alpakaHelper.cfgExtentNoHaloHelper_, alpakaHelper.devAcc_, SetFieldValuefromFunctionKernel<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->bufTmp), this->exactSolutionAndBCs_,alpakaHelper.indexLimitsDataAlpaka_, alpakaHelper.ds_, alpakaHelper.origin_, 
                                                    alpakaHelper.globalLocation_, alpakaHelper.nlocal_noguards_, alpakaHelper.haloSize_ ))
    {
        errorFromIterationHistory_ = new T_data[maxIteration];
    }
    
    ~IterativeSolverBaseAlpaka()
    {
        delete[] errorFromIterationHistory_;
    }

    void setProblemAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB) 
    {
        applyDirichletBCsFromFunctionAlpaka(bufX);
        setFieldValuefromFunctionAlpaka(bufB);
        alpaka::wait(this->queueSolverNonBlocking1_);
        alpaka::wait(this->queueSolverNonBlocking2_);
    }

    virtual void operator()(T_data fieldX[],T_data fieldB[])
    {

    }

    
    template<bool isMainLoop, bool fieldData>
    void resetNeumanBCsAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufData)
    {
        alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx{0,0,0};
        alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge{0,0,0,0,0,0};
        
        int dirAlpaka;

        auto bufDataMdSpan = alpaka::experimental::getMdSpan(bufData);
        
        ResetNeumanBCsKernel<DIM, T_data, isMainLoop*fieldData, true> resetNeumanBCsKernelNegative;
        ResetNeumanBCsKernel<DIM, T_data, isMainLoop*fieldData, false> resetNeumanBCsKernelPositive;
        
        for(int dir=0; dir<DIM; dir++)
        {
            dirAlpaka = 2 - dir;
            if(hasBoundary_[2*dir] && bcsType_[2*dir]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                adjustIdx[dirAlpaka]=1;
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentResetNeumanBCsKernel_, resetNeumanBCsKernelNegative, bufDataMdSpan, this->exactSolutionAndBCs_, dirAlpaka, this->normFieldB_, 
                                    indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, adjustIdx, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                    this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
            }
            if(hasBoundary_[2*dir+1] && bcsType_[2*dir+1]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                indexLimitsEdge[2*dirAlpaka]=indexLimitsEdge[2*dirAlpaka+1] - guards_[dir];
                adjustIdx[dirAlpaka]=-1;
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentResetNeumanBCsKernel_, resetNeumanBCsKernelPositive, bufDataMdSpan, this->exactSolutionAndBCs_, dirAlpaka, this->normFieldB_, 
                                    indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, adjustIdx, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                    this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
            }
        }
    }



    template<typename T_data_here, bool isMainLoop, bool fieldData>
    void resetNeumanBCsAlpakaCast(alpaka::Buf<T_Dev, T_data_here, Dim, Idx>& bufData)
    {
        alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx{0,0,0};
        alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge{0,0,0,0,0,0};
        
        int dirAlpaka;

        auto bufDataMdSpan = alpaka::experimental::getMdSpan(bufData);
        
        ResetNeumanBCsKernel<DIM, T_data, isMainLoop*fieldData, true> resetNeumanBCsKernelNegative;
        ResetNeumanBCsKernel<DIM, T_data, isMainLoop*fieldData, false> resetNeumanBCsKernelPositive;
        
        for(int dir=0; dir<DIM; dir++)
        {
            dirAlpaka = 2 - dir;
            if(hasBoundary_[2*dir] && bcsType_[2*dir]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                adjustIdx[dirAlpaka]=1;
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentResetNeumanBCsKernel_, resetNeumanBCsKernelNegative, bufDataMdSpan, this->exactSolutionAndBCs_, dirAlpaka, this->normFieldB_, 
                                    indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, adjustIdx, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                    this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
            }
            if(hasBoundary_[2*dir+1] && bcsType_[2*dir+1]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                indexLimitsEdge[2*dirAlpaka]=indexLimitsEdge[2*dirAlpaka+1] - guards_[dir];
                adjustIdx[dirAlpaka]=-1;
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentResetNeumanBCsKernel_, resetNeumanBCsKernelPositive, bufDataMdSpan, this->exactSolutionAndBCs_, dirAlpaka, this->normFieldB_, 
                                    indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, adjustIdx, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                    this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
            }
        }
    }

    
    template<bool isMainLoop, bool communicationON>
    T_data normalizeProblemToFieldBNormAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB)
    {
        this->normFieldB_=1;

        auto bufXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto bufBMdSpan = alpaka::experimental::getMdSpan(bufB);
        
        T_data pSum[1]={0};
        T_data totSum[1]={0};
        
        auto dotPorductDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent1D1_);
        T_data* const ptrDotPorductDev{std::data(dotPorductDev)};

        auto pSumView = createView(this->alpakaHelper_.devHost_, pSum, this->alpakaHelper_.extent1D1_);
        auto totSumView = createView(this->alpakaHelper_.devHost_, totSum, this->alpakaHelper_.extent1D1_);
        auto totSumDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent1D1_);
        T_data* const ptrTotSumDev{std::data(totSumDev)};

        DotProductKernel<DIM,T_data> dotProductKernel;
        
        if constexpr (isMainLoop)
        {
            memcpy(this->queueSolver_, bufTmp, bufB, this->alpakaHelper_.extent_);
            auto bufTmpMdSpan = alpaka::experimental::getMdSpan(bufTmp);
            adjustFieldBForDirichletNeumanBCsAlpaka(bufX,bufTmp);
            alpaka::exec<Acc>(this->queueSolver_,workDivExtentNormalizeProblem_, dotProductKernel, bufTmpMdSpan, bufTmpMdSpan, 
                                ptrDotPorductDev, this->alpakaHelper_.offsetSolver_);
        }
        else
        {
            alpaka::exec<Acc>(this->queueSolver_,workDivExtentNormalizeProblem_, dotProductKernel, bufBMdSpan, bufBMdSpan, 
                                ptrDotPorductDev, this->alpakaHelper_.offsetSolver_);
        }
        
        if constexpr(communicationON)
        {
            MPI_Allreduce(ptrDotPorductDev, ptrTotSumDev, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
            memcpy(this->queueSolver_, totSumView, totSumDev, this->alpakaHelper_.extent1D1_);
            if(totSum[0]>0)
                this->normFieldB_ = std::sqrt(totSum[0]);
            else
                this->normFieldB_ = 1;
        }
        else
        {
            memcpy(this->queueSolver_, pSumView, dotPorductDev, this->alpakaHelper_.extent1D1_);
            //std::cout<< "Norm fieldB1: " << pSum[0] <<std::endl;
            if(pSum[0]>0)
                this->normFieldB_ = std::sqrt(pSum[0]);
            else
                this->normFieldB_ = 1;
        }
        
        this-> template resetNormFieldsAlpaka<communicationON>(bufX,bufB,1/this->normFieldB_);

        return this->normFieldB_;
    }


   
    template<bool isMainLoop,bool communicationON>
    T_data computeErrorOperatorAAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufRk)
    {
        auto fieldXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto fieldBMdSpan = alpaka::experimental::getMdSpan(bufB);
        auto rkMdSpan = alpaka::experimental::getMdSpan(bufRk);

        T_data pSum[1];
        T_data totSum[1];

        auto dotPorductDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent1D1_);
        T_data* const ptrDotPorductDev{std::data(dotPorductDev)};

        auto pSumView = createView(this->alpakaHelper_.devHost_, pSum, this->alpakaHelper_.extent1D1_);
        auto totSumView = createView(this->alpakaHelper_.devHost_, totSum, this->alpakaHelper_.extent1D1_);
        auto totSumDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent1D1_);
        T_data* const ptrTotSumDev{std::data(totSumDev)};

        ComputeErrorOperatorAKernel<DIM, T_data> computeErrorOperatorAKernel;
        auto workDivExtent = alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_, this->alpakaHelper_.devAcc_, computeErrorOperatorAKernel, fieldXMdSpan, fieldBMdSpan, rkMdSpan, 
                                    ptrDotPorductDev, this->alpakaHelper_.ds_, this->alpakaHelper_.offsetSolver_);
        

        if constexpr(communicationON)
        {
            this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufX));
            this->communicatorMPI_.waitAllandCheckRcv();
        }
           
        this-> template resetNeumanBCsAlpaka<isMainLoop,true>(bufX);

        alpaka::exec<Acc>(this->queueSolver_, workDivExtent,  computeErrorOperatorAKernel, fieldXMdSpan, fieldBMdSpan, rkMdSpan, 
                                    ptrDotPorductDev, this->alpakaHelper_.ds_, this->alpakaHelper_.offsetSolver_);
        

        if constexpr(communicationON)
        {
            MPI_Allreduce(ptrDotPorductDev, ptrTotSumDev, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
            memcpy(this->queueSolver_, totSumView, totSumDev, this->alpakaHelper_.extent1D1_);
            return std::sqrt(totSum[0]);
        }
        else
        {
            memcpy(this->queueSolver_, pSumView, dotPorductDev, this->alpakaHelper_.extent1D1_);
            return std::sqrt(pSum[0]);
        }
    }


    template<bool communicationON>
    void resetNormFieldsAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB, const T_data norm)
    {
        auto fieldXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto fieldBMdSpan = alpaka::experimental::getMdSpan(bufB);

        ResetNormFieldsKernel<DIM, T_data> resetNormFieldsKernel;
        auto workDivExtent = alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, resetNormFieldsKernel, fieldXMdSpan, fieldBMdSpan, 
                                    norm, this->alpakaHelper_.haloSize_);
        

        alpaka::exec<Acc>(this->queueSolver_, workDivExtent, resetNormFieldsKernel, fieldXMdSpan, fieldBMdSpan, norm, this->alpakaHelper_.haloSize_);
        

        if constexpr(communicationON)
        {
            this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufX));
            this->communicatorMPI_.waitAllandCheckRcv();
        }
    }


    template<bool isMainLoop, bool fieldData>
    void resetNeumanBCs(T_data field[])
    {
        std::array<int,6> indexLimitsBCs=indexLimitsData_;
        std::array<int,3> adjustIndex={0,0,0};
        int indxGuard,indxData,indxBord;
        T_data x,y,z;
        
        for(int dir=0; dir<DIM; dir++)
        {
            if(hasBoundary_[2*dir] && bcsType_[2*dir]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsBCs=indexLimitsData_;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + guards_[dir];
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {   
                            indxGuard = i - adjustIndex[0] + stride_j_*(j - adjustIndex[1]) + stride_k_*(k - adjustIndex[2]);
                            if constexpr(fieldData && isMainLoop) // apply Neuman BCs to field data --> take into account BCs value
                            {
                                x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                                y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                                z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                                
                                if constexpr(orderNeumanBcs==1) // Neuman BCs 1st order
                                {
                                    indxBord= i + stride_j_*j + stride_k_*k;
                                    field[indxGuard] = field[indxBord] - ds_[dir] * exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / this->normFieldB_;
                                }
                                else // Neuman BCs 2nd order 
                                {
                                    indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                    field[indxGuard] = field[indxData] - 2 * ds_[dir] * exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / this->normFieldB_;
                                }   
                            }
                            else // apply Neuman BCs to helper fields in iterative solver --> no BCs
                            {
                                if constexpr(orderNeumanBcs==1) // Neuman BCs 1st order
                                {
                                    indxBord= i + stride_j_*j + stride_k_*k;
                                    field[indxGuard] = field[indxBord];
                                }
                                else // Neuman BCs 2nd order 
                                {
                                    indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                    field[indxGuard] = field[indxData];
                                }
                            }
                        }
                    }
                }
            }
            if(hasBoundary_[2*dir+1] && bcsType_[2*dir+1]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsBCs=indexLimitsData_;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                adjustIndex[dir] = -1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxGuard = i - adjustIndex[0] + stride_j_*(j - adjustIndex[1]) + stride_k_*(k - adjustIndex[2]);
                            if constexpr(fieldData && isMainLoop) // apply Neuman BCs to field data --> take into account BCs value
                            {
                                x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                                y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                                z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                                
                                if constexpr(orderNeumanBcs==1) // Neuman BCs 1st order
                                {
                                    indxBord= i + stride_j_*j + stride_k_*k;
                                    field[indxGuard] = field[indxBord] + ds_[dir]*exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / this->normFieldB_;
                                }
                                else // Neuman BCs 2nd order 
                                {
                                    indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                    field[indxGuard] = field[indxData] + 2 * ds_[dir]*exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / this->normFieldB_;
                                }   
                            }
                            else // apply Neuman BCs to helper fields in iterative solver --> no BCs
                            {
                                if constexpr(orderNeumanBcs==1) // Neuman BCs 1st order
                                {
                                    indxBord= i + stride_j_*j + stride_k_*k;
                                    field[indxGuard] = field[indxBord];
                                }
                                else // Neuman BCs 2nd order 
                                {
                                    indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                    field[indxGuard] = field[indxData];
                                }
                            }   
                        }
                    }
                }
            }
        }
    }





    T_data checkSolutionLocalGlobal(T_data fieldX[])
    {
        T_data* solution = new T_data[ntotlocal_guards_];
        int i,j,k,indx;
        T_data x,y,z;
        T_data errorLocal=0;
        T_data errorLocalMax=-1;
        T_data err;
        T_data* errorGlobalMaxBlock = new T_data[ntot_ranks_];
        T_data* errorGlobalMaxPoint = new T_data[ntot_ranks_];

        this->communicatorMPI_(fieldX);
        //this->communicatorMPI_.waitAllandCheckSend();
        this->communicatorMPI_.waitAllandCheckRcv();
        this-> template resetNeumanBCs<true,true>(fieldX);

        for(k=indexLimitsData_[4]; k<indexLimitsData_[5]; k++)
        {
            for(j=indexLimitsData_[2]; j<indexLimitsData_[3]; j++)
            {
                for(i=indexLimitsData_[0]; i<indexLimitsData_[1]; i++)
                {
                    x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                    y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                    z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                    indx = i + stride_j_*j + stride_k_*k;
                    solution[indx] = exactSolutionAndBCs_.trueSolutionFxyz(x,y,z);
                    err = std::abs(fieldX[indx]-solution[indx]);
                    errorLocal += err;
                    if(err > errorLocalMax)
                        errorLocalMax = err;
                }
            }
        }
        //std::cout<<" Debug checksolution globalLocation " << globalLocation[0] << " "<<globalLocation[1] <<" "<< globalLocation[2]<< " errorLocal "<<errorLocal << 
        //       " errorLocalAvg "<< errorLocal/blockGrid.getNtotLocalNoGuards() <<" indexLimitsData "<< indexLimitsData[0]<< " "<<indexLimitsData[1] <<" "<< indexLimitsData[2] << " " <<indexLimitsData[3] << " "<< indexLimitsData[4] << " "<< indexLimitsData[5] << std::endl;

        MPI_Request* requestsSend = new MPI_Request[ntot_ranks_-1];
        MPI_Status*  statusesSend = new MPI_Status[ntot_ranks_-1];
        MPI_Request* requestsRcv = new MPI_Request[ntot_ranks_-1];
        MPI_Status*  statusesRcv = new MPI_Status[ntot_ranks_-1];

        errorGlobalMaxBlock[0]=errorLocal;
        for(int i=1; i<ntot_ranks_; i++)
        {
            if(my_rank_==0)
            {
                MPI_Irecv(&errorGlobalMaxBlock[i], 1, getMPIType<T_data>(), i, 0, MPI_COMM_WORLD, &requestsRcv[i-1]);
            }
            else if(my_rank_==i)
            {
                MPI_Isend(&errorLocal, 1, getMPIType<T_data>(), 0, 0, MPI_COMM_WORLD, &requestsSend[i-1]);
            }
        }
        if(my_rank_==0)
        {
            MPI_Waitall(ntot_ranks_-1, requestsRcv, statusesRcv);
            for (int i = 0; i < ntot_ranks_-1; i++) 
            {
                if (statusesRcv[i].MPI_ERROR != MPI_SUCCESS) 
                {
                    std::cerr << "Error in request " << i << ": " << statusesRcv[i].MPI_ERROR << std::endl;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        errorGlobalMaxPoint[0]=errorLocalMax;
        for(int i=1; i<ntot_ranks_; i++)
        {
            if(my_rank_==0)
            {
                MPI_Irecv(&errorGlobalMaxPoint[i], 1, getMPIType<T_data>(), i, 0, MPI_COMM_WORLD, &requestsRcv[i-1]);
            }
            else if(my_rank_==i)
            {
                MPI_Isend(&errorLocalMax, 1, getMPIType<T_data>(), 0, 0, MPI_COMM_WORLD, &requestsSend[i-1]);
            }
        }
        if(my_rank_==0)
        {
            MPI_Waitall(ntot_ranks_-1, requestsRcv, statusesRcv);
            for (int i = 0; i < ntot_ranks_-1; i++) 
            {
                if (statusesRcv[i].MPI_ERROR != MPI_SUCCESS) 
                {
                    std::cerr << "Error in request " << i << ": " << statusesRcv[i].MPI_ERROR << std::endl;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(my_rank_==0)
        {   
            errorGlobalMaxBlock[0]=errorLocal;
            errorGlobalMaxPoint[0]=errorLocalMax;
            T_data maxErorrLocalBlock = errorGlobalMaxBlock[0];
            T_data maxErorrLocalPoint = errorGlobalMaxPoint[0];
            int indxMaxErrorBlock=0;
            int indxMaxErrorPoint=0;
            for(int i=0; i<ntot_ranks_; i++)
            {
                if(errorGlobalMaxBlock[i]>maxErorrLocalBlock)
                {
                    maxErorrLocalBlock = errorGlobalMaxBlock[i];
                    indxMaxErrorBlock = i;
                }
                if(errorGlobalMaxPoint[i]>maxErorrLocalPoint)
                {
                    maxErorrLocalPoint = errorGlobalMaxPoint[i];
                    indxMaxErrorPoint = i;
                }
            }
            std::cout<<"Max error local block avg "<< maxErorrLocalBlock/blockGrid_.getNtotLocalNoGuards() << " in rank "<<indxMaxErrorBlock << std::endl;
            std::cout<<"Max error local point "<< maxErorrLocalPoint << " in rank "<<indxMaxErrorPoint << std::endl;
        }


               
        delete[] requestsSend;
        delete[] statusesSend;
        delete[] requestsRcv;
        delete[] statusesRcv;       
        delete[] solution;
        return errorLocal;
    }



    void printFieldWithGuards(T_data data[])
    {
        const std::array<int,3>  globalLocation = blockGrid_.getGlobalLocation();
        const std::array<int,3>  nlocal_guards = blockGrid_.getNlocalGuards();
        const int totRanks = blockGrid_.getNranks()[0]*blockGrid_.getNranks()[1]*blockGrid_.getNranks()[2];
        int i,j,k,indx;
        MPI_Barrier(MPI_COMM_WORLD);
        for(int n=0; n<totRanks; n++)    
        {   
            MPI_Barrier(MPI_COMM_WORLD);
            if(blockGrid_.getMyrank()==n)
            {
                std::cout<<"Print field with guards, globalLocation "<<globalLocation[0]<< " " <<globalLocation[1]<< " "<<globalLocation[2] <<std::endl;
                std::cout.flush();
                for(k=0;k<nlocal_guards[2];k++)
                {   
                    std::cout<<"indexZ = "<< k <<std::endl;
                    for(j=0;j<nlocal_guards[1];j++)
                    {
                        for(i=0;i<nlocal_guards[0];i++)
                        {
                            indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            std::cout<< std::setw(3) << data[indx];
                            std::cout<< " ";
                        }
                        std::cout<<std::endl;
                        std::cout.flush();
                    }
                }
                std::cout.flush();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    std::chrono::duration<double> getDurationSolver() const
    {
        return durationSolver_;
    }
    T_data getErrorFromIteration() const
    {
        return errorFromIteration_;
    }
    T_data getErrorComputeOperator() const
    {
        return errorComputeOperator_;
    }
    int getNumIterationFinal() const
    {
        return numIterationFinal_;
    }

    TimeCounter<DIM, T_data> timeCounter;

    protected:


    void adjustFieldBForDirichletNeumanBCsAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB)
    {
        alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx{0,0,0};
        alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge{0,0,0,0,0,0};
        
        int dirAlpaka;

        auto bufXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto bufBMdSpan = alpaka::experimental::getMdSpan(bufB);
        
        ApplyNeumanBCsFieldBKernel<DIM, T_data, true> applyNeumanBCsFieldBKernelNegative;
        ApplyNeumanBCsFieldBKernel<DIM, T_data, false> applyNeumanBCsFieldBKernelPositive;
        ApplyDirichletBCsFieldBKernel<DIM,T_data> applyDirichletBCsFieldBKernel;
        
        for(int dir=0; dir<DIM; dir++)
        {
            dirAlpaka = 2 - dir;
            if(hasBoundary_[2*dir])
            {
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                if(bcsType_[2*dir]==0)
                {
                    indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                    adjustIdx[dirAlpaka]=1;
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentApplyDirichletBCsFieldBKernel_, applyDirichletBCsFieldBKernel, bufXMdSpan, bufBMdSpan,dirAlpaka,
                                            indexLimitsEdge,adjustIdx,  this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);

                }
                else if(bcsType_[2*dir]==1) //neuman
                {
                    indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentApplyNeumanBCsFieldBKernel_, applyNeumanBCsFieldBKernelNegative, bufBMdSpan, this->exactSolutionAndBCs_, dirAlpaka, 
                                        indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                        this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
                }
            }
            if(hasBoundary_[2*dir+1])
            {
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                if(bcsType_[2*dir+1]==0)
                {
                    indexLimitsEdge[2*dirAlpaka]=indexLimitsEdge[2*dirAlpaka+1] - guards_[dir];
                    adjustIdx[dirAlpaka]=-1;
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentApplyDirichletBCsFieldBKernel_, applyDirichletBCsFieldBKernel, bufXMdSpan, bufBMdSpan,dirAlpaka,
                                            indexLimitsEdge,adjustIdx,  this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);

                }
                else if(bcsType_[2*dir+1]==1) //neuman
                {
                    indexLimitsEdge[2*dirAlpaka]=indexLimitsEdge[2*dirAlpaka+1] - guards_[dir];
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentApplyNeumanBCsFieldBKernel_, applyNeumanBCsFieldBKernelPositive, bufBMdSpan, this->exactSolutionAndBCs_, dirAlpaka, 
                                        indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                        this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
                }
            }
        }
    }

   
    
    void setFieldValuefromFunctionAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& buf)
    {   
        auto bufMdSpan = alpaka::experimental::getMdSpan(buf);

        SetFieldValuefromFunctionKernel<DIM,T_data> setFieldValuefromFunctionKernel;

        alpaka::exec<Acc>(this->queueSolverNonBlocking2_, workDivExtentSetFieldValuefromFunctionKernel_, setFieldValuefromFunctionKernel, bufMdSpan, this->exactSolutionAndBCs_, 
                                        this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                        this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
    }


    void applyDirichletBCsFromFunctionAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx>& buf)
    {
        alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge{0,0,0,0,0,0};
        
        int dirAlpaka;

        auto bufMdSpan = alpaka::experimental::getMdSpan(buf);
            
        ApplyDirichletBCsFromFunctionKernel<DIM, T_data> applyDirichletBCsFromFunctionKernel;
        
        for(int dir=0; dir<DIM; dir++)
        {
            dirAlpaka = 2 - dir;
            if(hasBoundary_[2*dir] && bcsType_[2*dir]==0)
            {
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                alpaka::exec<Acc>(this->queueSolverNonBlocking1_, workDivExtentApplyDirichletBCsFromFunctionKernel_, applyDirichletBCsFromFunctionKernel, bufMdSpan, this->exactSolutionAndBCs_, 
                                        indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                        this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );

            }
            if(hasBoundary_[2*dir+1] && bcsType_[2*dir+1]==0)
            {
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                indexLimitsEdge[2*dirAlpaka]=indexLimitsEdge[2*dirAlpaka+1] - guards_[dir];
                alpaka::exec<Acc>(this->queueSolverNonBlocking1_, workDivExtentApplyDirichletBCsFromFunctionKernel_, applyDirichletBCsFromFunctionKernel, bufMdSpan, this->exactSolutionAndBCs_, 
                                        indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                        this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
            }
        }
    }



    const BlockGrid<DIM,T_data>& blockGrid_;
    const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs_;
    CommunicatorMPI<DIM,T_data>& communicatorMPI_;
    const AlpakaHelper<DIM,T_data>& alpakaHelper_;
    alpaka::Queue<Acc, alpaka::Blocking> queueSolver_;
    alpaka::Queue<Acc, alpaka::NonBlocking> queueSolverNonBlocking1_;
    alpaka::Queue<Acc, alpaka::NonBlocking> queueSolverNonBlocking2_;
    const int my_rank_;
    const int ntot_ranks_;
    const std::array<int,3> globalLocation_;
    const std::array<T_data,3> origin_;
    const std::array<T_data,3> ds_;
    const std::array<int,3> guards_;
    const std::array<int,6> indexLimitsData_;
    const std::array<int,6> indexLimitsSolver_;
    const std::array<int,3> nlocal_noguards_;
    const std::array<int,3> nlocal_guards_;
    const int ntotlocal_guards_;
    const std::array<bool,6> hasBoundary_;
    const std::array<int,6> bcsType_;
    const int stride_j_;
    const int stride_k_;

    std::array<T_data,2> eigenValuesLocal_;
    std::array<T_data,2> eigenValuesGlobal_;


    T_data normFieldB_;
    T_data errorFromIteration_;
    T_data errorComputeOperator_;
    int numIterationFinal_;

    std::chrono::duration<double> durationSolver_;

    T_data* errorFromIterationHistory_;
    T_data* ptrTmp;
    //alpaka::KernelCfg<Acc> const cfgExtentNoHaloSolverBase_;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufTmp;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentResetNeumanBCsKernel_;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentNormalizeProblem_;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentApplyNeumanBCsFieldBKernel_;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentApplyDirichletBCsFieldBKernel_;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentApplyDirichletBCsFromFunctionKernel_;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentSetFieldValuefromFunctionKernel_;

    private:
};