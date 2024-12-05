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
        numIterationFinal_(0)
    {
        errorFromIterationHistory_ = new T_data[maxIteration];
    }
    
    ~IterativeSolverBaseAlpaka()
    {
        delete[] errorFromIterationHistory_;
    }

    void setProblem(T_data fieldX[],T_data fieldB[]) 
    {
        applyDirichletBCsFromFunction(fieldX);
        setFieldValuefromFunction(fieldB);
    }

    virtual void operator()(T_data fieldX[],T_data fieldB[])
    {

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



    template<bool isMainLoop, bool fieldData>
    void resetNeumanBCsAlpaka(alpaka::Buf<T_Dev, T_data, Dim, Idx> bufData)
    {

        std::array<int,6> indexLimitsBCs=indexLimitsData_;
        alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx{0,0,0};
        alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge{0,0,0,0,0,0};
        int indxGuard,indxData,indxBord;
        int dirAlpaka;
        T_data x,y,z;

        auto bufDataMdSpan = alpaka::experimental::getMdSpan(bufData);
        alpaka::Queue<Acc, alpaka::Blocking> queue{this->alpakaHelper_.devAcc_};
        alpaka::KernelCfg<Acc> const cfgExtentNoHalo{this->alpakaHelper_.extentNoHalo_, this->alpakaHelper_.elemPerThread_};

        ResetNeumanBCsKernel<DIM,T_data, isMainLoop*fieldData> resetNeumanBCsKernel;
        auto workDivExtent = alpaka::getValidWorkDiv(cfgExtentNoHalo, this->alpakaHelper_.devAcc_, resetNeumanBCsKernel, 
                                                    bufDataMdSpan, this->exactSolutionAndBCs_, 1, this->normFieldB_, 
                                                    indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, adjustIdx, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                                    this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
        
        for(int dir=0; dir<DIM; dir++)
        {
            dirAlpaka = DIM - 1 - dir;
            if(hasBoundary_[2*dir] && bcsType_[2*dir]==1)
            {
                //std::cout<< "Debug in resetNeumanBCs " <<std::endl;
                indexLimitsBCs=indexLimitsData_;
                indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                adjustIdx={0,0,0};
                indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                adjustIdx[dirAlpaka]=1;
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

                                indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                //field[indxGuard] = field[indxData] - 2 * ds_[dir] * exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / this->normFieldB_;  
                            }
                            else // apply Neuman BCs to helper fields in iterative solver --> no BCs
                            {
                                indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                //field[indxGuard] = field[indxData];
                        
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
                                
                                indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                //field[indxGuard] = field[indxData] + 2 * ds_[dir]*exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / this->normFieldB_;
                            
                            }
                            else // apply Neuman BCs to helper fields in iterative solver --> no BCs
                            {
                                indxData = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                //field[indxGuard] = field[indxData];
                            }   
                        }
                    }
                }
            }
        }
    }

    template<bool isMainLoop, bool communicationON>
    T_data normalizeProblemToFieldBNorm(T_data fieldX[], T_data fieldB[])
    {
        this->normFieldB_=1;
        T_data pSumNorm=0.0;
        int indx;
        if constexpr (isMainLoop)
        {
            T_data* fieldBtmp = new T_data[this->ntotlocal_guards_];
            std::memcpy(fieldBtmp, fieldB, this->ntotlocal_guards_* sizeof(T_data));
            adjustFieldBForDirichletNeumanBCs(fieldX,fieldBtmp);

            for(int k=this->indexLimitsSolver_[4]; k< this->indexLimitsSolver_[5]; k++ )
            {
                for(int j=this->indexLimitsSolver_[2]; j< this->indexLimitsSolver_[3]; j++ )
                {
                    for(int i=this->indexLimitsSolver_[0]; i< this->indexLimitsSolver_[1]; i++ )
                    {
                        indx = i + this->stride_j_*j + this->stride_k_*k; 
                        pSumNorm += fieldBtmp[indx]*fieldBtmp[indx];
                    }
                }
            }
            /*
            for(int i=0; i< this-> ntotlocal_guards_; i++)
            {
                pSumNorm += fieldBtmp[i]*fieldBtmp[i];
            }
            */
            delete[] fieldBtmp;
        }
        else
        {
            for(int k=this->indexLimitsSolver_[4]; k< this->indexLimitsSolver_[5]; k++ )
            {
                for(int j=this->indexLimitsSolver_[2]; j< this->indexLimitsSolver_[3]; j++ )
                {
                    for(int i=this->indexLimitsSolver_[0]; i< this->indexLimitsSolver_[1]; i++ )
                    {
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        pSumNorm += fieldB[indx]*fieldB[indx];
                    }
                }
            }
        }
        if constexpr (communicationON)
        {
            MPI_Reduce(&pSumNorm, &normFieldB_, 1, getMPIType<T_data>(), MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&normFieldB_, 1, getMPIType<T_data>(), 0, MPI_COMM_WORLD);
            this->normFieldB_=std::sqrt(this->normFieldB_);
        }
        else
        {
            this->normFieldB_=std::sqrt(pSumNorm);
        }
        
        for(int i=0; i<ntotlocal_guards_; i++)
        {
            fieldX[i]/=this->normFieldB_;
            fieldB[i]/=this->normFieldB_;
        }

        return this->normFieldB_;
    }

    template<bool isMainLoop,bool communicationON>
    T_data computeErrorOperatorA(T_data fieldX[], const T_data fieldB[], T_data rk[], const MatrixFreeOperatorA<DIM,T_data>& operatorA)
    {
        const int imin=indexLimitsSolver_[0];
        const int imax=indexLimitsSolver_[1];
        const int jmin=indexLimitsSolver_[2];
        const int jmax=indexLimitsSolver_[3];
        const int kmin=indexLimitsSolver_[4];
        const int kmax=indexLimitsSolver_[5];
        int i,j,k,indx;

        T_data pSum=0.0;
        T_data totSum=0.0;

        if constexpr(communicationON)
        {
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
        this-> template resetNeumanBCs<isMainLoop,true>(fieldX);

        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx = i + this->stride_j_*j + this->stride_k_*k;
                    rk[indx] = fieldB[indx] - operatorA(i,j,k,fieldX);
                    pSum += rk[indx]*rk[indx];
                }
            }
        }

        if constexpr(communicationON)
        {
            MPI_Reduce(&pSum, &totSum, 1, getMPIType<T_data>(), MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&totSum, 1, getMPIType<T_data>(), 0, MPI_COMM_WORLD);
            return std::sqrt(totSum);
        }
        else
        {
            return std::sqrt(pSum);
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

    void adjustFieldBForDirichletNeumanBCs(T_data fieldX[],T_data fieldB[])
    {
        std::array<int,6> indexLimitsBCs=indexLimitsData_;
        std::array<int,3> adjustIndex={0,0,0};
        int indxX,indxB;
        T_data x,y,z;
        
        for(int dir=0; dir<DIM; dir++)
        {
            if(hasBoundary_[2*dir])
            {
                indexLimitsBCs=indexLimitsData_;
                adjustIndex={0,0,0};
                if(bcsType_[2*dir]==0) // Dirichlet
                {
                    indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + guards_[dir];
                    adjustIndex[dir]=1;
                    for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5]; k++)
                    {
                        for(int j=indexLimitsBCs[2]; j<indexLimitsBCs[3]; j++)
                        {
                            for(int i=indexLimitsBCs[0]; i<indexLimitsBCs[1]; i++)
                            {
                                indxX = i + stride_j_*j + stride_k_*k;
                                indxB = i+adjustIndex[0] + stride_j_*(j+adjustIndex[1]) + stride_k_*(k+adjustIndex[2]);
                                fieldB[indxB] -= fieldX[indxX]/(ds_[dir]*ds_[dir]);   
                            }
                        }
                    }
                }
                else if(bcsType_[2*dir]==1) //neuman
                {
                    indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + guards_[dir];
                    for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5]; k++)
                    {
                        for(int j=indexLimitsBCs[2]; j<indexLimitsBCs[3]; j++)
                        {
                            for(int i=indexLimitsBCs[0]; i<indexLimitsBCs[1]; i++)
                            {

                                x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                                y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                                z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                                indxB = i + stride_j_*j + stride_k_*k;
                                if constexpr(orderNeumanBcs==1) // Neuman BCs 1st order
                                {
                                    fieldB[indxB] += exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / ds_[dir];
                                }
                                else // Neuman BCs 2nd order 
                                {
                                    fieldB[indxB] += 2 * exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / ds_[dir];
                                }
                            }
                        }
                    }
                }
            }
            if(hasBoundary_[2*dir+1])
            {
                indexLimitsBCs=indexLimitsData_;
                adjustIndex={0,0,0};
                if(bcsType_[2*dir+1]==0) // Dirichlet
                {
                    indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - guards_[dir];
                    adjustIndex[dir]=-1;
                    for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5]; k++)
                    {
                        for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3]; j++)
                        {
                            for(int i=indexLimitsBCs[0]; i<indexLimitsBCs[1]; i++)
                            {
                                indxX = i + stride_j_*j + stride_k_*k;
                                indxB = i + adjustIndex[0] + stride_j_*(j + adjustIndex[1]) + stride_k_*(k + adjustIndex[2]);
                                fieldB[indxB] -= fieldX[indxX]/(ds_[dir]*ds_[dir]);   
                            }
                        }
                    }
                }
                else if(bcsType_[2*dir+1]==1) //neuman
                {
                    indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - guards_[dir];
                    for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                    {
                        for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                        {
                            for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                            {
                                x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                                y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                                z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                                indxB = i + stride_j_*j + stride_k_*k;
                                if constexpr(orderNeumanBcs==1) // Neuman BCs 1st order
                                {
                                    fieldB[indxB] -= exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / ds_[dir];
                                }
                                else // Neuman BCs 2nd order 
                                {
                                    fieldB[indxB] -= 2 * exactSolutionAndBCs_.trueSolutionDdir(x,y,z,dir) / ds_[dir];
                                }   
                            }
                        }
                    }
                }
            }
        }
    }


    inline void setFieldValuefromFunction(T_data data[])
    {   
        int i,j,k,indx;
        T_data x,y,z;
        for(k=indexLimitsData_[4]; k<indexLimitsData_[5]; k++)
        {
            for(j=indexLimitsData_[2]; j<indexLimitsData_[3]; j++)
            {
                for(i=indexLimitsData_[0]; i<indexLimitsData_[1]; i++)
                {
                    x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                    y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                    z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                    indx= i + this->stride_j_*j + this->stride_k_*k;
                    data[indx]= exactSolutionAndBCs_.setFieldB(x,y,z);
                }
            }
        }
    }

    void applyDirichletBCsFromFunction(T_data data[])
        {
            int indx;
            T_data x,y,z;
            for(int dir=0; dir<DIM; dir++)
            {
                if(hasBoundary_[2*dir] && bcsType_[2*dir] == 0)
                {
                    std::array<int,6> indexLimitsBCs=indexLimitsData_;
                    indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + guards_[dir];
                    for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5]; k++)
                    {
                        for(int j=indexLimitsBCs[2]; j<indexLimitsBCs[3]; j++)
                        {
                            for(int i=indexLimitsBCs[0]; i<indexLimitsBCs[1]; i++)
                            {
                                x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                                y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                                z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                                indx = i + this->stride_j_*j + this->stride_k_*k;
                                data[indx]=exactSolutionAndBCs_.trueSolutionFxyz(x,y,z);   
                            }
                        }
                    }
                }
                if(hasBoundary_[2*dir+1] && bcsType_[2*dir+1] == 0)
                {
                    std::array<int,6> indexLimitsBCs=indexLimitsData_;
                    indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - guards_[dir];
                    for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                    {
                        for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                        {
                            for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                            {
                                x=origin_[0] + (i-indexLimitsData_[0])*ds_[0] + globalLocation_[0]*(nlocal_noguards_[0])*ds_[0];
                                y=origin_[1] + (j-indexLimitsData_[2])*ds_[1] + globalLocation_[1]*(nlocal_noguards_[1])*ds_[1];
                                z=origin_[2] + (k-indexLimitsData_[4])*ds_[2] + globalLocation_[2]*(nlocal_noguards_[2])*ds_[2];
                                indx = i + this->stride_j_*j + this->stride_k_*k;
                                data[indx]=exactSolutionAndBCs_.trueSolutionFxyz(x,y,z);   
                            }
                        }
                    }
                
                }
            }
        }



    const BlockGrid<DIM,T_data>& blockGrid_;
    const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs_;
    CommunicatorMPI<DIM,T_data>& communicatorMPI_;
    const AlpakaHelper<DIM,T_data>& alpakaHelper_;
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

    private:
};