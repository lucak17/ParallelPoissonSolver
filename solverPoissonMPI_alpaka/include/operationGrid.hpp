#include <mpi.h>
#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 
#include <thread>
#include <chrono> // For std::chrono

#pragma once
#include "blockGrid.hpp"
#include "inputParam.hpp"
#include "communicationMPI.hpp"




template<int DIM, typename T_data>
void setFieldValuefromFunction(const BlockGrid<DIM,T_data>& blockGrid,T_data data[],ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs)
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,3>  nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3>  nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<int,6> indexLimitsData = blockGrid.getIndexLimitsData();
    const std::array<T_data,3>  ds = blockGrid.getDs();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();

    int i,j,k,indx;
    T_data x,y,z;
    for(k=indexLimitsData[4];k<indexLimitsData[5];k++)
    {
        for(j=indexLimitsData[2];j<indexLimitsData[3];j++)
        {
            for(i=indexLimitsData[0];i<indexLimitsData[1];i++)
            {
                x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                //data[indx]= functionXYZ(x,y,z);
                data[indx]= exactSolutionAndBCs.setFieldB(x,y,z);
            }
        }
    }
}




template<int DIM, typename T_data>
void adjustFieldBForDirichletNeumanBCs(const BlockGrid<DIM,T_data>& blockGrid,ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs,T_data fieldX[],T_data fieldB[])
{
    const std::array<int,3> globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxX,indxB;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            indexLimitsBCs=indexLimitsData;
            adjustIndex={0,0,0};
            if(bcsType[2*dir]==0) // Dirichlet
            {
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir]+1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxX = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxB = i+adjustIndex[0] + nlocal_guards[0]*(j+adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k+adjustIndex[2]);
                            fieldB[indxB]-=fieldX[indxX]/(ds[dir]*ds[dir]);   
                        }
                    }
                }
            }
            else if(bcsType[2*dir]==1) //neuman
            {
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxB = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            fieldB[indxB]+= 2 * exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / ds[dir];   
                        }
                    }
                }
            }
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            indexLimitsBCs=indexLimitsData;
            adjustIndex={0,0,0};
            if(bcsType[2*dir+1]==0) // Dirichlet
            {
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1]-1;
                adjustIndex[dir]=-1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxX = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxB = i+adjustIndex[0] + nlocal_guards[0]*(j+adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k+adjustIndex[2]);
                            fieldB[indxB]-=fieldX[indxX]/(ds[dir]*ds[dir]);   
                        }
                    }
                }
            }
            else if(bcsType[2*dir+1]==1) //neuman
            {
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxB = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            fieldB[indxB]-= 2 * exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / ds[dir];   
                        }
                    }
                }
            }
        }
    }
}



template<int DIM, typename T_data>
void adjustFieldBForDirichletNeumanBCsOrder1(const BlockGrid<DIM,T_data>& blockGrid,ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs,T_data fieldX[],T_data fieldB[])
{
    const std::array<int,3> globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxX,indxB;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            indexLimitsBCs=indexLimitsData;
            adjustIndex={0,0,0};
            if(bcsType[2*dir]==0) // Dirichlet
            {
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir]+1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxX = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxB = i+adjustIndex[0] + nlocal_guards[0]*(j+adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k+adjustIndex[2]);
                            fieldB[indxB]-=fieldX[indxX]/(ds[dir]*ds[dir]);   
                        }
                    }
                }
            }
            else if(bcsType[2*dir]==1) //neuman
            {
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxB = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            fieldB[indxB]+=  exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / ds[dir];   
                        }
                    }
                }
            }
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            indexLimitsBCs=indexLimitsData;
            adjustIndex={0,0,0};
            if(bcsType[2*dir+1]==0) // Dirichlet
            {
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1]-1;
                adjustIndex[dir]=-1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxX = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxB = i+adjustIndex[0] + nlocal_guards[0]*(j+adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k+adjustIndex[2]);
                            fieldB[indxB]-=fieldX[indxX]/(ds[dir]*ds[dir]);   
                        }
                    }
                }
            }
            else if(bcsType[2*dir+1]==1) //neuman
            {
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxB = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            fieldB[indxB]-=  exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / ds[dir];   
                        }
                    }
                }
            }
        }
    }
}






template<int DIM, typename T_data>
void halveFieldBForNeumanBCs(const BlockGrid<DIM,T_data>& blockGrid,T_data fieldB[])
{
    const int ntotlocal_guards = blockGrid.getNtotLocalGuards();
    const std::array<int,3> globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> indexLimitsSolver=blockGrid.getIndexLimitsSolver();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    std::array<int,6> indexLimitsBCs=indexLimitsSolver;
    int indxX,indxB;

    T_data* fieldTmp = new T_data[ntotlocal_guards];
    std::memcpy(fieldTmp, fieldB, ntotlocal_guards* sizeof(T_data));
    
    // loop to halve the fieldB Neuman border
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {            
            if(bcsType[2*dir]==1) //neuman
            {
                indexLimitsBCs=indexLimitsSolver;
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                for(int i=0;i<DIM;i++)
                {
                    if(i!=dir && (blockGrid.getHasBoundary()[2*i] || blockGrid.getHasBoundary()[2*i+1]) )
                    {
                        if(bcsType[2*i]==0)
                        {
                            indexLimitsBCs[2*i]+=1;
                        }
                        if(bcsType[2*i+1]==0)
                        {
                            indexLimitsBCs[2*i+1]-=1;
                        }
                    }
                }
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxB = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            fieldB[indxB] = fieldTmp[indxB]/2;
                        }
                    }
                }
            }
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            if(bcsType[2*dir+1]==1) //neuman
            {
                indexLimitsBCs=indexLimitsSolver;
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                for(int i=0;i<DIM;i++)
                {
                    if(i!=dir && (blockGrid.getHasBoundary()[2*i] || blockGrid.getHasBoundary()[2*i+1]) )
                    {
                        if(bcsType[2*i]==0)
                        {
                            indexLimitsBCs[2*i]+=1;
                        }
                        if(bcsType[2*i+1]==0)
                        {
                            indexLimitsBCs[2*i+1]-=1;
                        }
                    }
                }
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            indxB = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            fieldB[indxB] = fieldTmp[indxB]/2;
                        }
                    }
                }
            }
        }
    }

    delete[] fieldTmp;
}



template<int DIM, typename T_data>
T_data normalizeProblemToFieldBNorm(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data fieldX[],T_data fieldB[])
{
    const int ntotlocal_guards = blockGrid.getNtotLocalGuards();
    T_data fieldBNorm=0;
    T_data pSumNorm=0;
    T_data* fieldBtmp = new T_data[ntotlocal_guards];

    std::memcpy(fieldBtmp, fieldB, ntotlocal_guards* sizeof(T_data));
    adjustFieldBForDirichletNeumanBCs(blockGrid,exactSolutionAndBCs,fieldX,fieldBtmp);
    adjustFieldBForDirichletNeumanBCsOrder1(blockGrid,exactSolutionAndBCs,fieldX,fieldBtmp);

    //halveFieldBForNeumanBCs(blockGrid,fieldBtmp);

    for(int i=0;i<ntotlocal_guards; i++)
    {
        pSumNorm+=fieldBtmp[i]*fieldBtmp[i];
    }
    MPI_Reduce(&pSumNorm, &fieldBNorm, 1, getMPIType<T_data>(), MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fieldBNorm, 1, getMPIType<T_data>(), 0, MPI_COMM_WORLD);
    fieldBNorm=std::sqrt(fieldBNorm);
    //fieldBNorm=1000;
    for(int i=0;i<ntotlocal_guards; i++)
    {
        fieldX[i]/=fieldBNorm;
        fieldB[i]/=fieldBNorm;
    }

    delete[] fieldBtmp;

    return fieldBNorm;
}


template<int DIM, typename T_data>
T_data normalizeProblemToFieldBNormPreconditioned(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data fieldX[],T_data fieldB[])
{
    const int ntotlocal_guards = blockGrid.getNtotLocalGuards();
    T_data fieldBNorm=0;
    T_data pSumNorm=0;
    T_data errorLocal=0;
    int localIterations;
    T_data* fieldBtmp = new T_data[ntotlocal_guards];
    T_data* fieldBtmp2 = new T_data[ntotlocal_guards];
    std::fill(fieldBtmp2, fieldBtmp2 + ntotlocal_guards, 0);

    std::memcpy(fieldBtmp, fieldB, ntotlocal_guards* sizeof(T_data));
    adjustFieldBForDirichletNeumanBCs(blockGrid,exactSolutionAndBCs,fieldX,fieldBtmp);

    
    localIterations=localBiCGSTABLoop(blockGrid,exactSolutionAndBCs,fieldBtmp2,fieldBtmp,errorLocal);
    std::swap(fieldBtmp2,fieldBtmp);

    for(int i=0;i<ntotlocal_guards; i++)
    {
        pSumNorm+=fieldBtmp[i]*fieldBtmp[i];
    }
    MPI_Reduce(&pSumNorm, &fieldBNorm, 1, getMPIType<T_data>(), MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fieldBNorm, 1, getMPIType<T_data>(), 0, MPI_COMM_WORLD);
    fieldBNorm=std::sqrt(fieldBNorm);
    //fieldBNorm=1000;
    for(int i=0;i<ntotlocal_guards; i++)
    {
        fieldX[i]/=fieldBNorm;
        fieldB[i]/=fieldBNorm;
    }

    delete[] fieldBtmp;
    delete[] fieldBtmp2;

    return fieldBNorm;
}







template<int DIM, typename T_data>
void resetNeumanBCsV0(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data field[])
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<int,3> guards = blockGrid.getGuards();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard,indxData;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            indexLimitsBCs=indexLimitsData;
            adjustIndex={0,0,0};
            if(bcsType[2*dir]==1) // Neuman
            {   
                std::cout<< "Debug in resetNeumanBCsV0 globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir <<std::endl;
                indexLimitsBCs[2*dir]-=guards[dir];
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir]+guards[dir];
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i + adjustIndex[0] - indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j + adjustIndex[1] - indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k + adjustIndex[2] - indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxGuard = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxData = i+2*adjustIndex[0] + nlocal_guards[0]*(j+2*adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k+2*adjustIndex[2]);
                            field[indxGuard]=field[indxData] - 2 * ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir);   
                        }
                    }
                }
            }         
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            indexLimitsBCs=indexLimitsData;
            adjustIndex={0,0,0};
            if(bcsType[2*dir+1]==1) // Neuman
            {
                std::cout<< "Debug in resetNeumanBCsV0 globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir+1 <<std::endl;
                indexLimitsBCs[2*dir+1]+=guards[dir];
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1]-guards[dir];
                adjustIndex[dir]=-1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i + adjustIndex[0] - indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j + adjustIndex[1] - indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k + adjustIndex[2] - indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxGuard = i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxData = i+2*adjustIndex[0] + nlocal_guards[0]*(j+2*adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k+2*adjustIndex[2]);
                            field[indxGuard]=field[indxData] + 2 * ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir);   
                        }
                    }
                }
            }
        }
    }
}


template<int DIM, typename T_data>
void resetNeumanBCs(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data field[], const T_data& normFieldB)
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard,indxData;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            if(bcsType[2*dir]==1) // Neuman
            {   
                //std::cout<< "Debug in resetNeumanBCs globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            //indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            indxData = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            field[indxGuard]=field[indxData] - 2 * ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB;   
                        }
                    }
                }
            }         
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            if(bcsType[2*dir+1]==1) // Neuman
            {
                //std::cout<< "Debug in resetNeumanBCs globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir+1 <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            //indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            indxData = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            field[indxGuard]=field[indxData] + 2 * ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB;   
                        }
                    }
                }
            }
        }
    }
}






template<int DIM, typename T_data>
void resetNeumanBCsV2(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data field[], const T_data fieldB[], const T_data& normFieldB)
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard,indxData,indxBord;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            if(bcsType[2*dir]==1) // Neuman
            {   
                //std::cout<< "Debug in resetNeumanBCs globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxBord= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            //indxData = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            //field[indxGuard]=field[indxBord] - ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB + ds[dir]*ds[dir]*fieldB[indxBord]*0.5/normFieldB ;
                            field[indxGuard]=field[indxBord] - ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB;   
                        }
                    }
                }
            }         
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            if(bcsType[2*dir+1]==1) // Neuman
            {
                //std::cout<< "Debug in resetNeumanBCs globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir+1 <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxBord= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            //indxData = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            //field[indxGuard]=field[indxData] + 2 * ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB;
                            //field[indxGuard]=field[indxBord] +  ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB + ds[dir]*ds[dir]*fieldB[indxBord]*0.5/normFieldB;
                            field[indxGuard]=field[indxBord] +  ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB;
                        }
                    }
                }
            }
        }
    }
}




template<int DIM, typename T_data>
void resetNeumanBCsOperator(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data field[])
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard,indxData;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            if(bcsType[2*dir]==1) // Neuman
            {   
                //std::cout<< "Debug in resetNeumanBCsOperator globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            //indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            indxData = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            field[indxGuard]=field[indxData];
                        }
                    }
                }
            }         
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            if(bcsType[2*dir+1]==1) // Neuman
            {
                //std::cout<< "Debug in resetNeumanBCsOperator globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir+1 <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            //indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            indxData = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            field[indxGuard]=field[indxData];   
                        }
                    }
                }
            }
        }
    }
}



template<int DIM, typename T_data>
void resetNeumanBCsOperatorV2(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, T_data field[])
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard,indxData,indxBord;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            if(bcsType[2*dir]==1) // Neuman
            {   
                //std::cout<< "Debug in resetNeumanBCsOperator globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxBord= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            //indxData = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            field[indxGuard]=field[indxBord];
                        }
                    }
                }
            }         
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            if(bcsType[2*dir+1]==1) // Neuman
            {
                //std::cout<< "Debug in resetNeumanBCsOperator globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir+1 <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxBord= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            //indxData = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            field[indxGuard]=field[indxBord];   
                        }
                    }
                }
            }
        }
    }
}





template<int DIM, typename T_data>
void adjustResVecForNeumanBCs(const BlockGrid<DIM,T_data>& blockGrid, T_data field[])
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> bcsType=blockGrid.getBcsType();
    const std::array<T_data,3> ds = blockGrid.getDs();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    std::array<int,6> indexLimitsBCs=indexLimitsData;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard,indxData,indxBord;
    T_data x,y,z;
    
    for(int dir=0; dir<DIM; dir++)
    {
        if(blockGrid.getHasBoundary()[2*dir])
        {
            if(bcsType[2*dir]==1) // Neuman
            {   
                //std::cout<< "Debug in resetNeumanBCs globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxBord= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            //indxGuard = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            //indxData = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            field[indxBord]*=2;
                        }
                    }
                }
            }         
        }
        if(blockGrid.getHasBoundary()[2*dir+1])
        {
            if(bcsType[2*dir+1]==1) // Neuman
            {
                //std::cout<< "Debug in resetNeumanBCs globalLocation "<<globalLocation[0] << " "<<globalLocation[1] << " "<<globalLocation[2] << " dir "<<2*dir+1 <<std::endl;
                indexLimitsBCs=indexLimitsData;
                adjustIndex={0,0,0};
                indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - 1;
                adjustIndex[dir]=1;
                for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
                {
                    for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
                    {
                        for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                        {
                            x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                            y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                            z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                            indxBord= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                            indxGuard = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                            //indxData = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                            //field[indxGuard]=field[indxData] + 2 * ds[dir]*exactSolutionAndBCs.trueSolutionDdir(x,y,z,dir) / normFieldB;
                            field[indxBord]*=2;
                        }
                    }
                }
            }
        }
    }
}








template<int DIM, typename T_data>
void setBCsSolverValue(const BlockGrid<DIM,T_data>& blockGrid, T_data field[], T_data value)
{
    //const std::array<int,6> indexLimitsData=blockGrid.getIndexLimitsData();
    const std::array<int,6> indexLimitsSolver=blockGrid.getIndexLimitsSolver();
    const std::array<int,3> nlocal_guards = blockGrid.getNlocalGuards();
    //const std::array<int,3> nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<int,3>  guards = blockGrid.getGuards();
    std::array<int,6> indexLimitsBCs=indexLimitsSolver;
    std::array<int,3> adjustIndex={0,0,0};
    int indxGuard;
    
    // set boundary condition for the system --> loop over guard points or bord if the block has boundary
    for(int dir=0; dir<DIM; dir++)
    {
        indexLimitsBCs=indexLimitsSolver;
        adjustIndex={0,0,0};
        indexLimitsBCs[2*dir+1]=indexLimitsBCs[2*dir] + guards[dir];
        adjustIndex[dir]=1;
        for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
        {
            for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
            {
                for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                {
                    indxGuard = i - adjustIndex[0] + nlocal_guards[0]*(j - adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k - adjustIndex[2]);
                    field[indxGuard]=value;
                }
            }
        }

        indexLimitsBCs=indexLimitsSolver;
        adjustIndex={0,0,0};
        indexLimitsBCs[2*dir]=indexLimitsBCs[2*dir+1] - guards[dir];
        adjustIndex[dir]=1;
        for(int k=indexLimitsBCs[4]; k<indexLimitsBCs[5];k++)
        {
            for(int j=indexLimitsBCs[2];j<indexLimitsBCs[3];j++)
            {
                for(int i=indexLimitsBCs[0];i<indexLimitsBCs[1];i++)
                {
                    indxGuard = i + adjustIndex[0] + nlocal_guards[0]*(j + adjustIndex[1]) + nlocal_guards[0]*nlocal_guards[1]*(k + adjustIndex[2]);
                    field[indxGuard]=value;   
                }
            }
        }
    }
        
}




template<int DIM, typename T_data>
T_data checkSolutionLocal(const BlockGrid<DIM,T_data>& blockGrid, ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs)
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,3>  nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,3>  nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<int,6> indexLimitsData = blockGrid.getIndexLimitsData();
    const std::array<T_data,3>  ds = blockGrid.getDs();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    const int ntotlocal_guards = blockGrid.getNtotLocalGuards();
    
    T_data* fieldX = blockGrid.getFieldX();
    T_data* solution = new T_data[ntotlocal_guards];
    int i,j,k,indx;
    T_data x,y,z;
    T_data errorLocal=0;
    for(k=indexLimitsData[4];k<indexLimitsData[5];k++)
    {
        for(j=indexLimitsData[2];j<indexLimitsData[3];j++)
        {
            for(i=indexLimitsData[0];i<indexLimitsData[1];i++)
            {
                x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
               // std::cout<< "x "<< x <<" y "<< y <<" z "<<z <<std::endl;
                solution[indx]=exactSolutionAndBCs.trueSolutionFxyz(x,y,z);
                errorLocal+=std::abs(fieldX[indx]-solution[indx]);
            }
        }
    }
    //std::cout<<" Debug checksolution globalLocation " << globalLocation[0] << " "<<globalLocation[1] <<" "<< globalLocation[2]<< " errorLocal "<<errorLocal << 
      //       " errorLocalAvg "<< errorLocal/blockGrid.getNtotLocalNoGuards() <<" indexLimitsData "<< indexLimitsData[0]<< " "<<indexLimitsData[1] <<" "<< indexLimitsData[2] << " " <<indexLimitsData[3] << " "<< indexLimitsData[4] << " "<< indexLimitsData[5] << std::endl;
    delete[] solution;
    return errorLocal;
}



template<int DIM, typename T_data>
void printFieldWithGuards(const BlockGrid<DIM,T_data>& blockGrid, T_data data[])
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,3>  nlocal_guards = blockGrid.getNlocalGuards();
    const int totRanks = blockGrid.getNranks()[0]*blockGrid.getNranks()[1]*blockGrid.getNranks()[2];
    int i,j,k,indx;
    MPI_Barrier(MPI_COMM_WORLD);
    for(int n=0; n<totRanks; n++)    
    {   
        MPI_Barrier(MPI_COMM_WORLD);
        if(blockGrid.getMyrank()==n)
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

template<int DIM, typename T_data>
void printFieldNoGuards(const BlockGrid<DIM,T_data>& blockGrid, T_data data[])
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,3>  nlocal_guards = blockGrid.getNlocalGuards();
    const std::array<int,6> indexLimitsData = blockGrid.getIndexLimitsData();
    
    int i,j,k,indx;
    std::cout<<"Print field no guards, globalLocation "<<globalLocation[0]<< " " <<globalLocation[1]<< " "<<globalLocation[2] <<std::endl;
    for(k=indexLimitsData[4];k<indexLimitsData[5];k++)
    {
        std::cout<<"indexZ = "<< k <<std::endl;
        for(j=indexLimitsData[2];j<indexLimitsData[3];j++)
        {
            for(i=indexLimitsData[0];i<indexLimitsData[1];i++)
            {
                indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                std::cout<< std::setw(3) << data[indx];
                std::cout<< " ";
            }
            std::cout<<std::endl;
        }
    }
}


template<int DIM, typename T_data>
void printGridNoGuards(const BlockGrid<DIM,T_data>& blockGrid)
{
    const std::array<int,3>  globalLocation = blockGrid.getGlobalLocation();
    const std::array<int,3>  nlocal_noguards = blockGrid.getNlocalNoGuards();
    const std::array<int,6> indexLimitsData = blockGrid.getIndexLimitsData();
    const std::array<T_data,3>  ds = blockGrid.getDs();
    const std::array<T_data,3>  origin = blockGrid.getOrigin();
    
    int i,j,k;
    T_data x,y,z;
    std::cout<<"Print grid no guards, globalLocation "<<globalLocation[0]<< " " <<globalLocation[1]<< " "<<globalLocation[2] <<std::endl;
    for(k=indexLimitsData[4];k<indexLimitsData[5];k++)
    {
        std::cout<<"indexZ = "<< k <<std::endl;
        for(j=indexLimitsData[2];j<indexLimitsData[3];j++)
        {
            for(i=indexLimitsData[0];i<indexLimitsData[1];i++)
            {
                x=origin[0] + (i-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];
                y=origin[1] + (j-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
                z=origin[2] + (k-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
                //indx= i + nlocal_guards[0]*j + nlocal_guards[0]*nlocal_guards[1]*k;
                std::cout<< std::setw(3) << x;
                std::cout<< " ";
            }
            std::cout<<std::endl;
        }
    }
}