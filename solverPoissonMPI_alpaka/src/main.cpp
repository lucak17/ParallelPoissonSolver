#include <mpi.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <array>
#include <cstring>
#include <cmath>

#include <alpaka/alpaka.hpp>

#include "inputParam.hpp"
#include "communicationMPI.hpp"
#include "blockGrid.hpp"
#include "alpakaHelper.hpp"
//#include "matrixFreeOperatorA.hpp"
//#include "operationGrid.hpp"


// Nranks, ds, guards always size 3, then exceeding dimensions are not used 

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Get the current time
    auto now = std::chrono::system_clock::now();
    // Convert to time_t (which represents system time in seconds)
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    // Convert to local time and print using std::put_time
    auto start = std::chrono::high_resolution_clock::now();

    // read number MPI processes in x y z
    std::array<int,3> nranks_tmp;
    if constexpr (DIM==3){
        nranks_tmp = {std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3])};
    }
    else if constexpr (DIM==2){
        nranks_tmp = {std::atoi(argv[1]),std::atoi(argv[2]),1};
    }
    else if constexpr (DIM==1){
        nranks_tmp = {std::atoi(argv[1]),1,1};
    }
    else
    {
        std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
        exit(-1); // Exit with failure status
    }
    // number MPI processes in each direction x y z 
    const std::array<int, 3> nranks = nranks_tmp;
    if(nranks[0]*nranks[1]*nranks[2]!=world_size)
    {
        std::cerr << "Error: configuration of MPI ranks not coherent! DIM = "<< DIM << " worldSize "<<world_size <<" ranks "<< nranks[0] <<" "<<nranks[1]<<" "<<nranks[2]<< " "<< std::endl;
        exit(-1);
    }

    
    // create object that holds grid info
    BlockGrid<DIM,T_data> blockGrid(nranks,my_rank,npglobal,ds,origin,guards,bcsType,bcsValue);
    auto nlocal_noguards = blockGrid.getNlocalNoGuards();
    auto nlocal_guards = blockGrid.getNlocalGuards();
    auto guards = blockGrid.getGuards(); 
    auto npGlobal = blockGrid.getNpglobal();

    if(my_rank==0)
    {
        std::cout<< "Current local time and date: " << std::put_time(std::localtime(&currentTime), "%Y-%m-%d %H:%M:%S") << std::endl;
        std::cout<< "Domain DIM = "<< DIM <<  " - Number of MPI tasks "<<nranks[0] << " "<< nranks[1] << " "<< nranks[2] << " - Tot MPI ranks " << nranks[0]*nranks[1]*nranks[2] << " - Max threads per MPI rank "<< 1 << " - Tot threads " << nranks[0]*nranks[1]*nranks[2] << std::endl;
        std::cout<< "Global grid size from block "<< nlocal_noguards[0] * nranks[0]<< " " << nlocal_noguards[1] * nranks[1]<< " " << nlocal_noguards[2] * nranks[2] << " - Global number of points " << nranks[0]*nranks[1]*nranks[2]*blockGrid.getNtotLocalNoGuards() << std::endl;
        std::cout<< "Domain local Np xyz no guards "<<  nlocal_noguards[0]<< " " << nlocal_noguards[1]<< " " << nlocal_noguards[2]<<  " - Domain local Np xyz guards = "<<nlocal_guards[0] << " "<< nlocal_guards[1] << " "<< nlocal_guards[2] << " - Guards size "<<guards[0] << " "<< guards[1] << " "<< guards[2]  <<std::endl;
        std::cout<< "Total local number of points noguards "<<blockGrid.getNtotLocalNoGuards()<< " - total local number of points guards "<< blockGrid.getNtotLocalGuards() <<std::endl;
        std::cout<< "Total local number of points noguards per thread "<<blockGrid.getNtotLocalNoGuards()<< " - total local number of points guards per thread "<< blockGrid.getNtotLocalGuards() <<std::endl;
        std::cout<< "Domain global origin xyz "<<  origin[0]<< " " << origin[1]<< " " << origin[2]<< " - domain global extension xyz "<<  origin[0] + (npglobal[0]-1)*ds[0]<< " " << origin[1] + (npglobal[1]-1)*ds[1]<< " " << origin[2]+ (npglobal[2]-1)*ds[2]<< " - Ds xyz  = "<<ds[0] << " "<< ds[1] << " "<< ds[2] <<std::endl;
        std::cout<< "Boundary condition type " << blockGrid.getBcsType()[0] << " " << blockGrid.getBcsType()[1] << " " << blockGrid.getBcsType()[2] << " " << blockGrid.getBcsType()[3] << " " << blockGrid.getBcsType()[4] << " " << blockGrid.getBcsType()[5]  <<std::endl;
        std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
    }


    // objects that hold communication, exact solution and BCs
    CommunicatorMPI<DIM,T_data>  communicator(blockGrid);
    ExactSolutionAndBCs<DIM,T_data> exactSolutionAndBCs;
    AlpakaHelper<DIM,T_data> alpakaHelper(blockGrid);
    
    
    // iterative solver object
    T_Solver solver(blockGrid,exactSolutionAndBCs,communicator,alpakaHelper);

    // define fieldData
    T_data* fieldX = new T_data[blockGrid.getNtotLocalGuards()];
    T_data* fieldB = new T_data[blockGrid.getNtotLocalGuards()];
    std::fill(fieldX, fieldX + blockGrid.getNtotLocalGuards(), 0);
    std::fill(fieldB, fieldB + blockGrid.getNtotLocalGuards(), 0);

    //solver.setProblem(fieldX,fieldB);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto startSolver = std::chrono::high_resolution_clock::now();
    
    // iterative solver
    solver(fieldX,fieldB);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSolver = std::chrono::high_resolution_clock::now();

    if(my_rank==0)
    {
        std::cout<<"Iterative solver finished with iter: "<< solver.getNumIterationFinal()<< " - Total preconditione iterations: "<< solver.getNumIterationPreconditionerFinal() << "\n" << 
        " error from algo "<< solver.getErrorFromIteration() << 
        " error r=b-Ax "<< solver.getErrorComputeOperator() << " errorAvgtot "<< solver.getErrorComputeOperator()/static_cast<T_data>(blockGrid.getNtotNpglobal())<< std::endl;
    }  


    // check solution correctenss
    solver.checkSolutionLocalGlobal(fieldX);
    std::cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // output residual history
    if constexpr (writeResidual)
    {
        if(my_rank==0)
        {
            solver.writeResidualHistory();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    

    // output solution
    if constexpr (writeSolution)
    {
        MPI_File fileSolution;
        MPI_Status status;
        MPI_Offset offset = sizeof(T_data) * blockGrid.getNtotLocalGuards() * my_rank ;
        
        MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fileSolution);

        MPI_File_seek(fileSolution, offset, MPI_SEEK_SET);
        MPI_File_write(fileSolution, fieldX, blockGrid.getNtotLocalGuards(), getMPIType<T_data>(), &status);
        MPI_File_close(&fileSolution);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    std::chrono::duration<double> durationSolver = endSolver - startSolver;
    std::chrono::duration<double> duration = end - start;
    if(my_rank==0)
    {
        std::cout << "Solver time: " << durationSolver.count() << " seconds" << std::endl;
        std::cout << "SolverInFunction time: " << solver.getDurationSolver().count() << " seconds" << " - average time cycle " << solver.getDurationSolver().count()/solver.getNumIterationFinal() << " seconds" << std::endl;
        solver.timeCounter.printAverageTime(solver.getNumIterationFinal());
        std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
        std::cout<<  "End program. "<< std::endl;
    }

    delete[] fieldX;
    delete[] fieldB;
    
    // Finalize MPI
    MPI_Finalize();

    return 0;
}