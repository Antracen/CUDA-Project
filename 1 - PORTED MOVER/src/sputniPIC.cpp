/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include "MemCpy.h" // Library for transferring data between host and device

int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    /* MODIFICATION:
        The following indented lines are used to allocate memory and transfer
        data to device.
    */

        // Grid size is used in multiple points of the code.
        size_t grid_size = (grd.nxn*grd.nyn*grd.nzn);
        int num_species = param.ns;

        // Grid malloc data
        grid *device_grd;             // Located on device. Stores device array pointers.

        // Field malloc data
        EMfield *device_field;          // Located on device. Stores device array pointers.

        parameters *device_param;

        // Particle malloc data
        device_part_arrays device_part[param.ns];              // Located on device. Stores device array pointers.

        /*
            The vector "pointers_to_free" stores device pointers
            which must be freed at the end of the program
            to avoid memory leaks.
        */
        vector<void*> pointers_to_free;

        // Malloc all variables on device.
        device_grd_malloc_and_transfer(&grd, &device_grd, grid_size, pointers_to_free);
        device_field_malloc_and_transfer(&field, &device_field, grid_size, pointers_to_free);
        device_param_malloc_and_transfer(&param, &device_param, pointers_to_free);
        device_part_malloc(part, device_part, param.ns, pointers_to_free);

        device_part_transfer(part, device_part, num_species, cudaMemcpyHostToDevice);

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
    
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);        



        // implicit mover
        iMover = cpuSecond(); // start timer for mover        
        for (int is=0; is < param.ns; is++)
            mover_PC(device_part[is], device_field, device_grd, device_param, part[is].nop, part[is].species_ID, param.n_sub_cycles, part[is].qom, part[is].NiterMover); // MODIFICATION: Function call has been modified.
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        // Transfer back particles for interpolation
        device_part_transfer(part, device_part, num_species, cudaMemcpyDeviceToHost);

        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);



        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation  
    


    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    /* DEALLOCATE DEVICE MEMORY */
        for(auto &p : pointers_to_free)
            cudaFree(p);

    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;

    cudaDeviceReset(); // Must have to detect leaks

    // exit
    return 0;
}


