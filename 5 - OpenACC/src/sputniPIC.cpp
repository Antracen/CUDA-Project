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



int mover_PC(struct particles* part, int is, struct EMfield* field, struct grid* grd, struct parameters* param);

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
    
    size_t grid_size = (grd.nxn*grd.nyn*grd.nzn);

    #pragma acc enter data copyin(grd)
    #pragma acc enter data copyin(grd.XN_flat[0:grid_size], grd.YN_flat[0:grid_size], grd.ZN_flat[0:grid_size])
    
    #pragma acc enter data copyin(field)
    #pragma acc enter data copyin(field.Ex_flat[0:grid_size], field.Ey_flat[0:grid_size], field.Ez_flat[0:grid_size], field.Bxn_flat[0:grid_size], field.Byn_flat[0:grid_size], field.Bzn_flat[0:grid_size])

    #pragma acc enter data copyin(param)
    

    int num_species = param.ns;
    #pragma acc enter data copyin(part[0:num_species])

    for(int is = 0; is < num_species; is++) {
        long npmax = part[is].npmax;
        #pragma acc enter data create(part[is].x[0:npmax], part[is].y[0:npmax], part[is].z[0:npmax], part[is].u[0:npmax], part[is].v[0:npmax], part[is].w[0:npmax], part[is].q[0:npmax])
    }


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
        for (int is=0; is < param.ns; is++) {
            int nop = part[is].npmax;
            #pragma acc update device(part[is].x[0:nop], part[is].y[0:nop], part[is].z[0:nop], part[is].u[0:nop], part[is].v[0:nop], part[is].w[0:nop], part[is].q[0:nop])
            mover_PC(part,is,&field,&grd,&param);
            #pragma acc update host(part[is].x[0:nop], part[is].y[0:nop], part[is].z[0:nop], part[is].u[0:nop], part[is].v[0:nop], part[is].w[0:nop], part[is].q[0:nop])
        }
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++) {
            interpP2G(&part[is],&ids[is],&grd);
        }
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

    #pragma acc exit data delete(param)

    #pragma acc exit data delete(grd.XN_flat[0:grid_size], grd.YN_flat[0:grid_size], grd.ZN_flat[0:grid_size])
    #pragma acc exit data delete(grd)

    #pragma acc exit data delete(field.Ex_flat[0:grid_size], field.Ey_flat[0:grid_size], field.Ez_flat[0:grid_size], field.Bxn_flat[0:grid_size], field.Byn_flat[0:grid_size], field.Bzn_flat[0:grid_size])
    #pragma acc exit data delete(field)

    for(int is = 0; is < num_species; is++) {
        long npmax = part[is].npmax;
        #pragma acc exit data delete(part[is].x[0:npmax], part[is].y[0:npmax], part[is].z[0:npmax], part[is].u[0:npmax], part[is].v[0:npmax], part[is].w[0:npmax], part[is].q[0:npmax])
    }

    #pragma acc exit data delete(part[0:num_species])
    
    // stop timeri
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}



/** particle mover */
int mover_PC(struct particles* part, int is, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part[is].species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part[is].n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part[is].qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    int i;
    int i_sub;
    int innter;
    int ii;
    int jj;
    int kk;
    int ii2;
    int jj2;
    int kk2;
    
    size_t grid_size = (grd->nxn*grd->nyn*grd->nzn);
    
    // start subcycling
    // move each particle with new fields
    #pragma acc parallel loop \
                present(grd, field, part[0:is], param) \
                private(i_sub, innter, ii, jj, kk, xptilde, yptilde, zptilde, uptilde, vptilde, wptilde, ix, iy, iz, weight[0:2][0:2][0:2], xi[0:2], eta[0:2], zeta[0:2], Exl, Eyl, Ezl, Bxl, Byl, Bzl)
    for (i=0; i <  part[is].nop; i++){
        for (i_sub=0; i_sub <  part[is].n_sub_cycles; i_sub++){
            xptilde = part[is].x[i];
            yptilde = part[is].y[i];
            zptilde = part[is].z[i];
            // calculate the average velocity iteratively
            for(innter=0; innter < part[is].NiterMover; innter++){
            //     // interpolation G-->P
                ix = 2 +  int((part[is].x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part[is].y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part[is].z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part[is].x[i] - grd->XN_flat[(ix - 1)*grd->nyn*grd->nzn + iy*grd->nzn + iz];
                eta[0]  = part[is].y[i] - grd->XN_flat[ix*grd->nyn*grd->nzn + (iy+1)*grd->nzn + iz];
                zeta[0] = part[is].z[i] - grd->ZN_flat[ix*grd->nyn*grd->nzn + iy*grd->nzn + (iz+1)];
                xi[1]   = grd->XN_flat[ix*grd->nyn*grd->nzn + iy*grd->nzn + iz] - part[is].x[i];
                eta[1]  = grd->YN_flat[ix*grd->nyn*grd->nzn + iy*grd->nzn + iz] - part[is].y[i];
                zeta[1] = grd->ZN_flat[ix*grd->nyn*grd->nzn + iy*grd->nzn + iz] - part[is].z[i];
                for (ii = 0; ii < 2; ii++)
                    for (jj = 0; jj < 2; jj++)
                        for (kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (ii=0; ii < 2; ii++)
                    for (jj=0; jj < 2; jj++)
                        for(kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex_flat[(ix- ii)*grd->nyn*grd->nzn + (iy -jj)*grd->nzn + (iz- kk)];
                            Eyl += weight[ii][jj][kk]*field->Ey_flat[(ix- ii)*grd->nyn*grd->nzn + (iy -jj)*grd->nzn + (iz- kk)];
                            Ezl += weight[ii][jj][kk]*field->Ez_flat[(ix- ii)*grd->nyn*grd->nzn + (iy -jj)*grd->nzn + (iz -kk)];
                            Bxl += weight[ii][jj][kk]*field->Bxn_flat[(ix- ii)*grd->nyn*grd->nzn + (iy -jj)*grd->nzn + (iz -kk)];
                            Byl += weight[ii][jj][kk]*field->Byn_flat[(ix- ii)*grd->nyn*grd->nzn + (iy -jj)*grd->nzn + (iz -kk)];
                            Bzl += weight[ii][jj][kk]*field->Bzn_flat[(ix- ii)*grd->nyn*grd->nzn + (iy -jj)*grd->nzn + (iz -kk)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part[is].u[i] + qomdt2*Exl;
                vt= part[is].v[i] + qomdt2*Eyl;
                wt= part[is].w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part[is].x[i] = xptilde + uptilde*dto2;
                part[is].y[i] = yptilde + vptilde*dto2;
                part[is].z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part[is].u[i]= 2.0*uptilde - part[is].u[i];
            part[is].v[i]= 2.0*vptilde - part[is].v[i];
            part[is].w[i]= 2.0*wptilde - part[is].w[i];
            part[is].x[i] = xptilde + uptilde*dt_sub_cycling;
            part[is].y[i] = yptilde + vptilde*dt_sub_cycling;
            part[is].z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            ////////
            ////////
            //////// BC
                                        
            // X-DIRECTION: BC particles
            if (part[is].x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part[is].x[i] = part[is].x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part[is].u[i] = -part[is].u[i];
                    part[is].x[i] = 2*grd->Lx - part[is].x[i];
                }
            }
                                                                        
            if (part[is].x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part[is].x[i] = part[is].x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part[is].u[i] = -part[is].u[i];
                    part[is].x[i] = -part[is].x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part[is].y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part[is].y[i] = part[is].y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part[is].v[i] = -part[is].v[i];
                    part[is].y[i] = 2*grd->Ly - part[is].y[i];
                }
            }
                                                                        
            if (part[is].y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part[is].y[i] = part[is].y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part[is].v[i] = -part[is].v[i];
                    part[is].y[i] = -part[is].y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part[is].z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part[is].z[i] = part[is].z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part[is].w[i] = -part[is].w[i];
                    part[is].z[i] = 2*grd->Lz - part[is].z[i];
                }
            }
                                                                        
            if (part[is].z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part[is].z[i] = part[is].z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part[is].w[i] = -part[is].w[i];
                    part[is].z[i] = -part[is].z[i];
                }
            }
                                                                    
        }  // end of subcycling
    } // end of one particle

    return(0); // exit succcesfully
}
