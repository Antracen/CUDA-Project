#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "MemCpy.h"

#define TPB (16*20) // Threads per block

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{

    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/*
    The kernel makes every thread work on one single particle.
*/
__global__
void mover_PC_kernel(struct device_part_arrays part, struct EMfield* field, struct grid* grd, struct parameters *param, long nop, int n_sub_cycles, FPpart qom, int NiterMover) {

    // Find global thread index
    int part_index = blockIdx.x*blockDim.x + threadIdx.x;

    // Don't go outside array
    if(part_index >= nop) return;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    /*
        I removed the loop going through every particle since each thread handles one particle.
    */
    // start subcycling
    for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){
        // move each particle with new fields
            xptilde = part.x[part_index];
            yptilde = part.y[part_index];
            zptilde = part.z[part_index];
            
            // calculate the average velocity iteratively
            for(int innter=0; innter < NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part.x[part_index] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part.y[part_index] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part.z[part_index] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part.x[part_index] - grd->XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)];
                eta[0]  = part.y[part_index] - grd->YN_flat[get_idx(ix, iy-1, iz, grd->nyn, grd->nzn)];
                zeta[0] = part.z[part_index] - grd->ZN_flat[get_idx(ix, iy, iz-1, grd->nyn, grd->nzn)];
                xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part.x[part_index];
                eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part.y[part_index];
                zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part.z[part_index];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                            Exl += weight[ii][jj][kk]*field->Ex_flat[index];
                            Eyl += weight[ii][jj][kk]*field->Ey_flat[index];
                            Ezl += weight[ii][jj][kk]*field->Ez_flat[index];
                            Bxl += weight[ii][jj][kk]*field->Bxn_flat[index];
                            Byl += weight[ii][jj][kk]*field->Byn_flat[index];
                            Bzl += weight[ii][jj][kk]*field->Bzn_flat[index];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part.u[part_index] + qomdt2*Exl;
                vt= part.v[part_index] + qomdt2*Eyl;
                wt= part.w[part_index] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom; // ut, vt, Bzl
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part.x[part_index] = xptilde + uptilde*dto2;
                part.y[part_index] = yptilde + vptilde*dto2;
                part.z[part_index] = zptilde + wptilde*dto2;
                
            } // end of iteration

            // update the final position and velocity
            part.u[part_index]= 2.0*uptilde - part.u[part_index];
            part.v[part_index]= 2.0*vptilde - part.v[part_index];
            part.w[part_index]= 2.0*wptilde - part.w[part_index];
            part.x[part_index] = xptilde + uptilde*dt_sub_cycling;
            part.y[part_index] = yptilde + vptilde*dt_sub_cycling;
            part.z[part_index] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part.x[part_index] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part.x[part_index] = part.x[part_index] - grd->Lx;
                } else { // REFLECTING BC
                    part.u[part_index] = -part.u[part_index];
                    part.x[part_index] = 2*grd->Lx - part.x[part_index];
                }
            }
                                                                        
            if (part.x[part_index] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part.x[part_index] = part.x[part_index] + grd->Lx;
                } else { // REFLECTING BC
                    part.u[part_index] = -part.u[part_index];
                    part.x[part_index] = -part.x[part_index];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part.y[part_index] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part.y[part_index] = part.y[part_index] - grd->Ly;
                } else { // REFLECTING BC
                    part.v[part_index] = -part.v[part_index];
                    part.y[part_index] = 2*grd->Ly - part.y[part_index];
                }
            }
                                                                        
            if (part.y[part_index] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part.y[part_index] = part.y[part_index] + grd->Ly;
                } else { // REFLECTING BC
                    part.v[part_index] = -part.v[part_index];
                    part.y[part_index] = -part.y[part_index];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part.z[part_index] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part.z[part_index] = part.z[part_index] - grd->Lz;
                } else { // REFLECTING BC
                    part.w[part_index] = -part.w[part_index];
                    part.z[part_index] = 2*grd->Lz - part.z[part_index];
                }
            }
                                                                        
            if (part.z[part_index] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part.z[part_index] = part.z[part_index] + grd->Lz;
                } else { // REFLECTING BC
                    part.w[part_index] = -part.w[part_index];
                    part.z[part_index] = -part.z[part_index];
                }
            }
                                                                        
            
    } // end of one particle
}

/** MODIFICATION: New particle mover */ // TODO COMMENT MORE
int mover_PC(struct device_part_arrays part, EMfield *field, grid* grd, parameters *param, long nop, int species_ID, int n_sub_cycles, FPpart qom, int NiterMover)
{
    // print species and subcycling
    //std::cout << "***  MOVER with SUBCYCLYING "<< n_sub_cycles << " - species " << species_ID << " ***" << std::endl;
    int blocks = ((nop + TPB - 1) / TPB);
    mover_PC_kernel<<<blocks, TPB>>>(part, field, grd, param, nop, n_sub_cycles, qom, NiterMover);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cout << "Kernel mover_PC error: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
    return 0;
}


/** Interpolation Particle --> Grid: This is for species */
// TODO COMMENT WHEN DONE
__global__
void interpP2G_kernel(device_part_arrays part, device_ids_arrays ids, grid* grd, long nop) {

    int part_index = blockIdx.x*blockDim.x + threadIdx.x;
    if(part_index >= nop) return;

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
            
    // determine cell: can we change to int()? is it faster?
    ix = 2 + int(floor((part.x[part_index] - grd->xStart) * grd->invdx));
    iy = 2 + int(floor((part.y[part_index] - grd->yStart) * grd->invdy));
    iz = 2 + int(floor((part.z[part_index] - grd->zStart) * grd->invdz));
    
    // distances from node
    xi[0]   = part.x[part_index] - grd->XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)];
    eta[0]  = part.y[part_index] - grd->YN_flat[get_idx(ix, iy-1, iz, grd->nyn, grd->nzn)];
    zeta[0] = part.z[part_index] - grd->ZN_flat[get_idx(ix, iy, iz-1, grd->nyn, grd->nzn)];
    xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part.x[part_index];
    eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part.y[part_index];
    zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part.z[part_index];
    
    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weight[ii][jj][kk] = part.q[part_index] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
    
    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++){
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.rhon_flat[index]), weight[ii][jj][kk] * grd->invVOL);
            }
    
    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[part_index] * weight[ii][jj][kk];
    
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.Jx_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.v[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.Jy_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    
    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.w[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.Jz_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[part_index] * part.u[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.pxx_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[part_index] * part.v[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.pxy_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    
    
    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.u[part_index] * part.w[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.pxz_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    
    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.v[part_index] * part.v[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.pyy_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    
    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.v[part_index] * part.w[part_index] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.pyz_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
    
    
    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part.w[part_index] * part.w[part_index] * weight[ii][jj][kk];
    for (int ii=0; ii < 2; ii++)
        for (int jj=0; jj < 2; jj++)
            for(int kk=0; kk < 2; kk++) {
                long index = get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn);
                atomicAdd(&(ids.pzz_flat[index]), temp[ii][jj][kk] * grd->invVOL);
            }
}

// TODO COMMENT WHEN DONE
void interpP2G(struct device_part_arrays part, struct device_ids_arrays ids, grid* grd, long nop) {
    int blocks = (nop + TPB - 1) / TPB;
    interpP2G_kernel<<<blocks, TPB>>>(part, ids, grd, nop);
}
