#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

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


/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        // for (int ii = 0; ii < 2; ii++)
        //     for (int jj = 0; jj < 2; jj++)
        //         for (int kk = 0; kk < 2; kk++)
        //             temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        // for (int ii=0; ii < 2; ii++)
        //     for (int jj=0; jj < 2; jj++)
        //         for(int kk=0; kk < 2; kk++)
        //             ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
