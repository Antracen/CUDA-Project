#include "MemCpy.h"
#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"
#include "helper_cuda.h"

/*
    Allocate memory for grid on device and transfer the grid "grd" to the device.
    After allocation, "device_grd" points to the grid on the device.
*/
void device_grd_malloc_and_transfer(grid *grd, grid **device_grd, size_t grid_size, vector<void*> &pointers_to_free) {

    // Allocate memory for grid on device.
    checkCudaErrors(cudaMalloc(device_grd, sizeof(grid)));

    // Backup host grid array pointers
    FPfield *XN_flat = grd->XN_flat;
    FPfield *YN_flat = grd->YN_flat;
    FPfield *ZN_flat = grd->ZN_flat;

    // Allocate memory for the dynamic arrays on the device. Save in grid member variables.
    checkCudaErrors(cudaMalloc(&(grd->XN_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(grd->YN_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(grd->ZN_flat), grid_size*sizeof(FPfield)));

    // Transfer dynamic arrays to device.
    checkCudaErrors(cudaMemcpy(grd->XN_flat, XN_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(grd->YN_flat, YN_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(grd->ZN_flat, ZN_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));

    // Transfer grid to device.
    checkCudaErrors(cudaMemcpy(*device_grd, grd, sizeof(grid), cudaMemcpyHostToDevice));

    // Store which pointers need to be freed.
    pointers_to_free.push_back(*device_grd);
    pointers_to_free.push_back(grd->XN_flat);
    pointers_to_free.push_back(grd->YN_flat);
    pointers_to_free.push_back(grd->ZN_flat);

    // Restore host grid array pointers
    grd->XN_flat = XN_flat;
    grd->YN_flat = YN_flat;
    grd->ZN_flat = ZN_flat;
}

/*
    Allocate memory for field on device and transfer field "field" from host to device.
    After allocation, "device_field" points to the field on the device.
*/
void device_field_malloc_and_transfer(EMfield *field, EMfield **device_field, size_t grid_size, vector<void*> &pointers_to_free) {

    // Allocate memory for field on device.
    checkCudaErrors(cudaMalloc(device_field, sizeof(EMfield)));


    // Backup host pointers
    FPfield *Ex_flat = field->Ex_flat;
    FPfield *Ey_flat = field->Ey_flat;
    FPfield *Ez_flat = field->Ez_flat;
    FPfield *Bxn_flat = field->Bxn_flat;
    FPfield *Byn_flat = field->Byn_flat;
    FPfield *Bzn_flat = field->Bzn_flat;

    // Allocate memory for the dynamic arrays on the device.
    checkCudaErrors(cudaMalloc(&(field->Ex_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(field->Ey_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(field->Ez_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(field->Bxn_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(field->Byn_flat), grid_size*sizeof(FPfield)));
    checkCudaErrors(cudaMalloc(&(field->Bzn_flat), grid_size*sizeof(FPfield)));

    // Transfer dynamic arrays to device.
    checkCudaErrors(cudaMemcpy(field->Ex_flat, Ex_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(field->Ey_flat, Ey_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(field->Ez_flat, Ez_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(field->Bxn_flat, Bxn_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(field->Byn_flat, Byn_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(field->Bzn_flat, Bzn_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice));

    // Transfer field to device.
    checkCudaErrors(cudaMemcpy(*device_field, field, sizeof(EMfield), cudaMemcpyHostToDevice));

    // Store which pointers need to be freed
    pointers_to_free.push_back(*device_field);
    pointers_to_free.push_back(field->Ex_flat);
    pointers_to_free.push_back(field->Ey_flat);
    pointers_to_free.push_back(field->Ez_flat);
    pointers_to_free.push_back(field->Bxn_flat);
    pointers_to_free.push_back(field->Byn_flat);
    pointers_to_free.push_back(field->Bzn_flat);

    // Restore host pointers
    field->Ex_flat = Ex_flat;
    field->Ey_flat = Ey_flat;
    field->Ez_flat = Ez_flat;
    field->Bxn_flat = Bxn_flat;
    field->Byn_flat = Byn_flat;
    field->Bzn_flat = Bzn_flat;
    
}

/*
    Allocate memory for parameters on device and transfer parameters "param" from host to device.
    After allocation, "device_param" points to the field on the device.
*/
void device_param_malloc_and_transfer(parameters *param, parameters **device_param, vector<void*> &pointers_to_free) {

    // Allocate memory for parameters on device
    checkCudaErrors(cudaMalloc(device_param, sizeof(parameters)));


    // Transfer parameters to device
    cudaMemcpy(*device_param, param, sizeof(parameters), cudaMemcpyHostToDevice);

    pointers_to_free.push_back(*device_param);
}

/*  TODO COMMENT WHEN DONE
    Allocate memory for the species on the device.
    Memory for the species particles are also allocated.
    After allocation "device_part_pointers" point to the species on the device
    "host_part_pointers" also point to the species but it is used by the host in memory copying

    The function allocates memory for the particles part[start] to part[end-1]
*/
void device_part_malloc(struct device_part_arrays &device_part, size_t batch_size, size_t num_species, size_t total_particles, vector<void*> &pointers_to_free) {
 
    long alloc_size;
    if(total_particles > batch_size) alloc_size = batch_size;
    else alloc_size = total_particles;

    std::cout << "Allocated space for " << alloc_size << " particles on device" << std::endl;

    checkCudaErrors(cudaMalloc(&(device_part.x), alloc_size*sizeof(FPpart)));
    checkCudaErrors(cudaMalloc(&(device_part.y), alloc_size*sizeof(FPpart)));
    checkCudaErrors(cudaMalloc(&(device_part.z), alloc_size*sizeof(FPpart)));
    checkCudaErrors(cudaMalloc(&(device_part.u), alloc_size*sizeof(FPpart)));
    checkCudaErrors(cudaMalloc(&(device_part.v), alloc_size*sizeof(FPpart)));
    checkCudaErrors(cudaMalloc(&(device_part.w), alloc_size*sizeof(FPpart)));
    checkCudaErrors(cudaMalloc(&(device_part.q), alloc_size*sizeof(FPinterp)));

    pointers_to_free.push_back(device_part.x);
    pointers_to_free.push_back(device_part.y);
    pointers_to_free.push_back(device_part.z);
    pointers_to_free.push_back(device_part.u);
    pointers_to_free.push_back(device_part.v);
    pointers_to_free.push_back(device_part.w);
    pointers_to_free.push_back(device_part.q);
}

// TODO COMMENT WHEN DONE
void device_part_transfer(struct particles &part, struct device_part_arrays &device_part, int is, long particle_at, long transfer_size, cudaMemcpyKind direction) {

    long npmax;
    if(direction == cudaMemcpyHostToDevice) {
        checkCudaErrors(cudaMemcpy(device_part.x, &(part.x[particle_at]), transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(device_part.y, &(part.y[particle_at]), transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(device_part.z, &(part.z[particle_at]), transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(device_part.u, &(part.u[particle_at]), transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(device_part.v, &(part.v[particle_at]), transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(device_part.w, &(part.w[particle_at]), transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(device_part.q, &(part.q[particle_at]), transfer_size*sizeof(FPinterp), direction));
    } else {
        checkCudaErrors(cudaMemcpy(&(part.x[particle_at]), device_part.x, transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(&(part.y[particle_at]), device_part.y, transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(&(part.z[particle_at]), device_part.z, transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(&(part.u[particle_at]), device_part.u, transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(&(part.v[particle_at]), device_part.v, transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(&(part.w[particle_at]), device_part.w, transfer_size*sizeof(FPpart), direction));
        checkCudaErrors(cudaMemcpy(&(part.q[particle_at]), device_part.q, transfer_size*sizeof(FPinterp), direction));
    }
}

// TODO COMMENT WHEN DONE
void device_ids_malloc(struct device_ids_arrays *device_ids, size_t grid_size, size_t num_species, vector<void*> &pointers_to_free) {

    for(size_t i = 0; i < num_species; i++) {
        // Allocate memory for the dynamic arrays on the device.
        checkCudaErrors(cudaMalloc(&(device_ids[i].rhon_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].Jx_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].Jy_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].Jz_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].pxx_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].pxy_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].pxz_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].pyy_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].pyz_flat), grid_size*sizeof(FPinterp)));
        checkCudaErrors(cudaMalloc(&(device_ids[i].pzz_flat), grid_size*sizeof(FPinterp)));

        pointers_to_free.push_back(device_ids[i].rhon_flat);
        pointers_to_free.push_back(device_ids[i].Jx_flat);
        pointers_to_free.push_back(device_ids[i].Jy_flat);
        pointers_to_free.push_back(device_ids[i].Jz_flat);
        pointers_to_free.push_back(device_ids[i].pxx_flat);
        pointers_to_free.push_back(device_ids[i].pxy_flat);
        pointers_to_free.push_back(device_ids[i].pxz_flat);
        pointers_to_free.push_back(device_ids[i].pyy_flat);
        pointers_to_free.push_back(device_ids[i].pyz_flat);
        pointers_to_free.push_back(device_ids[i].pzz_flat);
    }
}

// TODO COMMENT WHEN DONE
void device_ids_transfer(interpDensSpecies *ids, struct device_ids_arrays *host_ids_pointers, size_t grid_size, size_t num_species, cudaMemcpyKind direction) {

    if(direction == cudaMemcpyHostToDevice) {
        for(size_t i = 0; i < num_species; i++) {
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].rhon_flat, ids[i].rhon_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].Jx_flat, ids[i].Jx_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].Jy_flat, ids[i].Jy_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].Jz_flat, ids[i].Jz_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].pxx_flat, ids[i].pxx_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].pxy_flat, ids[i].pxy_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].pxz_flat, ids[i].pxz_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].pyy_flat, ids[i].pyy_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].pyz_flat, ids[i].pyz_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(host_ids_pointers[i].pzz_flat, ids[i].pzz_flat, grid_size*sizeof(FPinterp), direction));
        }
    } else {
        for(size_t i = 0; i < num_species; i++) {
            checkCudaErrors(cudaMemcpy(ids[i].rhon_flat, host_ids_pointers[i].rhon_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].Jx_flat, host_ids_pointers[i].Jx_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].Jy_flat, host_ids_pointers[i].Jy_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].Jz_flat, host_ids_pointers[i].Jz_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].pxx_flat, host_ids_pointers[i].pxx_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].pxy_flat, host_ids_pointers[i].pxy_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].pxz_flat, host_ids_pointers[i].pxz_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].pyy_flat, host_ids_pointers[i].pyy_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].pyz_flat, host_ids_pointers[i].pyz_flat, grid_size*sizeof(FPinterp), direction));
            checkCudaErrors(cudaMemcpy(ids[i].pzz_flat, host_ids_pointers[i].pzz_flat, grid_size*sizeof(FPinterp), direction));
        }
    }
}