#ifndef MEMCPY_H
#define MEMCPY_H

#include "Particles.h"
#include "EMfield.h"
#include "Grid.h"
#include <vector>

using std::vector;

/*
    The struct "device_part_arrays" is a modified version of the struct "particles".
    It contains only the array pointers. These pointers will point to device memory.
*/
struct device_part_arrays {
    FPpart *x;
    FPpart *y;
    FPpart *z;
    FPpart *u;
    FPpart *v;
    FPpart *w;
    FPpart *q;
};

/*
    The struct "device_ids_arrays" is a modified version of the struct "interpDensSpecies".
    It contains only the members which are needed on the device.
*/
struct device_ids_arrays {
    FPinterp *rhon_flat;
    FPinterp *Jx_flat;
    FPinterp *Jy_flat;
    FPinterp *Jz_flat;
    FPinterp *pxx_flat;
    FPinterp *pxy_flat;
    FPinterp *pxz_flat;
    FPinterp *pyy_flat;
    FPinterp *pyz_flat;
    FPinterp *pzz_flat;
};

// Functions for malloc on device and transfer host <-> device
void device_grd_malloc_and_transfer(struct grid*, struct grid**, size_t, vector<void*>&);
void device_field_malloc_and_transfer(struct EMfield*, struct EMfield**, size_t, vector<void*>&);
void device_param_malloc_and_transfer(struct parameters*, struct parameters**, vector<void*>&);
void device_part_malloc(struct device_part_arrays&, size_t, size_t, size_t, vector<void*>&);
void device_part_transfer(struct particles&, struct device_part_arrays&, int, long, long, cudaMemcpyKind);
void device_ids_malloc(device_ids_arrays*, size_t, size_t, vector<void*>&);
void device_ids_transfer(interpDensSpecies*, device_ids_arrays*, size_t, size_t, cudaMemcpyKind);
#endif