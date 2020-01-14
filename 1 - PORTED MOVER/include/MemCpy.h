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

// Functions for malloc on device and transfer host <-> device
void device_grd_malloc_and_transfer(struct grid*, struct grid**, size_t, vector<void*>&);
void device_field_malloc_and_transfer(struct EMfield*, struct EMfield**, size_t, vector<void*>&);
void device_param_malloc_and_transfer(struct parameters*, struct parameters**, vector<void*>&);
void device_part_malloc(struct particles*, struct device_part_arrays*, size_t, vector<void*>&);
void device_part_transfer(struct particles*, struct device_part_arrays*, size_t, cudaMemcpyKind);

#endif