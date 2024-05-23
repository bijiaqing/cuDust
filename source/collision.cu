#include "cudust.cuh"
#include "cukd/knn.h"   // for cukd::cct::knn, cukd::HeapCandidateList
#include "curand_kernel.h"

// =========================================================================================================================

__global__ 
void rngstate_init(curandState *dev_rngstate, int seed)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        curand_init(seed, idx, 0, &dev_rngstate[idx]);
    }
}

// =========================================================================================================================

__global__
void treenode_init (swarm *dev_particle, tree *dev_treenode)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        // float azi = static_cast<float>(dev_particle[idx].position.x);
        // float rad = static_cast<float>(dev_particle[idx].position.y);
        // float col = static_cast<float>(dev_particle[idx].position.z);

        // dev_treenode[idx].xyz.x = rad*sin(col)*cos(azi);
        // dev_treenode[idx].xyz.y = rad*sin(col)*sin(azi);
        // dev_treenode[idx].xyz.z = rad*cos(col);
        // dev_treenode[idx].index_old = idx;

        dev_treenode[idx].xyz.x = static_cast<float>(dev_particle[idx].position.x);
        dev_treenode[idx].xyz.y = static_cast<float>(dev_particle[idx].position.y);
        dev_treenode[idx].xyz.z = static_cast<float>(dev_particle[idx].position.z);
        dev_treenode[idx].index_old = idx;
    }
}

// =========================================================================================================================

__device__
real get_distance (swarm *dev_particle, int idx_1, int idx_2)
{
    real azi_1, rad_1, col_1;
    real azi_2, rad_2, col_2;

    azi_1  = dev_particle[idx_1].position.x;
    rad_1  = dev_particle[idx_1].position.y;
    col_1  = dev_particle[idx_1].position.z;

    azi_2  = dev_particle[idx_2].position.x;
    rad_2  = dev_particle[idx_2].position.y;
    col_2  = dev_particle[idx_2].position.z;

    return sqrt(rad_1*rad_1 + rad_2*rad_2 - 2.0*rad_1*rad_2*(sin(col_1)*sin(col_2)*cos(azi_1 - azi_2) + cos(col_1)*cos(col_2)));
}

__device__
real get_collrate (swarm *dev_particle, int idx_1, int idx_2, real dist_max)
{
    real azi, rad, col, v_azi, v_rad, v_col;
    real vx_1, vy_1, vz_1, size_1, mass_1;
    real vx_2, vy_2, vz_2, size_2, mass_2;
    real delta_v, lambda;

    azi    = dev_particle[idx_1].position.x;
    rad    = dev_particle[idx_1].position.y;
    col    = dev_particle[idx_1].position.z;
    v_azi  = dev_particle[idx_1].velocity.x / rad / sin(col);
    v_rad  = dev_particle[idx_1].velocity.y;
    v_col  = dev_particle[idx_1].velocity.z / rad;
    size_1 = dev_particle[idx_1].dustsize;
    mass_1 = dev_particle[idx_1].numgrain*size_1*size_1*size_1;

    vx_1 = v_rad*sin(col)*cos(azi) + v_col*cos(col)*cos(azi) - v_azi*sin(azi);
    vy_1 = v_rad*sin(col)*sin(azi) + v_col*cos(col)*sin(azi) + v_azi*cos(azi);
    vz_1 = v_rad*cos(col)          + v_col*sin(col);

    azi    = dev_particle[idx_2].position.x;
    rad    = dev_particle[idx_2].position.y;
    col    = dev_particle[idx_2].position.z;
    v_azi  = dev_particle[idx_2].velocity.x / rad / sin(col);
    v_rad  = dev_particle[idx_2].velocity.y;
    v_col  = dev_particle[idx_2].velocity.z / rad;
    size_2 = dev_particle[idx_2].dustsize;
    mass_2 = dev_particle[idx_2].numgrain*size_2*size_2*size_2;

    vx_2 = v_rad*sin(col)*cos(azi) + v_col*cos(col)*cos(azi) - v_azi*sin(azi);
    vy_2 = v_rad*sin(col)*sin(azi) + v_col*cos(col)*sin(azi) + v_azi*cos(azi);
    vz_2 = v_rad*cos(col)          + v_col*sin(col);

    delta_v = sqrt((vx_1 - vx_2)*(vx_1 - vx_2) + (vy_1 - vy_2)*(vy_1 - vy_2) + (vz_1 - vz_2)*(vz_1 - vz_2));
    
    lambda  = (size_1 + size_2)*(size_1 + size_2)*delta_v / dist_max / dist_max / dist_max; // the choice of V is not very physical...
    lambda *= mass_2 / size_2 / size_2 / size_2;
    
    if (mass_2 < 0.1*mass_1)
    {
        lambda /= 0.1*mass_1 / mass_2;
    }
    
    return lambda;
}

__global__
void collrate_calc (swarm *dev_particle, swarm_tmp *dev_tmp_info, tree *dev_treenode, int *dev_collrate, 
    const cukd::box_t<float3> *dev_boundbox)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        // float max_query_dist = 0.05;    // change to gas scale height if needed

        using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;
        candidatelist query_result(static_cast<float>(MAX_QUERY_DIST));
        cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_treenode[idx].xyz, *dev_boundbox, dev_treenode, NUM_PAR);

        real collrate_ij = 0.0; // collision rate between particle i and j
        real collrate_i  = 0.0; // total collision rate for particle i

        int tmp_idx   = query_result.returnIndex(0); // this is always the farthest one, if exists
        int idx_old_1 = dev_treenode[idx].index_old;
        int idx_old_2;
        real dist_max = MAX_QUERY_DIST;

        if (tmp_idx != -1)
        {
            idx_old_2 = dev_treenode[tmp_idx].index_old;
            dist_max = get_distance(dev_particle, idx_old_1, idx_old_2);
        }

        for(int j = 0; j < KNN_SIZE; j++)
        {
            collrate_ij = 0.0;
            tmp_idx  = query_result.returnIndex(j);

            if (tmp_idx != -1)
            {
                idx_old_2 = dev_treenode[tmp_idx].index_old;
                // collrate_ij = COLLRATE_NORM*get_collrate(dev_particle, idx_old_1, idx_old_2, dist_max);
                collrate_ij = 1.0;
            }

            collrate_i += collrate_ij;
        }
        
        // if (idx < 200) printf("%.8e\n", collrate_i);
        
        dev_tmp_info[idx_old_1].collrate_i = collrate_i;
        atomicMax(dev_collrate, static_cast<int>(collrate_i));
    }
}

// =========================================================================================================================

__global__
void dustcoag_calc (swarm *dev_particle, swarm_tmp *dev_tmp_info, tree *dev_treenode, real *dev_timestep, 
    const cukd::box_t<float3> *dev_boundbox, curandState *dev_rngstate)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        int idx_old_1 = dev_treenode[idx].index_old;

        real rand_collide_i = curand_uniform_double(&dev_rngstate[idx_old_1]); // (0,1]
        real real_collide_i = (*dev_timestep)*dev_tmp_info[idx_old_1].collrate_i;

        if (real_collide_i >= rand_collide_i)   // means a collision happens 
        {
            using candidatelist = cukd::HeapCandidateList<KNN_SIZE>;
            candidatelist query_result(static_cast<float>(MAX_QUERY_DIST));
            cukd::cct::knn <candidatelist, tree, tree_traits> (query_result, dev_treenode[idx].xyz, *dev_boundbox, dev_treenode, NUM_PAR);

            int tmp_idx = query_result.returnIndex(0);
            int idx_old_2;
            real dist_max = MAX_QUERY_DIST;

            // if (tmp_idx != -1)
            // {
            //     idx_old_2 = dev_treenode[tmp_idx].index_old;
            //     dist_max = get_distance(dev_particle, idx_old_1, idx_old_2);
            // }

            real rand_collide_ij = real_collide_i*curand_uniform_double(&dev_rngstate[idx_old_1]);
            real real_collide_ij = 0.0;

            int idx_knn = 0;

            while (real_collide_ij < rand_collide_ij && idx_knn < KNN_SIZE)
            {
                tmp_idx = query_result.returnIndex(idx_knn);

                if (tmp_idx != -1)
                {
                    idx_old_2 = dev_treenode[tmp_idx].index_old;
                    // real_collide_ij += COLLRATE_NORM*get_collrate(dev_particle, idx_old_1, idx_old_2, dist_max);
                    real_collide_ij += 1.0;
                }

                idx_knn++;
            }

            // collide with idx_old_2 and merge
            real size_1 = dev_particle[idx_old_1].dustsize;
            real size_2 = dev_particle[idx_old_2].dustsize;
            real size_3 = cbrt(size_1*size_1*size_1 + size_2*size_2*size_2);

            dev_tmp_info[idx_old_1].dustsize = size_3;
            dev_tmp_info[idx_old_1].numgrain = dev_particle[idx_old_1].numgrain*(size_1/size_3)*(size_1/size_3)*(size_1/size_3);
        }
    }
}

// =========================================================================================================================

__global__
void dustsize_updt (swarm *dev_particle, swarm_tmp *dev_tmp_info)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= 0 && idx < NUM_PAR)
    {
        dev_particle[idx].dustsize = dev_tmp_info[idx].dustsize;
        dev_particle[idx].numgrain = dev_tmp_info[idx].numgrain;
    }
}