// #define DEBUG

#include <nppdefs.h>
#include <omp.h>
#include <pthread.h>
#include <sm_60_atomic_functions.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef DEBUG
#include <chrono>
#define __debug_printf(fmt, args...) printf(fmt, ##args);
#define __START_TIME(ID) auto start_##ID = std::chrono::high_resolution_clock::now();
#define __END_TIME(ID)                                                                                         \
    auto stop_##ID = std::chrono::high_resolution_clock::now();                                                \
    int duration_##ID = std::chrono::duration_cast<std::chrono::milliseconds>(stop_##ID - start_##ID).count(); \
    __debug_printf("duration of %s: %d milliseconds\n", #ID, duration_##ID);
#define CUDA_CALL(F)                                                          \
    if ((F != cudaSuccess)) {                                                 \
        printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
               __FILE__, __LINE__);                                           \
        exit(-1);                                                             \
    }
#define CUDA_CHECK()                                                          \
    if ((cudaPeekAtLastError()) != cudaSuccess) {                             \
        printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
               __FILE__, __LINE__ - 1);                                       \
        exit(-1);                                                             \
    }
#else
#define __debug_printf(fmt, args...)
#define __START_TIME(ID)
#define __END_TIME(ID)
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double sq_eps = eps * eps;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
__device__ double gravity_device_mass_gpu(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double sq_planet_radius = planet_radius * planet_radius;
const double missile_speed = 1e6;
double get_missile_dist(int step) { return (missile_speed * missile_speed * dt * dt) * (step * step); }
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
__device__ double get_missile_cost_gpu(double t) { return 1e5 + 1e3 * t; }

const int n_sync_steps = 2000;
const int threads_per_block = 512;
const int cuda_nstreams = 3;
}  // namespace param

void read_input(const char* filename, int& n, double*& qxyz, double*& vxyz, double*& m, int& device_cnt, int*& device_id) {
    std::ifstream fin(filename);
    int planet, asteroid;
    fin >> n >> planet >> asteroid;

    std::string type;
    std::vector<double> tmp_qxyz(3 * n);
    std::vector<double> tmp_vxyz(3 * n);
    std::vector<double> tmp_m(n);
    std::vector<int> tmp_devices;
    std::set<int> indices;
    for (int i = 0; i < n; i++) {
        indices.insert(i);
        fin >> tmp_qxyz[3 * i] >> tmp_qxyz[3 * i + 1] >> tmp_qxyz[3 * i + 2] >> tmp_vxyz[3 * i] >> tmp_vxyz[3 * i + 1] >> tmp_vxyz[3 * i + 2] >> tmp_m[i] >> type;
        if (type == "device") {
            tmp_devices.push_back(i);
        }
    }

    qxyz = (double*)malloc(3 * n * sizeof(double));
    vxyz = (double*)malloc(3 * n * sizeof(double));
    m = (double*)malloc(n * sizeof(double));
    device_id = (int*)malloc(n * sizeof(int));
    device_cnt = tmp_devices.size();
    for (int i = 0; i < n; i++) {
        int tmp_i;
        if (i == 0) {
            tmp_i = planet;
        } else if (i == 1) {
            tmp_i = asteroid;
        } else if (i < device_cnt + 2) {
            tmp_i = tmp_devices[i - 2];
            device_id[i] = tmp_devices[i - 2];
        } else {
            tmp_i = *indices.begin();
        }
        qxyz[i * 3 + 0] = tmp_qxyz[tmp_i * 3 + 0];
        qxyz[i * 3 + 1] = tmp_qxyz[tmp_i * 3 + 1];
        qxyz[i * 3 + 2] = tmp_qxyz[tmp_i * 3 + 2];
        vxyz[i * 3 + 0] = tmp_vxyz[tmp_i * 3 + 0];
        vxyz[i * 3 + 1] = tmp_vxyz[tmp_i * 3 + 1];
        vxyz[i * 3 + 2] = tmp_vxyz[tmp_i * 3 + 2];
        m[i] = tmp_m[tmp_i];
        indices.erase(tmp_i);
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
                  int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

template <class T>
__global__ void clear_array_gpu(int n, T* array) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        array[i] = (T)0;
    }
}

__global__ void set_isdevice_gpu(int device_cnt, int* devices, bool* isdevice) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < device_cnt) {
        isdevice[devices[i]] = true;
    }
}

__global__ void compute_accelerations_gpu(const int step, const int n, const double* qxyz, double* vxyz, double* axyz, const double* m, const int device_cnt) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / n;
    int j = index % n;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (j > 1 && j < device_cnt + 2) {
            mj = param::gravity_device_mass_gpu(mj, step * param::dt);
        }
        double dx = qxyz[j * 3 + 0] - qxyz[i * 3 + 0];
        double dy = qxyz[j * 3 + 1] - qxyz[i * 3 + 1];
        double dz = qxyz[j * 3 + 2] - qxyz[i * 3 + 2];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        atomicAdd(&axyz[i * 3 + 0], param::G * mj * dx / dist3);
        atomicAdd(&axyz[i * 3 + 1], param::G * mj * dy / dist3);
        atomicAdd(&axyz[i * 3 + 2], param::G * mj * dz / dist3);
    }
}

__global__ void compute_accelerations_1_gpu(const int step, const int n, const double* qxyz, double* vxyz, double* axyz, const double* m, const int device_cnt) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / n;
    int j = index % n;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (j > 1 && j < device_cnt + 2) {
            mj = 0;
        }
        double dx = qxyz[j * 3 + 0] - qxyz[i * 3 + 0];
        double dy = qxyz[j * 3 + 1] - qxyz[i * 3 + 1];
        double dz = qxyz[j * 3 + 2] - qxyz[i * 3 + 2];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        atomicAdd(&axyz[i * 3 + 0], param::G * mj * dx / dist3);
        atomicAdd(&axyz[i * 3 + 1], param::G * mj * dy / dist3);
        atomicAdd(&axyz[i * 3 + 2], param::G * mj * dz / dist3);
    }
}

__global__ void compute_accelerations_3_gpu(const bool* hit, const int step, const int n, const double* qxyz, double* vxyz, double* axyz, const double* m, const int device_cnt) {
    if (*hit)
        return;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / n;
    int j = index % n;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (j > 1 && j < device_cnt + 2)
            mj = param::gravity_device_mass_gpu(mj, step * param::dt);

        double dx = qxyz[j * 3 + 0] - qxyz[i * 3 + 0];
        double dy = qxyz[j * 3 + 1] - qxyz[i * 3 + 1];
        double dz = qxyz[j * 3 + 2] - qxyz[i * 3 + 2];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        atomicAdd(&axyz[i * 3 + 0], param::G * mj * dx / dist3);
        atomicAdd(&axyz[i * 3 + 1], param::G * mj * dy / dist3);
        atomicAdd(&axyz[i * 3 + 2], param::G * mj * dz / dist3);
    }
}

__global__ void update_velocities_gpu(const int n, double* vxyz, double* axyz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update velocities
    if (i < 3 * n) {
        vxyz[i] += axyz[i] * param::dt;
        axyz[i] = 0;
    }
}

__global__ void clear_a_gpu(const int n, double* axyz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < 3 * n) {
        axyz[i] = 0;
    }
}

__global__ void update_positions_gpu(const int n, double* qxyz, const double* vxyz, const double* axyz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update positions
    if (i < 3 * n) {
        qxyz[i] += (vxyz[i] + axyz[i] * param::dt) * param::dt;
    }
}

__global__ void problem1(double* min_dist, const int n, const double* qxyz) {
    double dx = qxyz[0] - qxyz[3];
    double dy = qxyz[1] - qxyz[4];
    double dz = qxyz[2] - qxyz[5];
    double tmp_dst = sqrt(dx * dx + dy * dy + dz * dz);
    if (tmp_dst < *min_dist)
        *min_dist = tmp_dst;
}

__global__ void problem2(int* hit_time_step, const int step, const int n, const double* qxyz) {
    if (*hit_time_step != -2)
        return;
    double dx = qxyz[0] - qxyz[3];
    double dy = qxyz[1] - qxyz[4];
    double dz = qxyz[2] - qxyz[5];
    if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
        *hit_time_step = step;
    }
}

__global__ void problem3(bool* hit, double* cost, const int step, const int n, const int d, const double* qxyz, double* m) {
    if (*hit)
        return;
    double dx = qxyz[0] - qxyz[3];
    double dy = qxyz[1] - qxyz[4];
    double dz = qxyz[2] - qxyz[5];
    if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
        *hit = true;
        return;
    }
    if (m[d] != 0) {
        dx = qxyz[0] - qxyz[d * 3 + 0];
        dy = qxyz[1] - qxyz[d * 3 + 1];
        dz = qxyz[2] - qxyz[d * 3 + 2];
        double missle_dist = (param::missile_speed * param::dt) * step;
        if (dx * dx + dy * dy + dz * dz < missle_dist * missle_dist) {
            *cost = param::get_missile_cost_gpu((step + 1) * param::dt);
            m[d] = 0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n;
    double* qxyz0;
    double* vxyz0;
    double* m0;
    int device_cnt;
    int* device_id;
    read_input(argv[1], n, qxyz0, vxyz0, m0, device_cnt, device_id);

    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    int gravity_device_id = -1;
    double missile_cost = 0;

    dim3 BlockDim(param::threads_per_block);
    // dim3 GridDim(ceil((float)n / param::threads_per_block));
    auto GridDim = [&](int n) -> dim3 {
        return (ceil((float)n / param::threads_per_block));
    };

#pragma omp parallel for num_threads(2)
    for (int task = 0; task < 2; task++) {
        CUDA_CALL(cudaSetDevice(omp_get_thread_num()));
        cudaStream_t streams[param::cuda_nstreams];
        for (int i = 0; i < param::cuda_nstreams; i++) {
            CUDA_CALL(cudaStreamCreate(&streams[i]));
        }

        double* g_qxyz;
        double* g_vxyz;
        double* g_axyz;
        double* g_m;
        double* g_min_dist;
        int* g_hit_time_step;

        CUDA_CALL(cudaMalloc(&g_qxyz, 3 * n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_vxyz, 3 * n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_axyz, 3 * n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_m, n * sizeof(double)));
        if (task == 0) {
            CUDA_CALL(cudaMallocManaged(&g_min_dist, sizeof(double)));
            CUDA_CALL(cudaMemPrefetchAsync(g_min_dist, sizeof(double), cudaCpuDeviceId, streams[0]));
            *g_min_dist = std::numeric_limits<double>::infinity();
            CUDA_CALL(cudaMemPrefetchAsync(g_min_dist, sizeof(double), omp_get_thread_num(), streams[0]));
        } else if (task == 1) {
            CUDA_CALL(cudaMallocManaged(&g_hit_time_step, sizeof(int)));
            CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), cudaCpuDeviceId, streams[0]));
            *g_hit_time_step = -2;
            CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), omp_get_thread_num(), streams[0]));
        }

        CUDA_CALL(cudaMemcpyAsync(g_qxyz, qxyz0, 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vxyz, vxyz0, 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_axyz);

        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                if (task == 0)
                    compute_accelerations_1_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(step, n, g_qxyz, g_vxyz, g_axyz, g_m, device_cnt);
                else
                    compute_accelerations_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(step, n, g_qxyz, g_vxyz, g_axyz, g_m, device_cnt);
                update_positions_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_qxyz, g_vxyz, g_axyz);
            }
            if (task == 0) {
                problem1<<<1, 1, 0, streams[0]>>>(g_min_dist, n, g_qxyz);
                update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vxyz, g_axyz);
            } else if (task == 1) {
                problem2<<<1, 1, 0, streams[0]>>>(g_hit_time_step, step, n, g_qxyz);
                update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vxyz, g_axyz);
                if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                    CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), cudaCpuDeviceId, streams[0]));
                    cudaStreamSynchronize(streams[0]);
                    hit_time_step = *g_hit_time_step;
                    if (hit_time_step != -2)
                        break;
                }
            }
        }
        if (task == 0) {
            CUDA_CALL(cudaMemPrefetchAsync(g_min_dist, sizeof(double), cudaCpuDeviceId, streams[0]));
            cudaStreamSynchronize(streams[0]);
            min_dist = *g_min_dist;
        } else if (task == 1) {
            if (hit_time_step == -2) {
                CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), cudaCpuDeviceId, streams[0]));
                cudaStreamSynchronize(streams[0]);
                hit_time_step = *g_hit_time_step;
            }
        }
        for (int i = 0; i < param::cuda_nstreams; i++) {
            CUDA_CALL(cudaStreamDestroy(streams[i]));
        }
        CUDA_CALL(cudaFree(g_qxyz));
        CUDA_CALL(cudaFree(g_vxyz));
        CUDA_CALL(cudaFree(g_axyz));
        CUDA_CALL(cudaFree(g_m));
        if (task == 0)
            CUDA_CALL(cudaFree(g_min_dist));
        else if (task == 1)
            CUDA_CALL(cudaFree(g_hit_time_step));
    }  // omp end

    if (hit_time_step != -2) {
        // Problem 3
        gravity_device_id = -1;
        missile_cost = std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(static) num_threads(2)
        for (int di = 0; di < device_cnt; di++) {
            int thread_id = omp_get_thread_num();
            cudaSetDevice(thread_id);
            cudaStream_t streams[param::cuda_nstreams];
            for (int i = 0; i < param::cuda_nstreams; i++) {
                cudaStreamCreate(&streams[i]);
            }

            int d = di + 2;
            double* g_qxyz;
            double* g_vxyz;
            double* g_axyz;
            double* g_m;
            bool hit = false;
            bool* g_hit;  // hit and destroyed
            double* g_cost;

            CUDA_CALL(cudaMalloc(&g_qxyz, 3 * n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_vxyz, 3 * n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_axyz, 3 * n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_m, n * sizeof(double)));

            CUDA_CALL(cudaMallocManaged(&g_hit, sizeof(bool)));
            CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), cudaCpuDeviceId, streams[0]));
            *g_hit = false;
            CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), thread_id, streams[0]));
            CUDA_CALL(cudaMallocManaged(&g_cost, sizeof(double)));
            CUDA_CALL(cudaMemPrefetchAsync(g_cost, sizeof(bool), cudaCpuDeviceId, streams[0]));
            *g_cost = std::numeric_limits<double>::infinity();
            CUDA_CALL(cudaMemPrefetchAsync(g_cost, sizeof(bool), thread_id, streams[0]));

            CUDA_CALL(cudaMemcpyAsync(g_qxyz, qxyz0, 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vxyz, vxyz0, 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_axyz);

            // hit break
            for (int step = 0; step <= param::n_steps; step++) {
                if (step > 0) {
                    compute_accelerations_3_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(g_hit, step, n, g_qxyz, g_vxyz, g_axyz, g_m, device_cnt);
                    update_positions_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_qxyz, g_vxyz, g_axyz);
                }
                problem3<<<1, 1, 0, streams[0]>>>(g_hit, g_cost, step, n, d, g_qxyz, g_m);
                update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vxyz, g_axyz);
                if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                    CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), cudaCpuDeviceId, streams[0]));
                    cudaStreamSynchronize(streams[0]);
                    hit = *g_hit;
                    if (hit)
                        break;
                }
            }
            if (!hit) {
                CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), cudaCpuDeviceId, streams[0]));
                CUDA_CALL(cudaMemPrefetchAsync(g_cost, sizeof(double), cudaCpuDeviceId, streams[0]));
                cudaStreamSynchronize(streams[0]);
#pragma omp critical
                if (!*g_hit && *g_cost < missile_cost) {
                    gravity_device_id = d;
                    missile_cost = *g_cost;
                }
            }
            for (int i = 0; i < param::cuda_nstreams; i++) {
                cudaStreamDestroy(streams[i]);
            }
            CUDA_CALL(cudaFree(g_qxyz));
            CUDA_CALL(cudaFree(g_vxyz));
            CUDA_CALL(cudaFree(g_axyz));
            CUDA_CALL(cudaFree(g_m));
            CUDA_CALL(cudaFree(g_hit));
            CUDA_CALL(cudaFree(g_cost));
        }  // omp end
        if (gravity_device_id == -1) {
            missile_cost = 0;
        } else {
            gravity_device_id = device_id[gravity_device_id];
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);

    CUDA_CHECK();

    free(qxyz0);
    free(vxyz0);
    free(m0);
    free(device_id);
}

/*
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b20.in outputs/b20.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b30.in outputs/b30.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b40.in outputs/b40.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b50.in outputs/b50.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b60.in outputs/b60.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b70.in outputs/b70.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b80.in outputs/b80.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b90.in outputs/b90.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b100.in outputs/b100.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b200.in outputs/b200.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b512.in outputs/b512.out
make; srun -pipc22 -c2 --gres=gpu:2 ./hw5 testcases/b1024.in outputs/b1024.out

make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b20.in outputs/b20.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b30.in outputs/b30.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b40.in outputs/b40.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b50.in outputs/b50.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b60.in outputs/b60.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b70.in outputs/b70.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b80.in outputs/b80.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b90.in outputs/b90.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b100.in outputs/b100.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b200.in outputs/b200.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b512.in outputs/b512.out
make; srun -pipc22 -c2 --gres=gpu:2 cuda-memcheck ./hw5 testcases/b1024.in outputs/b1024.out

make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b20.in outputs/b20.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b30.in outputs/b30.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b40.in outputs/b40.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b50.in outputs/b50.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b60.in outputs/b60.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b70.in outputs/b70.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b80.in outputs/b80.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b90.in outputs/b90.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b100.in outputs/b100.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b200.in outputs/b200.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b512.in outputs/b512.out
make; srun -pipc22 -c2 --gres=gpu:2 nvprof ./hw5 testcases/b1024.in outputs/b1024.out
 */