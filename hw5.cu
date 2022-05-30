// #define DEBUG
// #define USE_SHARED_MEMORY

#include <nppdefs.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
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
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
__device__ double gravity_device_mass_gpu(double m0, double t) {
    if (m0 == 0)
        return 0;
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
__device__ double gravity_device_mass_fst_gpu(double m0, double fst) {
    return m0 + 0.5 * m0 * fst;
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_dist(int step) { return (missile_speed * missile_speed * dt * dt) * (step * step); }
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
__device__ double get_missile_cost_gpu(double t) { return 1e5 + 1e3 * t; }

const int n_sync_steps = 2000;
const int threads_per_block = 64;
const int threads_x = 4;
const int threads_y = 64;
const int cuda_nstreams = 3;
const dim3 BlockDim(param::threads_per_block);
dim3 GridDim(int n) {
    return dim3(ceil((float)n / param::threads_per_block));
}
const dim3 BlockDim2D(param::threads_x, param::threads_y);
dim3 GridDim2D(int n) {
    return dim3(ceil((float)n / param::threads_x), ceil((float)n / param::threads_y));
};
}  // namespace param

std::mutex g_mutex;

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

__global__ void calc_step2fst_gpu(double* step2fst) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < param::n_steps) {
        step2fst[i] = fabs(sin(i * param::dt / 6000));
    }
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
#ifndef USE_SHARED_MEMORY
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (j > 1 && j < device_cnt + 2) {
            // mj = param::gravity_device_mass_fst_gpu(mj, fst);
            mj = param::gravity_device_mass_gpu(mj, step * param::dt);
        }
        double dxyz[3];
        dxyz[0] = qxyz[j * 3 + 0] - qxyz[i * 3 + 0];
        dxyz[1] = qxyz[j * 3 + 1] - qxyz[i * 3 + 1];
        dxyz[2] = qxyz[j * 3 + 2] - qxyz[i * 3 + 2];
        double dist3 = pow(dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2] + param::eps * param::eps, 1.5);

        const double c = param::G * mj / dist3;
        atomicAdd(&axyz[i * 3 + 0], c * dxyz[0]);
        atomicAdd(&axyz[i * 3 + 1], c * dxyz[1]);
        atomicAdd(&axyz[i * 3 + 2], c * dxyz[2]);
    }
#else
    __shared__ double s_i_qxyz[param::threads_x * 3];
    __shared__ double s_j_qxyz[param::threads_y * 3];
    __shared__ double s_j_m[param::threads_y];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (threadIdx.y < 3) {
        s_i_qxyz[threadIdx.x * 3 + threadIdx.y] = qxyz[i * 3 + threadIdx.y];
    }
    if (threadIdx.x < 3) {
        s_j_qxyz[threadIdx.y * 3 + threadIdx.x] = qxyz[j * 3 + threadIdx.x];
    } else if (threadIdx.x < 4) {
        s_j_m[threadIdx.y] = m[j];
        if (j > 1 && j < device_cnt + 2) {
            // s_j_m[threadIdx.y] = param::gravity_device_mass_fst_gpu(s_j_m[threadIdx.y], fst);
            s_j_m[threadIdx.y] = param::gravity_device_mass_gpu(s_j_m[threadIdx.y], step * param::dt);
        }
    }
    __syncthreads();
    // compute accelerations
    if (i < n && j < n && i != j) {
        double dxyz[3];
        dxyz[0] = s_j_qxyz[threadIdx.y * 3 + 0] - s_i_qxyz[threadIdx.x * 3 + 0];
        dxyz[1] = s_j_qxyz[threadIdx.y * 3 + 1] - s_i_qxyz[threadIdx.x * 3 + 1];
        dxyz[2] = s_j_qxyz[threadIdx.y * 3 + 2] - s_i_qxyz[threadIdx.x * 3 + 2];
        double dist3 =
            pow(dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2] + param::eps * param::eps, 1.5);

        const double c = param::G * s_j_m[threadIdx.y] / dist3;
        atomicAdd(&axyz[i * 3 + 0], c * dxyz[0]);
        atomicAdd(&axyz[i * 3 + 1], c * dxyz[1]);
        atomicAdd(&axyz[i * 3 + 2], c * dxyz[2]);
    }
#endif
}

__global__ void update_velocities_gpu(const int n, double* vxyz, double* axyz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update velocities
    if (i < 3 * n) {
        vxyz[i] += axyz[i] * param::dt;
        axyz[i] = 0;
    }
}

__global__ void clear_device_m_gpu(const int device_cnt, double* m) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < device_cnt) {
        m[i + 2] = 0;
    }
}

__global__ void clear_a_gpu(const int n, double* axyz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < 3 * n) {
        axyz[i] = 0;
    }
}

__global__ void update_positions_gpu(const int n, double* qxyz, double* vxyz, double* axyz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update positions
    if (i < 3 * n) {
        vxyz[i] += axyz[i] * param::dt;
        qxyz[i] += vxyz[i] * param::dt;
        axyz[i] = 0;
    }
}

__global__ void calc_sq_min_dist_gpu(bool* done, double* sq_min_dist, const double* qxyz) {
    double dx = qxyz[0] - qxyz[3];
    double dy = qxyz[1] - qxyz[4];
    double dz = qxyz[2] - qxyz[5];
    double tmp_dst = dx * dx + dy * dy + dz * dz;
    if (tmp_dst < *sq_min_dist) {
        *sq_min_dist = tmp_dst;
        *done = false;
    } else {
        *done = true;
    }
}

__global__ void calc_hit_time_step_gpu(int* hit_time_step, const int step, const double* qxyz) {
    if (*hit_time_step == -2) {
        double dx = qxyz[0] - qxyz[3];
        double dy = qxyz[1] - qxyz[4];
        double dz = qxyz[2] - qxyz[5];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            *hit_time_step = step;
        }
    }
}

__global__ void problem3_preprocess_gpu(const int step, const int n, const double* qxyz, const double* vxyz, const int device_cnt, int* p3_step, double* p3_qxyz, double* p3_vxyz) {
    // int index = blockDim.x * blockIdx.x + threadIdx.x;
    int di = threadIdx.x;
    if (p3_step[di] == -2) {
        int d = di + 2;
        double dx = qxyz[0] - qxyz[d * 3];
        double dy = qxyz[1] - qxyz[d * 3 + 1];
        double dz = qxyz[2] - qxyz[d * 3 + 2];
        double missle_dist = (param::missile_speed * param::dt) * step;
        if (dx * dx + dy * dy + dz * dz < missle_dist * missle_dist) {
            p3_step[di] = step;
            int c = di * 3 * n;
            for (int i = 0; i < n; i++) {
                p3_qxyz[c + i * 3 + 0] = qxyz[i * 3 + 0];
                p3_qxyz[c + i * 3 + 1] = qxyz[i * 3 + 1];
                p3_qxyz[c + i * 3 + 2] = qxyz[i * 3 + 2];
                p3_vxyz[c + i * 3 + 0] = vxyz[i * 3 + 0];
                p3_vxyz[c + i * 3 + 1] = vxyz[i * 3 + 1];
                p3_vxyz[c + i * 3 + 2] = vxyz[i * 3 + 2];
            }
        }
    }
}

__global__ void missile_cost_gpu(bool* hit, double* cost, const int step, const int n, const int d, const double* qxyz, double* m) {
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

void t_problem_12(int tid, int n, int device_cnt, double* qxyz0, double* vxyz0, double* m0, double& min_dist, int& hit_time_step, int* p3_step, double* p3_qxyz, double* p3_vxyz) {
    CUDA_CALL(cudaSetDevice(tid));
    cudaStream_t streams[param::cuda_nstreams];
    for (int i = 0; i < param::cuda_nstreams; i++)
        CUDA_CALL(cudaStreamCreate(&streams[i]));

    double* g_qxyz;
    double* g_vxyz;
    double* g_axyz;
    double* g_m;
    bool done = false;
    bool* g_done;
    double* g_sq_min_dist;
    int* g_hit_time_step;
    int* g_p3_step;
    double* g_p3_qxyz;
    double* g_p3_vxyz;

    CUDA_CALL(cudaMalloc(&g_qxyz, 3 * n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&g_vxyz, 3 * n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&g_axyz, 3 * n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&g_m, n * sizeof(double)));
    CUDA_CALL(cudaMemcpyAsync(g_qxyz, qxyz0, 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CALL(cudaMemcpyAsync(g_vxyz, vxyz0, 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));

    if (tid == 0) {
        CUDA_CALL(cudaMalloc(&g_done, sizeof(bool)));
        CUDA_CALL(cudaMemset(g_done, 0, sizeof(bool)));
        CUDA_CALL(cudaMalloc(&g_sq_min_dist, sizeof(double)));
        CUDA_CALL(cudaMemcpyAsync(g_sq_min_dist, &min_dist, sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        clear_device_m_gpu<<<param::GridDim(device_cnt), param::BlockDim, 0, streams[0]>>>(device_cnt, g_m);
    } else if (tid == 1) {
        CUDA_CALL(cudaMalloc(&g_p3_step, device_cnt * sizeof(int)));
        CUDA_CALL(cudaMalloc(&g_p3_qxyz, 3 * n * device_cnt * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_p3_vxyz, 3 * n * device_cnt * sizeof(double)));
        CUDA_CALL(cudaMemcpyAsync(g_p3_step, p3_step, device_cnt * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMalloc(&g_hit_time_step, sizeof(int)));
        CUDA_CALL(cudaMemcpyAsync(g_hit_time_step, &hit_time_step, sizeof(int), cudaMemcpyHostToDevice, streams[0]));
    }
    clear_a_gpu<<<param::GridDim(3 * n), param::BlockDim, 0, streams[0]>>>(n, g_axyz);
    if (tid == 0) {
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_accelerations_gpu<<<param::GridDim2D(n), param::BlockDim2D, 0, streams[0]>>>(step, n, g_qxyz, g_vxyz, g_axyz, g_m, device_cnt);
                update_positions_gpu<<<param::GridDim(3 * n), param::BlockDim, 0, streams[0]>>>(n, g_qxyz, g_vxyz, g_axyz);
            }
            calc_sq_min_dist_gpu<<<1, 1, 0, streams[0]>>>(g_done, g_sq_min_dist, g_qxyz);
            /* if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                CUDA_CALL(cudaMemcpy(&done, g_done, sizeof(bool), cudaMemcpyDeviceToHost));
                if (done)
                    break;
            } */
        }
    } else if (tid == 1) {
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_accelerations_gpu<<<param::GridDim2D(n), param::BlockDim2D, 0, streams[0]>>>(step, n, g_qxyz, g_vxyz, g_axyz, g_m, device_cnt);
                update_positions_gpu<<<param::GridDim(3 * n), param::BlockDim, 0, streams[0]>>>(n, g_qxyz, g_vxyz, g_axyz);
            }
            problem3_preprocess_gpu<<<1, device_cnt, 0, streams[0]>>>(step, n, g_qxyz, g_vxyz, device_cnt, g_p3_step, g_p3_qxyz, g_p3_vxyz);
            calc_hit_time_step_gpu<<<1, 1, 0, streams[0]>>>(g_hit_time_step, step, g_qxyz);
            if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                CUDA_CALL(cudaMemcpy(&hit_time_step, g_hit_time_step, sizeof(int), cudaMemcpyDeviceToHost));
                if (hit_time_step != -2)
                    break;
            }
        }
    }
    if (tid == 0) {
        CUDA_CALL(cudaMemcpy(&min_dist, g_sq_min_dist, sizeof(double), cudaMemcpyDeviceToHost));
        min_dist = sqrt(min_dist);
    } else if (tid == 1) {
        if (hit_time_step == -2)
            CUDA_CALL(cudaMemcpyAsync(&hit_time_step, g_hit_time_step, sizeof(int), cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(p3_step, g_p3_step, device_cnt * sizeof(int), cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(p3_qxyz, g_p3_qxyz, 3 * n * device_cnt * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(p3_vxyz, g_p3_vxyz, 3 * n * device_cnt * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
    }
    CUDA_CHECK();
    cudaDeviceSynchronize();
    for (int i = 0; i < param::cuda_nstreams; i++)
        CUDA_CALL(cudaStreamDestroy(streams[i]));
    CUDA_CALL(cudaFree(g_qxyz));
    CUDA_CALL(cudaFree(g_vxyz));
    CUDA_CALL(cudaFree(g_axyz));
    CUDA_CALL(cudaFree(g_m));
    if (tid == 0) {
        CUDA_CALL(cudaFree(g_sq_min_dist));
    } else if (tid == 1) {
        CUDA_CALL(cudaFree(g_p3_step));
        CUDA_CALL(cudaFree(g_p3_qxyz));
        CUDA_CALL(cudaFree(g_p3_vxyz));
        CUDA_CALL(cudaFree(g_hit_time_step));
    }
}

void t_problem_3(int tid, int n, int& global_di, int device_cnt, int* p3_step, double* p3_qxyz, double* p3_vxyz, double* m0, int& gravity_device_id, double& missile_cost) {
    cudaSetDevice(tid);
    // Problem 3
    int di;
    bool thread_done = false;
    while (!thread_done) {
        g_mutex.lock();
        if (global_di < device_cnt)
            di = global_di++;
        else
            thread_done = true;
        g_mutex.unlock();
        if (p3_step[di] == -2 || thread_done)
            continue;

        cudaStream_t streams[param::cuda_nstreams];
        for (int i = 0; i < param::cuda_nstreams; i++)
            cudaStreamCreate(&streams[i]);

        int d = di + 2;
        double* g_qxyz;
        double* g_vxyz;
        double* g_axyz;
        double* g_m;
        bool hit = false;
        bool* g_hit;  // hit and destroyed
        double cost = std::numeric_limits<double>::infinity();
        double* g_cost;

        CUDA_CALL(cudaMalloc(&g_qxyz, 3 * n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_vxyz, 3 * n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_axyz, 3 * n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_m, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_hit, sizeof(bool)));
        CUDA_CALL(cudaMalloc(&g_cost, sizeof(double)));

        CUDA_CALL(cudaMemcpyAsync(g_qxyz, p3_qxyz + (3 * n * di), 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vxyz, p3_vxyz + (3 * n * di), 3 * n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_hit, &hit, sizeof(bool), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_cost, &cost, sizeof(double), cudaMemcpyHostToDevice, streams[0]));

        clear_a_gpu<<<param::GridDim(3 * n), param::BlockDim, 0, streams[0]>>>(n, g_axyz);
        for (int step = p3_step[di]; step <= param::n_steps; step++) {
            if (step > p3_step[di]) {
                compute_accelerations_gpu<<<param::GridDim2D(n), param::BlockDim2D, 0, streams[0]>>>(step, n, g_qxyz, g_vxyz, g_axyz, g_m, device_cnt);
                update_positions_gpu<<<param::GridDim(3 * n), param::BlockDim, 0, streams[0]>>>(n, g_qxyz, g_vxyz, g_axyz);
            }
            missile_cost_gpu<<<1, 1, 0, streams[0]>>>(g_hit, g_cost, step, n, d, g_qxyz, g_m);
            if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                CUDA_CALL(cudaMemcpy(&hit, g_hit, sizeof(bool), cudaMemcpyDeviceToHost));
                if (hit)
                    break;
            }
        }
        if (!hit) {
            CUDA_CALL(cudaMemcpy(&hit, g_hit, sizeof(bool), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(&cost, g_cost, sizeof(double), cudaMemcpyDeviceToHost));
            if (!hit && cost < missile_cost) {
                g_mutex.lock();
                gravity_device_id = d;
                missile_cost = cost;
                g_mutex.unlock();
            }
        }
        CUDA_CHECK();
        cudaDeviceSynchronize();
        for (int i = 0; i < param::cuda_nstreams; i++)
            cudaStreamDestroy(streams[i]);
        CUDA_CALL(cudaFree(g_qxyz));
        CUDA_CALL(cudaFree(g_vxyz));
        CUDA_CALL(cudaFree(g_axyz));
        CUDA_CALL(cudaFree(g_m));
        CUDA_CALL(cudaFree(g_hit));
        CUDA_CALL(cudaFree(g_cost));
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    __debug_printf("Start...\n");
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

    int* p3_step = (int*)malloc(device_cnt * sizeof(int));
    for (int i = 0; i < device_cnt; i++) p3_step[i] = -2;
    double* p3_qxyz = (double*)malloc(3 * n * device_cnt * sizeof(double));
    double* p3_vxyz = (double*)malloc(3 * n * device_cnt * sizeof(double));

    /* double step2fst[param::n_steps];
    {
        double* g_step2fst;
        cudaMalloc(&g_step2fst, param::n_steps * sizeof(double));
        calc_step2fst_gpu<<<param::GridDim(param::n_steps), param::BlockDim>>>(g_step2fst);
        cudaMemcpy(step2fst, g_step2fst, param::n_steps * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(g_step2fst);
    } */

    std::vector<std::thread> my_threads;
    // part 1
    for (int tid = 0; tid < 2; tid++) {
        my_threads.push_back(std::thread(t_problem_12, tid, n, device_cnt, qxyz0, vxyz0, m0, std::ref(min_dist), std::ref(hit_time_step), p3_step, p3_qxyz, p3_vxyz));
    }
    for (int tid = 0; tid < 2; tid++) {
        my_threads[tid].join();
    }
    my_threads.clear();

    if (hit_time_step != -2) {
        // part2
        gravity_device_id = -1;
        missile_cost = std::numeric_limits<double>::infinity();
        int global_di = 0;

        for (int tid = 0; tid < 2; tid++) {
            my_threads.push_back(std::thread(t_problem_3, tid, n, std::ref(global_di), device_cnt, p3_step, p3_qxyz, p3_vxyz, m0, std::ref(gravity_device_id), std::ref(missile_cost)));
        }
        for (int tid = 0; tid < 2; tid++) {
            my_threads[tid].join();
        }
        my_threads.clear();

        if (gravity_device_id == -1)
            missile_cost = 0;
        else
            gravity_device_id = device_id[gravity_device_id];
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);

    free(qxyz0);
    free(vxyz0);
    free(m0);
    free(device_id);
    free(p3_step);
    free(p3_qxyz);
    free(p3_vxyz);
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

make; nvprof ./hw5 testcases/b20.in outputs/b20.out
make; nvprof ./hw5 testcases/b30.in outputs/b30.out
make; nvprof ./hw5 testcases/b40.in outputs/b40.out
make; nvprof ./hw5 testcases/b50.in outputs/b50.out
make; nvprof ./hw5 testcases/b60.in outputs/b60.out
make; nvprof ./hw5 testcases/b70.in outputs/b70.out
make; nvprof ./hw5 testcases/b80.in outputs/b80.out
make; nvprof ./hw5 testcases/b90.in outputs/b90.out
make; nvprof ./hw5 testcases/b100.in outputs/b100.out
make; nvprof ./hw5 testcases/b200.in outputs/b200.out
make; nvprof ./hw5 testcases/b512.in outputs/b512.out
make; nvprof ./hw5 testcases/b1024.in outputs/b1024.out
 */