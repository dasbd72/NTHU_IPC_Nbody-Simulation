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
const int n_sync_steps = 4000;
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

const int threads_per_block = 512;
const int cuda_nstreams = 3;
}  // namespace param

void read_input(const char* filename, int& n, double*& qx, double*& qy, double*& qz,
                double*& vx, double*& vy, double*& vz, double*& m, int& device_cnt, int*& device_id) {
    std::ifstream fin(filename);
    int planet, asteroid;
    fin >> n >> planet >> asteroid;

    std::string type;
    std::vector<double> tmp_qx(n);
    std::vector<double> tmp_qy(n);
    std::vector<double> tmp_qz(n);
    std::vector<double> tmp_vx(n);
    std::vector<double> tmp_vy(n);
    std::vector<double> tmp_vz(n);
    std::vector<double> tmp_m(n);
    std::vector<int> tmp_devices;
    std::set<int> indices;
    for (int i = 0; i < n; i++) {
        indices.insert(i);
        fin >> tmp_qx[i] >> tmp_qy[i] >> tmp_qz[i] >> tmp_vx[i] >> tmp_vy[i] >> tmp_vz[i] >> tmp_m[i] >> type;
        if (type == "device") {
            tmp_devices.push_back(i);
        }
    }

    qx = (double*)malloc(n * sizeof(double));
    qy = (double*)malloc(n * sizeof(double));
    qz = (double*)malloc(n * sizeof(double));
    vx = (double*)malloc(n * sizeof(double));
    vy = (double*)malloc(n * sizeof(double));
    vz = (double*)malloc(n * sizeof(double));
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
        qx[i] = tmp_qx[tmp_i];
        qy[i] = tmp_qy[tmp_i];
        qz[i] = tmp_qz[tmp_i];
        vx[i] = tmp_vx[tmp_i];
        vy[i] = tmp_vy[tmp_i];
        vz[i] = tmp_vz[tmp_i];
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

__global__ void compute_accelerations_1_gpu(const int step, const int n, const double* qx, const double* qy, const double* qz, double* vx, double* vy, double* vz, double* ax, double* ay, double* az, const double* m, const int device_cnt) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / n;
    int j = index % n;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (j > 1 && j < device_cnt + 2) {
            mj = 0;
        }
        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        atomicAdd(&ax[i], param::G * mj * dx / dist3);
        atomicAdd(&ay[i], param::G * mj * dy / dist3);
        atomicAdd(&az[i], param::G * mj * dz / dist3);
    }
}

__global__ void compute_accelerations_2_gpu(const int step, const int n, const double* qx, const double* qy, const double* qz, double* vx, double* vy, double* vz, double* ax, double* ay, double* az, const double* m, const int device_cnt) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / n;
    int j = index % n;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (j > 1 && j < device_cnt + 2) {
            mj = param::gravity_device_mass_gpu(mj, step * param::dt);
        }
        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        atomicAdd(&ax[i], param::G * mj * dx / dist3);
        atomicAdd(&ay[i], param::G * mj * dy / dist3);
        atomicAdd(&az[i], param::G * mj * dz / dist3);
    }
}

__global__ void compute_accelerations_3_gpu(const bool* hit, const bool* destroyed, const int step, const int n, const int d, const double* qx, const double* qy, const double* qz, double* vx, double* vy, double* vz, double* ax, double* ay, double* az, const double* m, const int device_cnt) {
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

        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        atomicAdd(&ax[i], param::G * mj * dx / dist3);
        atomicAdd(&ay[i], param::G * mj * dy / dist3);
        atomicAdd(&az[i], param::G * mj * dz / dist3);
    }
}

__global__ void update_velocities_gpu(const int n, double* vx, double* vy, double* vz, double* ax, double* ay, double* az) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update velocities
    if (i < n) {
        vx[i] += ax[i] * param::dt;
        ax[i] = 0;
    } else if (i < 2 * n) {
        vy[i - n] += ay[i - n] * param::dt;
        ay[i - n] = 0;
    } else if (i < 3 * n) {
        vz[i - 2 * n] += az[i - 2 * n] * param::dt;
        az[i - 2 * n] = 0;
    }
}

__global__ void clear_a_gpu(const int n, double* ax, double* ay, double* az) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        ax[i] = 0;
    } else if (i < 2 * n) {
        ay[i - n] = 0;
    } else if (i < 3 * n) {
        az[i - 2 * n] = 0;
    }
}

__global__ void update_positions_gpu(const int n, double* qx, double* qy, double* qz, const double* vx, const double* vy, const double* vz, const double* ax, const double* ay, const double* az) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update positions
    if (i < n) {
        qx[i] += (vx[i] + ax[i] * param::dt) * param::dt;
    } else if (i < 2 * n) {
        qy[i - n] += (vy[i - n] + ay[i - n] * param::dt) * param::dt;
    } else if (i < 3 * n) {
        qz[i - 2 * n] += (vz[i - 2 * n] + az[i - 2 * n] * param::dt) * param::dt;
    }
}

__global__ void problem1(double* min_dist, const int n, const double* qx, const double* qy, const double* qz) {
    double dx = qx[0] - qx[1];
    double dy = qy[0] - qy[1];
    double dz = qz[0] - qz[1];
    double tmp_dst = sqrt(dx * dx + dy * dy + dz * dz);
    if (tmp_dst < *min_dist)
        *min_dist = tmp_dst;
}

__global__ void problem2(int* hit_time_step, const int step, const int n, const double* qx, const double* qy, const double* qz) {
    if (*hit_time_step != -2)
        return;
    double dx = qx[0] - qx[1];
    double dy = qy[0] - qy[1];
    double dz = qz[0] - qz[1];
    if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
        *hit_time_step = step;
    }
}

__global__ void problem3(bool* hit, bool* destroyed, double* cost, const int step, const int n, const int d, const double* qx, const double* qy, const double* qz, double* m) {
    if (*hit)
        return;
    double dx = qx[0] - qx[1];
    double dy = qy[0] - qy[1];
    double dz = qz[0] - qz[1];
    if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
        *hit = true;
        return;
    }
    if (!*destroyed) {
        dx = qx[0] - qx[d];
        dy = qy[0] - qy[d];
        dz = qz[0] - qz[d];
        double missle_dist = (param::missile_speed * param::dt) * step;
        if (dx * dx + dy * dy + dz * dz < missle_dist * missle_dist) {
            *destroyed = true;
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
    double *qx0, *qy0, *qz0;
    double *vx0, *vy0, *vz0;
    double* m0;
    int device_cnt;
    int* device_id;
    read_input(argv[1], n, qx0, qy0, qz0, vx0, vy0, vz0, m0, device_cnt, device_id);

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

        double *p_qx, *p_qy, *p_qz;
        double *g_qx, *g_qy, *g_qz;
        double *g_vx, *g_vy, *g_vz;
        double *g_ax, *g_ay, *g_az;
        double* g_m;
        double* g_min_dist;
        int* g_hit_time_step;

        CUDA_CALL(cudaMalloc(&p_qx, 2 * sizeof(double)));
        CUDA_CALL(cudaMalloc(&p_qy, 2 * sizeof(double)));
        CUDA_CALL(cudaMalloc(&p_qz, 2 * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_qx, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_qy, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_qz, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_vx, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_vy, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_vz, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_ax, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_ay, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_az, n * sizeof(double)));
        CUDA_CALL(cudaMalloc(&g_m, n * sizeof(double)));
        if (task == 0) {
            CUDA_CALL(cudaMallocManaged(&g_min_dist, sizeof(double)));
            CUDA_CALL(cudaMemPrefetchAsync(g_min_dist, sizeof(double), cudaCpuDeviceId, streams[1]));
            *g_min_dist = std::numeric_limits<double>::infinity();
            CUDA_CALL(cudaMemPrefetchAsync(g_min_dist, sizeof(double), omp_get_thread_num(), streams[1]));
        } else if (task == 1) {
            CUDA_CALL(cudaMallocManaged(&g_hit_time_step, sizeof(int)));
            CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), cudaCpuDeviceId, streams[1]));
            *g_hit_time_step = -2;
            CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), omp_get_thread_num(), streams[1]));
        }

        CUDA_CALL(cudaMemcpyAsync(p_qx, qx0, 2 * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
        CUDA_CALL(cudaMemcpyAsync(p_qy, qy0, 2 * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
        CUDA_CALL(cudaMemcpyAsync(p_qz, qz0, 2 * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
        CUDA_CALL(cudaMemcpyAsync(g_qx, qx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_qy, qy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_qz, qz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vx, vx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vy, vy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vz, vz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[1]>>>(n, g_ax, g_ay, g_az);

        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                if (task == 0)
                    compute_accelerations_1_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(step, n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az, g_m, device_cnt);
                else
                    compute_accelerations_2_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(step, n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az, g_m, device_cnt);
                update_positions_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az);
                CUDA_CALL(cudaStreamSynchronize(streams[1]));
                CUDA_CALL(cudaMemcpy(p_qx, g_qx, 2 * sizeof(double), cudaMemcpyDeviceToDevice));
                CUDA_CALL(cudaMemcpy(p_qy, g_qy, 2 * sizeof(double), cudaMemcpyDeviceToDevice));
                CUDA_CALL(cudaMemcpy(p_qz, g_qz, 2 * sizeof(double), cudaMemcpyDeviceToDevice));
                update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vx, g_vy, g_vz, g_ax, g_ay, g_az);
            }
            if (task == 0) {
                problem1<<<1, 1, 0, streams[1]>>>(g_min_dist, n, p_qx, p_qy, p_qz);
            } else if (task == 1) {
                problem2<<<1, 1, 0, streams[1]>>>(g_hit_time_step, step, n, p_qx, p_qy, p_qz);
                if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                    CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), cudaCpuDeviceId, streams[1]));
                    CUDA_CALL(cudaStreamSynchronize(streams[1]));
                    hit_time_step = *g_hit_time_step;
                    if (hit_time_step != -2)
                        break;
                }
            }
        }
        if (task == 0) {
            CUDA_CALL(cudaMemPrefetchAsync(g_min_dist, sizeof(double), cudaCpuDeviceId, streams[1]));
            CUDA_CALL(cudaStreamSynchronize(streams[1]));
            min_dist = *g_min_dist;
        } else if (task == 1) {
            if (hit_time_step == -2) {
                CUDA_CALL(cudaMemPrefetchAsync(g_hit_time_step, sizeof(int), cudaCpuDeviceId, streams[1]));
                CUDA_CALL(cudaStreamSynchronize(streams[1]));
                hit_time_step = *g_hit_time_step;
            }
        }
        for (int i = 0; i < param::cuda_nstreams; i++) {
            CUDA_CALL(cudaStreamDestroy(streams[i]));
        }
        CUDA_CALL(cudaFree(p_qx));
        CUDA_CALL(cudaFree(p_qy));
        CUDA_CALL(cudaFree(p_qz));
        CUDA_CALL(cudaFree(g_qx));
        CUDA_CALL(cudaFree(g_qy));
        CUDA_CALL(cudaFree(g_qz));
        CUDA_CALL(cudaFree(g_vx));
        CUDA_CALL(cudaFree(g_vy));
        CUDA_CALL(cudaFree(g_vz));
        CUDA_CALL(cudaFree(g_ax));
        CUDA_CALL(cudaFree(g_ay));
        CUDA_CALL(cudaFree(g_az));
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
            double *p_qx, *p_qy, *p_qz;
            double *g_qx, *g_qy, *g_qz;
            double *g_vx, *g_vy, *g_vz;
            double *g_ax, *g_ay, *g_az;
            double* g_m;
            bool hit = false;
            bool* g_hit;
            bool* g_destroyed;
            double* g_cost;

            CUDA_CALL(cudaMalloc(&p_qx, (2 + device_cnt) * sizeof(double)));
            CUDA_CALL(cudaMalloc(&p_qy, (2 + device_cnt) * sizeof(double)));
            CUDA_CALL(cudaMalloc(&p_qz, (2 + device_cnt) * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_qx, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_qy, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_qz, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_vx, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_vy, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_vz, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_ax, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_ay, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_az, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_m, n * sizeof(double)));
            CUDA_CALL(cudaMalloc(&g_destroyed, sizeof(bool)));

            CUDA_CALL(cudaMallocManaged(&g_hit, sizeof(bool)));
            CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), cudaCpuDeviceId, streams[1]));
            *g_hit = false;
            CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), thread_id, streams[1]));
            CUDA_CALL(cudaMallocManaged(&g_cost, sizeof(double)));
            CUDA_CALL(cudaMemPrefetchAsync(g_cost, sizeof(bool), cudaCpuDeviceId, streams[1]));
            *g_cost = std::numeric_limits<double>::infinity();
            CUDA_CALL(cudaMemPrefetchAsync(g_cost, sizeof(bool), thread_id, streams[1]));

            CUDA_CALL(cudaMemcpyAsync(p_qx, qx0, (2 + device_cnt) * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
            CUDA_CALL(cudaMemcpyAsync(p_qy, qy0, (2 + device_cnt) * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
            CUDA_CALL(cudaMemcpyAsync(p_qz, qz0, (2 + device_cnt) * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
            CUDA_CALL(cudaMemcpyAsync(g_qx, qx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_qy, qy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_qz, qz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vx, vx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vy, vy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vz, vz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_ax, g_ay, g_az);
            CUDA_CALL(cudaMemsetAsync(g_destroyed, m0[d] == 0, sizeof(bool), streams[1]));

            // hit break
            for (int step = 0; step <= param::n_steps && !hit; step++) {
                if (step > 0) {
                    cudaStreamSynchronize(streams[1]);
                    compute_accelerations_3_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(g_hit, g_destroyed, step, n, d, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az, g_m, device_cnt);
                    update_positions_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az);
                    CUDA_CALL(cudaMemcpy(p_qx, g_qx, (2 + device_cnt) * sizeof(double), cudaMemcpyDeviceToDevice));
                    CUDA_CALL(cudaMemcpy(p_qy, g_qy, (2 + device_cnt) * sizeof(double), cudaMemcpyDeviceToDevice));
                    CUDA_CALL(cudaMemcpy(p_qz, g_qz, (2 + device_cnt) * sizeof(double), cudaMemcpyDeviceToDevice));
                    update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vx, g_vy, g_vz, g_ax, g_ay, g_az);
                }
                problem3<<<1, 1, 0, streams[1]>>>(g_hit, g_destroyed, g_cost, step, n, d, p_qx, p_qy, p_qz, g_m);
                if (step % param::n_sync_steps == param::n_sync_steps - 1) {
                    CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), cudaCpuDeviceId, streams[1]));
                    cudaStreamSynchronize(streams[1]);
                    hit = *g_hit;
                    if (hit)
                        break;
                }
            }
            if (!hit) {
                CUDA_CALL(cudaMemPrefetchAsync(g_hit, sizeof(bool), cudaCpuDeviceId, streams[1]));
                CUDA_CALL(cudaMemPrefetchAsync(g_cost, sizeof(double), cudaCpuDeviceId, streams[1]));
                cudaStreamSynchronize(streams[1]);
#pragma omp critical
                if (!*g_hit && *g_cost < missile_cost) {
                    gravity_device_id = d;
                    missile_cost = *g_cost;
                }
            }
            for (int i = 0; i < param::cuda_nstreams; i++) {
                cudaStreamDestroy(streams[i]);
            }
            CUDA_CALL(cudaFree(p_qx));
            CUDA_CALL(cudaFree(p_qy));
            CUDA_CALL(cudaFree(p_qz));
            CUDA_CALL(cudaFree(g_qx));
            CUDA_CALL(cudaFree(g_qy));
            CUDA_CALL(cudaFree(g_qz));
            CUDA_CALL(cudaFree(g_vx));
            CUDA_CALL(cudaFree(g_vy));
            CUDA_CALL(cudaFree(g_vz));
            CUDA_CALL(cudaFree(g_ax));
            CUDA_CALL(cudaFree(g_ay));
            CUDA_CALL(cudaFree(g_az));
            CUDA_CALL(cudaFree(g_m));
            CUDA_CALL(cudaFree(g_hit));
            CUDA_CALL(cudaFree(g_destroyed));
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

    free(qx0);
    free(qy0);
    free(qz0);
    free(vx0);
    free(vy0);
    free(vz0);
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