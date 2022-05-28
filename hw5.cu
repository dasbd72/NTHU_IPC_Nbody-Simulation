#define DEBUG

#include <nppdefs.h>
#include <omp.h>
#include <pthread.h>
#include <sm_60_atomic_functions.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }

const int threads_per_block = 256;
const int cuda_nstreams = 3;
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid, double*& qx, double*& qy, double*& qz,
                double*& vx, double*& vy, double*& vz, double*& m, int& device_cnt, int*& devices) {
    std::ifstream fin(filename);
    std::string type;
    std::vector<int> tmp_devices;
    fin >> n >> planet >> asteroid;
    qx = (double*)malloc(n * sizeof(double));
    qy = (double*)malloc(n * sizeof(double));
    qz = (double*)malloc(n * sizeof(double));
    vx = (double*)malloc(n * sizeof(double));
    vy = (double*)malloc(n * sizeof(double));
    vz = (double*)malloc(n * sizeof(double));
    m = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type;
        if (type == "device") {
            tmp_devices.push_back(i);
        }
    }
    device_cnt = tmp_devices.size();
    devices = (int*)malloc(device_cnt * sizeof(int));
    for (int i = 0; i < device_cnt; i++) {
        devices[i] = tmp_devices[i];
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

__global__ void compute_accelerations_gpu(const bool isProblem1, const int step, const int n, const double* qx, const double* qy, const double* qz, const double* vx, const double* vy, const double* vz, double* ax, double* ay, double* az, const double* m, const bool* isdevice) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / n;
    int j = index % n;

    // compute accelerations
    if (i < n && j < n && i != j) {
        double mj = m[j];
        if (isdevice[j]) {
            if (isProblem1)
                mj = 0;
            else
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

__global__ void update_velocities_gpu(int n, double* vx, double* vy, double* vz, double* ax, double* ay, double* az) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update velocities
    if (i < n) {
        vx[i] += ax[i] * param::dt;
    } else if (i < 2 * n) {
        vy[i - n] += ay[i - n] * param::dt;
    } else if (i < 3 * n) {
        vz[i - 2 * n] += az[i - 2 * n] * param::dt;
    }
}

__global__ void clear_a_gpu(int n, double* ax, double* ay, double* az) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        ax[i] = 0;
    } else if (i < 2 * n) {
        ay[i - n] = 0;
    } else if (i < 3 * n) {
        az[i - 2 * n] = 0;
    }
}

__global__ void update_positions_gpu(int n, double* qx, double* qy, double* qz, double* vx, double* vy, double* vz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // update positions
    if (i < n) {
        qx[i] += vx[i] * param::dt;
    } else if (i < 2 * n) {
        qy[i - n] += vy[i - n] * param::dt;
    } else if (i < 3 * n) {
        qz[i - 2 * n] += vz[i - 2 * n] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    double *qx0, *qy0, *qz0, *vx0, *vy0, *vz0, *m0;
    int device_cnt;
    int* devices;
    read_input(argv[1], n, planet, asteroid, qx0, qy0, qz0, vx0, vy0, vz0, m0, device_cnt, devices);

    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    int gravity_device_id = -1;
    double missile_cost = 0;

    dim3 BlockDim(param::threads_per_block);
    // dim3 GridDim(ceil((float)n / param::threads_per_block));
    auto GridDim = [&](int n) -> dim3 {
        return (ceil((float)n / param::threads_per_block));
    };

#pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);
        cudaStream_t streams[param::cuda_nstreams];
        for (int i = 0; i < param::cuda_nstreams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        double *qx, *qy, *qz;
        double *g_qx, *g_qy, *g_qz, *g_vx, *g_vy, *g_vz, *g_ax, *g_ay, *g_az, *g_m;
        bool* g_isdevice;
        int* g_devices;

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
        CUDA_CALL(cudaMalloc(&g_isdevice, n * sizeof(bool)));
        CUDA_CALL(cudaMalloc(&g_devices, device_cnt * sizeof(int)));

        CUDA_CALL(cudaMemcpyAsync(g_qx, qx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_qy, qy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_qz, qz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vx, vx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vy, vy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_vz, vz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CUDA_CALL(cudaMemcpyAsync(g_devices, devices, device_cnt * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
        clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[1]>>>(n, g_ax, g_ay, g_az);
        clear_array_gpu<<<GridDim(n), BlockDim, 0, streams[0]>>>(n, g_isdevice);
        set_isdevice_gpu<<<GridDim(n), BlockDim, 0, streams[0]>>>(device_cnt, g_devices, g_isdevice);
        qx = (double*)malloc(n * sizeof(double));
        qy = (double*)malloc(n * sizeof(double));
        qz = (double*)malloc(n * sizeof(double));

        for (int step = 0; step <= param::n_steps; step++) {
            double dx, qx_p, qx_a;
            double dy, qy_p, qy_a;
            double dz, qz_p, qz_a;
            if (step == 0) {
                qx_p = qx0[planet];
                qy_p = qy0[planet];
                qz_p = qz0[planet];
                qx_a = qx0[asteroid];
                qy_a = qy0[asteroid];
                qz_a = qz0[asteroid];
            } else {
                cudaStreamSynchronize(streams[1]);
                if (thread_id == 0)
                    compute_accelerations_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(true, step, n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az, g_m, g_isdevice);
                else
                    compute_accelerations_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(false, step, n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az, g_m, g_isdevice);
                update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vx, g_vy, g_vz, g_ax, g_ay, g_az);
                cudaStreamSynchronize(streams[0]);
                clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[1]>>>(n, g_ax, g_ay, g_az);
                update_positions_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz);
                CUDA_CALL(cudaMemcpyAsync(qx, g_qx, n * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
                CUDA_CALL(cudaMemcpyAsync(qy, g_qy, n * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
                CUDA_CALL(cudaMemcpyAsync(qz, g_qz, n * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
                CUDA_CALL(cudaStreamSynchronize(streams[0]));
                qx_p = qx[planet];
                qy_p = qy[planet];
                qz_p = qz[planet];
                qx_a = qx[asteroid];
                qy_a = qy[asteroid];
                qz_a = qz[asteroid];
            }
            dx = qx_p - qx_a;
            dy = qy_p - qy_a;
            dz = qz_p - qz_a;
            if (thread_id == 0) {
                min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
            } else if (thread_id == 1) {
                if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                    hit_time_step = step;
                    break;
                }
            }
        }
        for (int i = 0; i < param::cuda_nstreams; i++) {
            cudaStreamDestroy(streams[i]);
        }
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
        CUDA_CALL(cudaFree(g_isdevice));
        CUDA_CALL(cudaFree(g_devices));
    }

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

            double *qx, *qy, *qz;
            double *g_qx, *g_qy, *g_qz, *g_vx, *g_vy, *g_vz, *g_ax, *g_ay, *g_az, *g_m;
            bool* g_isdevice;
            int* g_devices;

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
            CUDA_CALL(cudaMalloc(&g_isdevice, n * sizeof(bool)));
            CUDA_CALL(cudaMalloc(&g_devices, device_cnt * sizeof(int)));

            CUDA_CALL(cudaMemcpyAsync(g_qx, qx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_qy, qy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_qz, qz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vx, vx0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vy, vy0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_vz, vz0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_m, m0, n * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(g_devices, devices, device_cnt * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
            clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[1]>>>(n, g_ax, g_ay, g_az);
            clear_array_gpu<<<GridDim(n), BlockDim, 0, streams[0]>>>(n, g_isdevice);
            set_isdevice_gpu<<<GridDim(n), BlockDim, 0, streams[0]>>>(device_cnt, g_devices, g_isdevice);
            qx = (double*)malloc(n * sizeof(double));
            qy = (double*)malloc(n * sizeof(double));
            qz = (double*)malloc(n * sizeof(double));

            int d = devices[di];
            bool hit = false;
            bool destroyed = (m0[d] == 0);
            double cost = std::numeric_limits<double>::infinity();

            for (int step = 0; step <= param::n_steps && !hit; step++) {
                double dx, qx_p, qx_a, qx_d;
                double dy, qy_p, qy_a, qy_d;
                double dz, qz_p, qz_a, qz_d;
                if (step == 0) {
                    qx_p = qx0[planet];
                    qy_p = qy0[planet];
                    qz_p = qz0[planet];
                    qx_a = qx0[asteroid];
                    qy_a = qy0[asteroid];
                    qz_a = qz0[asteroid];
                } else {
                    cudaStreamSynchronize(streams[1]);
                    compute_accelerations_gpu<<<GridDim(n * n), BlockDim, 0, streams[0]>>>(false, step, n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz, g_ax, g_ay, g_az, g_m, g_isdevice);
                    update_velocities_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_vx, g_vy, g_vz, g_ax, g_ay, g_az);
                    cudaStreamSynchronize(streams[0]);
                    clear_a_gpu<<<GridDim(3 * n), BlockDim, 0, streams[1]>>>(n, g_ax, g_ay, g_az);
                    update_positions_gpu<<<GridDim(3 * n), BlockDim, 0, streams[0]>>>(n, g_qx, g_qy, g_qz, g_vx, g_vy, g_vz);
                    CUDA_CALL(cudaMemcpyAsync(qx, g_qx, n * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
                    CUDA_CALL(cudaMemcpyAsync(qy, g_qy, n * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
                    CUDA_CALL(cudaMemcpyAsync(qz, g_qz, n * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
                    CUDA_CALL(cudaStreamSynchronize(streams[0]));
                    qx_p = qx[planet];
                    qy_p = qy[planet];
                    qz_p = qz[planet];
                    qx_a = qx[asteroid];
                    qy_a = qy[asteroid];
                    qz_a = qz[asteroid];
                }
                dx = qx_p - qx_a;
                dy = qy_p - qy_a;
                dz = qz_p - qz_a;
                if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                    hit = true;
                    break;
                }
                if (!destroyed) {
                    if (step == 0) {
                        qx_d = qx0[d];
                        qy_d = qy0[d];
                        qz_d = qz0[d];
                    } else {
                        qx_d = qx[d];
                        qy_d = qy[d];
                        qz_d = qz[d];
                    }
                    dx = qx_p - qx_d;
                    dy = qy_p - qy_d;
                    dz = qz_p - qz_d;
                    double missle_dist = param::missile_speed * step * param::dt;
                    if (dx * dx + dy * dy + dz * dz < missle_dist * missle_dist) {
                        destroyed = true;
                        cost = param::get_missile_cost((step + 1) * param::dt);
                        CUDA_CALL(cudaMemsetAsync(g_m + d, 0, sizeof(double), streams[0]));
                    }
                }
            }
#pragma omp critical
            if (!hit && cost < missile_cost) {
                gravity_device_id = d;
                missile_cost = cost;
            }
            for (int i = 0; i < param::cuda_nstreams; i++) {
                cudaStreamDestroy(streams[i]);
            }
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
            CUDA_CALL(cudaFree(g_isdevice));
            CUDA_CALL(cudaFree(g_devices));
        }
        if (gravity_device_id == -1) {
            missile_cost = 0;
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
    free(devices);
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