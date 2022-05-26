#define DEBUG

#include <omp.h>
#include <pthread.h>

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
#else
#define __debug_printf(fmt, args...)
#define __START_TIME(ID)
#define __END_TIME(ID)
#endif

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
                std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
                std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                std::vector<double>& m, std::vector<bool>& isdevice, std::vector<int>& devices) {
    std::ifstream fin(filename);
    std::string type;
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    isdevice.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type;
        if (type == "device") {
            isdevice[i] = true;
            devices.push_back(i);
        }
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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
              std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
              std::vector<double>& vz, const std::vector<double>& m,
              const std::vector<bool>& isdevice) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i)
                continue;
            double mj = m[j];
            if (isdevice[j]) {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx0, qy0, qz0, vx0, vy0, vz0, m0;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<bool> isdevice;
    std::vector<int> devices;
    read_input(argv[1], n, planet, asteroid, qx0, qy0, qz0, vx0, vy0, vz0, m0, isdevice, devices);

    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    int gravity_device_id = -1;
    double missile_cost = 0;

    // Problem 1
    min_dist = std::numeric_limits<double>::infinity();
    qx = qx0, qy = qy0, qz = qz0, vx = vx0, vy = vy0, vz = vz0, m = m0;
    for (int i = 0; i < n; i++) {
        if (isdevice[i]) {
            m[i] = 0;
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, isdevice);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Problem 2
    hit_time_step = -2;
    qx = qx0, qy = qy0, qz = qz0, vx = vx0, vy = vy0, vz = vz0, m = m0;
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, isdevice);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3
    if (hit_time_step != -2) {
        gravity_device_id = -1;
        missile_cost = std::numeric_limits<double>::infinity();
        for (int d : devices) {
            qx = qx0, qy = qy0, qz = qz0, vx = vx0, vy = vy0, vz = vz0, m = m0;
            bool hit = false;
            double cost = std::numeric_limits<double>::infinity();
            for (int step = 0; step <= param::n_steps && !hit; step++) {
                if (step > 0) {
                    run_step(step, n, qx, qy, qz, vx, vy, vz, m, isdevice);
                }
                double dx = qx[planet] - qx[asteroid];
                double dy = qy[planet] - qy[asteroid];
                double dz = qz[planet] - qz[asteroid];
                if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                    hit = true;
                }
                if (m[d] != 0.0) {
                    dx = qx[planet] - qx[d];
                    dy = qy[planet] - qy[d];
                    dz = qz[planet] - qz[d];
                    double missle_dist = param::missile_speed * step * param::dt;
                    if (dx * dx + dy * dy + dz * dz < missle_dist * missle_dist) {
                        m[d] = 0.0;
                        cost = param::get_missile_cost((step + 1) * param::dt);
                    }
                }
            }
            if (!hit && cost < missile_cost) {
                gravity_device_id = d;
                missile_cost = cost;
            }
        }
        if (gravity_device_id == -1) {
            missile_cost = 0;
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
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
 */