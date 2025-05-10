#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <limits>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>

#ifdef _MSC_VER
#pragma optimize("gt", on)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3,unroll-loops,omit-frame-pointer,inline")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#endif

std::mutex file_mutex;
std::mutex console_mutex;

class Body
{
public:
    long double mass;
    std::array<long double, 2> position;
    std::array<long double, 2> velocity;

    Body(long double m, std::array<long double, 2> pos, std::array<long double, 2> vel)
        : mass(m), position(pos), velocity(vel) {}
    std::array<long double, 2> compute_acceleration(const std::vector<Body> &other_bodies, long double G = 1.0, long double softening = 0.2) const
    {
        std::array<long double, 2> acceleration = {0.0, 0.0};
        for (const auto &other : other_bodies)
        {
            if (&other != this)
            {
                std::array<long double, 2> r = {other.position[0] - position[0], other.position[1] - position[1]};
                long double distance = std::sqrtl(r[0] * r[0] + r[1] * r[1]);
                long double factor = G * other.mass / std::max(std::powl(distance, 3), std::powl(softening, 3));
                acceleration[0] += factor * r[0];
                acceleration[1] += factor * r[1];
            }
        }
        return acceleration;
    }
    std::array<long double, 4> get_state() const
    {
        return {position[0], position[1], velocity[0], velocity[1]};
    }
};

class System
{
private:
    long double G;
    std::vector<Body> bodies;

public:
    System(long double g = 1.0) : G(g) {}
    void set_state(const std::vector<long double> &state)
    {
        bodies.clear();
        for (size_t i = 0; i < state.size(); i += 4)
        {
            bodies.emplace_back(1.0,
                                std::array<long double, 2>{state[i], state[i + 1]},
                                std::array<long double, 2>{state[i + 2], state[i + 3]});
        }
    }
    std::vector<std::array<long double, 2>> compute_accelerations()
    {
        std::vector<std::array<long double, 2>> accelerations;
        accelerations.reserve(bodies.size());
        for (const auto &body : bodies)
        {
            accelerations.push_back(body.compute_acceleration(bodies, G));
        }
        return accelerations;
    }
    long double compute_total_energy() const
    {
        long double kinetic_energy = 0.0;
        long double potential_energy = 0.0;

        for (const auto &body : bodies)
        {
            kinetic_energy += 0.5 * body.mass * (body.velocity[0] * body.velocity[0] + body.velocity[1] * body.velocity[1]);
        }

        for (size_t i = 0; i < bodies.size(); ++i)
        {
            for (size_t j = i + 1; j < bodies.size(); ++j)
            {
                long double dx = bodies[j].position[0] - bodies[i].position[0];
                long double dy = bodies[j].position[1] - bodies[i].position[1];
                long double distance = std::sqrtl(dx * dx + dy * dy);
                potential_energy -= G * bodies[i].mass * bodies[j].mass / distance;
            }
        }

        return kinetic_energy + potential_energy;
    }
    std::vector<long double> get_state() const
    {
        std::vector<long double> state;
        state.reserve(bodies.size() * 4);
        for (const auto &body : bodies)
        {
            auto body_state = body.get_state();
            state.insert(state.end(), body_state.begin(), body_state.end());
        }
        return state;
    }

    std::vector<long double> integrate(long double dt, int num_steps, bool save_positions = false, bool save_energy = false)
    {
        if (save_positions)
        {
            std::vector<std::vector<std::array<long double, 2>>> all_positions(num_steps, std::vector<std::array<long double, 2>>(bodies.size()));

            for (int step = 0; step < num_steps; ++step)
            {
                auto accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    for (int j = 0; j < 2; ++j)
                    {
                        bodies[i].position[j] += bodies[i].velocity[j] * dt + 0.5 * accelerations[i][j] * dt * dt;
                    }
                }

                auto new_accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    for (int j = 0; j < 2; ++j)
                    {
                        bodies[i].velocity[j] += 0.5 * (accelerations[i][j] + new_accelerations[i][j]) * dt;
                    }
                    all_positions[step][i] = {bodies[i].position[0], bodies[i].position[1]};
                }
            }

            std::ofstream outfile("data/positions.csv");
            if (!outfile)
            {
                std::cerr << "Failed to open positions.csv for writing" << std::endl;
                return {};
            }

            for (int frame = 0; frame < num_steps; ++frame)
            {
                for (size_t body_idx = 0; body_idx < bodies.size(); ++body_idx)
                {
                    outfile << all_positions[frame][body_idx][0] << "," << all_positions[frame][body_idx][1] << std::endl;
                }
                outfile << std::endl;
            }

            return {};
        }
        else
        {
            for (int step = 0; step < num_steps; ++step)
            {
                auto accelerations = compute_accelerations();

                for (auto &body : bodies)
                {
                    for (int j = 0; j < 2; ++j)
                    {
                        body.position[j] += body.velocity[j] * dt + 0.5 * accelerations[&body - &bodies[0]][j] * dt * dt;
                    }
                }

                auto new_accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    for (int j = 0; j < 2; ++j)
                    {
                        bodies[i].velocity[j] += 0.5 * (accelerations[i][j] + new_accelerations[i][j]) * dt;
                    }
                }
            }

            return get_state();
        }
    }

    long double proximity(const std::vector<long double> &stan_state, long double total_time = 30, long double delta_t = 0.01)
    {
        set_state(stan_state);
        std::vector<long double> initial_state = get_state();
        long double min_proximity = std::numeric_limits<long double>::max();
        long double di = 0.0;

        const int steps = static_cast<int>(total_time / delta_t);

        for (int step = 0; step < steps; ++step)
        {
            std::vector<long double> new_state = integrate(delta_t, 1);

            long double df = 0.0;
            for (size_t i = 0; i < new_state.size(); ++i)
            {
                long double diff = new_state[i] - initial_state[i];
                df += diff * diff;
            }
            df = std::sqrtl(df);

            if (df < di && df < min_proximity)
            {
                min_proximity = df;
            }

            di = df;
        }

        if (min_proximity == std::numeric_limits<long double>::max())
        {
            min_proximity = di;
        }

        return min_proximity;
    }

    int closest_step(const std::vector<long double> &stan_state, long double total_time = 30, long double delta_t = 0.01)
    {
        set_state(stan_state);
        std::vector<long double> initial_state = get_state();
        long double min_proximity = std::numeric_limits<long double>::max();
        int min_step = 0;
        long double di = 0.0;

        const int steps = static_cast<int>(total_time / delta_t);

        for (int step = 0; step < steps; ++step)
        {
            std::vector<long double> new_state = integrate(delta_t, 1);

            long double df = 0.0;
            for (size_t i = 0; i < new_state.size(); ++i)
            {
                long double diff = new_state[i] - initial_state[i];
                df += diff * diff;
            }
            df = std::sqrtl(df);

            if (df < di && df < min_proximity)
            {
                min_proximity = df;
                min_step = step;
            }

            di = df;
        }

        return min_step;
    }
};

std::vector<long double> make_state(long double d, long double vx, long double vy)
{
    return {0, 0, vx, vy, d, 0, -vx / 2, -vy / 2, -d, 0, -vx / 2, -vy / 2};
}

void heatmap_batch(System &system, long double (*quantifier)(System &, const std::vector<long double> &, long double, long double),
                   int start_j, int end_j, int grid_size, long double vx_start, long double vy_start,
                   long double vx_step, long double vy_step, const std::string &outfile_path)
{

    std::vector<std::string> results;
    long double d = 1.0;

    auto start_time = std::chrono::high_resolution_clock::now();
    int current_iteration = 0;
    int total_iterations = (end_j - start_j) * grid_size;

    for (int j = start_j; j < end_j; ++j)
    {
        long double vy = vy_start + j * vy_step;

        for (int i = 0; i < grid_size; ++i)
        {
            long double vx = vx_start + i * vx_step;

            std::vector<long double> initial_state = make_state(d, vx, vy);
            long double proximity = quantifier(system, initial_state, 30, 0.01);

            std::ostringstream oss;
            oss << "1," << vy << "," << vx << "," << proximity;
            results.push_back(oss.str());

            ++current_iteration;
            if (current_iteration % 10 == 0)
            {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                long double progress = static_cast<long double>(current_iteration) / total_iterations;
                int eta = progress > 0 ? static_cast<int>((elapsed / progress) - elapsed) : 0;

                // Use a mutex to synchronize console output
                std::lock_guard<std::mutex> console_lock(console_mutex);
                std::cout << "\rBatch " << start_j << "-" << end_j << ": ["
                          << std::string(static_cast<int>(20 * progress), '#')
                          << std::string(20 - static_cast<int>(20 * progress), ' ')
                          << "] " << std::fixed << std::setprecision(1) << (progress * 100)
                          << "% | " << elapsed << "s/" << eta << "s     " << std::flush;
            }
        }
    }

    std::lock_guard<std::mutex> lock(file_mutex);
    std::ofstream outfile(outfile_path, std::ios::app);
    for (const auto &result : results)
    {
        outfile << result << "\n";
    }
}

void heatmap(System &system, long double (*quantifier)(System &, const std::vector<long double> &, long double, long double),
             long double vx_start, long double vx_end, long double vy_start, long double vy_end, int dimension)
{

    const int grid_size = dimension;
    const long double vx_step = (vx_end - vx_start) / (grid_size - 1);
    const long double vy_step = (vy_end - vy_start) / (grid_size - 1);

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    std::string outfile_path = "data/zoom.csv";

    std::ofstream outfile(outfile_path);
    outfile << "d,vy,vx,proximity\n";
    outfile.close();

    for (int thread_id = 0; thread_id < num_threads; ++thread_id)
    {
        int start_j = thread_id * grid_size / num_threads;
        int end_j = (thread_id + 1) * grid_size / num_threads;

        threads.emplace_back([&, start_j, end_j]()
                             {
            System local_system;
            heatmap_batch(local_system, quantifier, start_j, end_j, grid_size, 
                          vx_start, vy_start, vx_step, vy_step, outfile_path); });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    std::cout << "\nHeatmap calculation complete. Data saved to " << outfile_path << std::endl;
}

long double proximity_wrapper(System &system, const std::vector<long double> &initial_state, long double total_time, long double delta_t)
{
    return system.proximity(initial_state, total_time, delta_t);
}

int main()
{
    System system;
    long double vx_start, vx_end, vy_start, vy_end;
    int dimension;
    std::string mode;

    std::string input;
    std::cout << "What are we doing today? 'positions', 'energy' or 'normal'? ";
    std::cin >> mode;
    std::cin.ignore();

#ifdef _WIN32
    std::system("if not exist data mkdir data");
#else
    std::system("mkdir -p data");
#endif

    if (mode == "positions")
    {
        long double d, vx, vy;
        std::cout << "Enter d, vx, vy: ";
        std::cin >> d >> vx >> vy;
        std::vector<long double> initial_state = make_state(d, vx, vy);
        system.set_state(initial_state);
        system.integrate(0.01, 1000, true);
        std::cout << "Position data saved to data/positions.csv" << std::endl;
    }
    else
    {
        std::cout << "Enter vx start, vx end, vy start, vy end: ";
        std::getline(std::cin, input);
        std::cout << "Enter the dimension of grid (1 second per 100 pixels): ";
        std::cin >> dimension;

        std::istringstream iss(input);
        if (iss >> vx_start >> vx_end >> vy_start >> vy_end)
        {
            std::cout << "vx range: [" << vx_start << ", " << vx_end << "]\n";
            std::cout << "vy range: [" << vy_start << ", " << vy_end << "]\n";
            std::cout << "Resolution: " << dimension << "x" << dimension << " grid\n";
        }
        else
        {
            std::cerr << "Error: Invalid input format." << std::endl;
            return 1;
        }

        std::cout << "Calculating proximity heatmap...\n";
        heatmap(system, proximity_wrapper, vx_start, vx_end, vy_start, vy_end, dimension);
    }

    return 0;
}
