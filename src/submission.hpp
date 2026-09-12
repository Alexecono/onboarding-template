#pragma once

#include <cstddef>
#include <vector>
#include <thread>
#include <algorithm>
#include <mutex>
#include <condition_variable>

struct Worker {
  std::size_t start_row;
  std::size_t end_row;
  std::thread thread;
};

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:

std::size_t cols_;
std::size_t rows_;

std::size_t num_threads_;
std::size_t interior_rows_;
std::size_t base_interval_;
std::size_t extra;

// Represent 2D values as a flat 1D vector
std::vector<double> temps_;

std::vector<Worker> workers_;

std::mutex mutex_;
std::condition_variable wake_cv_;
std::condition_variable done_cv_;

const Grid* old_grid_ = nullptr;
Grid* new_grid_ = nullptr;

std::size_t iteration_ = 0;
std::size_t finished_workers_ = 0;
bool stop_ = false;


public:
  Grid(std::size_t rows, std::size_t cols);
  ~Grid();

  double& operator()(std::size_t i, std::size_t j);
  double  operator()(std::size_t i, std::size_t j) const;

  std::size_t get_rows() const { return rows_; }
  std::size_t get_cols() const { return cols_; }
  std::size_t get_iterations() const { return iteration_; }

  void activate_worker(std::size_t start_row, std::size_t end_row);
  void start_iteration(const Grid& old_grid);
  void wait_for_workers();

}; 

int additional_row(std::size_t& extra_rows) {
  if (extra_rows > 0) {
    extra_rows--;
    return 1;
  }
  return 0;
}

void update_grid(
    std::size_t start_row,
    std::size_t end_row,
    const Grid& old_grid,
    Grid& new_grid
);

void Grid::activate_worker(std::size_t start_row, std::size_t end_row) {
  std::size_t my_iteration = 0;

  while (true) {

    // Automatic unlocking of mutex when lock goes out of scope
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wakes up from notify_all(),
    // Checks if there is work to be done (or destructor is called),
    // lock if needed.

    // If new work: 
    //   update my_iteration and grids, unlock, do expensive calculation concurrently

    // If no new work (worker finished one iteration, others are still working): 
    //   Don't start a new calculation again, go to sleep, unlock, and wait for notify_all() again.
    //   Needs the mutex because it is possible to see no work to do,
    //   but before going to sleep, apply_stencil might have started a new iteration,
    //   and then sleeps, missing an iteration.
    wake_cv_.wait(lock, [&] {
      return iteration_ > my_iteration || stop_;
    });

    if (stop_)
        break;

    my_iteration = iteration_;

    const Grid* old = old_grid_;
    Grid* next = new_grid_;

    lock.unlock();

    update_grid(start_row, end_row, *old, *next);

    lock.lock();
    finished_workers_++;
    if (finished_workers_ == num_threads_)
      done_cv_.notify_one();
  }
}

void Grid::start_iteration(const Grid& old_grid) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    old_grid_ = &old_grid;
    new_grid_ = this;
    finished_workers_ = 0;
    iteration_++;
  }

  // Wake all persistent workers
  wake_cv_.notify_all();
}

void Grid::wait_for_workers() {
  std::unique_lock<std::mutex> lock(mutex_);

  done_cv_.wait(lock, [&] {
    return finished_workers_ == num_threads_;
  });
}

void update_grid(std::size_t start_row, std::size_t end_row, const Grid& old_grid, Grid& new_grid) {
  for (std::size_t i = start_row; i < end_row; ++i) {
        for (std::size_t j = 1; j < old_grid.get_cols() - 1; ++j) {

            new_grid(i, j) =
                0.5 * old_grid(i, j) +
                0.125 * (
                    old_grid(i - 1, j) +
                    old_grid(i + 1, j) +
                    old_grid(i, j - 1) +
                    old_grid(i, j + 1)
                );
        }
    }
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid){

  if (new_grid.get_iterations() == 0) {
    // Top and bottom rows
    for (std::size_t j = 0; j < old_grid.get_cols(); ++j) {
    new_grid(0, j) = old_grid(0, j);
    new_grid(old_grid.get_rows() - 1, j) =
        old_grid(old_grid.get_rows() - 1, j);
    }

    // Left and right columns
    for (std::size_t i = 1; i < old_grid.get_rows() - 1; ++i) {
      new_grid(i, 0) = old_grid(i, 0);
      new_grid(i, old_grid.get_cols() - 1) =
        old_grid(i, old_grid.get_cols() - 1);
    }
  }
   
  new_grid.start_iteration(old_grid);
  new_grid.wait_for_workers();
}

Grid::Grid(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), temps_(rows * cols, 0.0), interior_rows_(rows > 2 ? rows - 2 : 0) {

      num_threads_ = std::min(static_cast<std::size_t>(std::thread::hardware_concurrency()), interior_rows_);
      if (num_threads_ == 0)
        num_threads_ = 1;
      num_threads_ = 2;

      base_interval_ = interior_rows_ / num_threads_;
      extra = interior_rows_ % num_threads_;

      workers_.resize(num_threads_);

      std::size_t current_row = 1;

      for (std::size_t t = 0; t < num_threads_; ++t) {

        std::size_t rows_for_thread = base_interval_ + additional_row(extra);
        workers_[t].start_row = current_row;
        workers_[t].end_row = workers_[t].start_row + rows_for_thread;

        workers_[t].thread = std::thread(
            &Grid::activate_worker,
            this,
            workers_[t].start_row,
            workers_[t].end_row
        );

        current_row = workers_[t].end_row;

      }

     }

Grid::~Grid() {
  // Tell every worker to stop
  {
    std::lock_guard<std::mutex> lock(mutex_);

    stop_ = true;
  }

  // Wake workers that are sleeping on cv_
  wake_cv_.notify_all();

  // Wait for every thread to terminate
  for (auto& worker : workers_) {
    if (worker.thread.joinable()) {
      worker.thread.join();
    }
  }
}

double& Grid::operator()(std::size_t i, std::size_t j) {
  return temps_[i * cols_ + j];
}

double Grid::operator()(std::size_t i, std::size_t j) const {
  return temps_[i * cols_ + j];
}