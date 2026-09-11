#pragma once

#include <cstddef>
#include <vector>
#include <thread>
#include <algorithm>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:

public:
  Grid(std::size_t rows, std::size_t cols);

  double& operator()(std::size_t i, std::size_t j);
  double  operator()(std::size_t i, std::size_t j) const;

  std::size_t cols_;
  std::size_t rows_;

   // Represent 2D values as a flat 1D vector
  std::vector<double> temps_;
  std::size_t num_threads_;
  std::size_t interior_rows_;
  std::size_t base_interval_;
  std::size_t extra;
}; 

int additional_row(std::size_t& extra_rows) {
  if (extra_rows > 0) {
    extra_rows--;
    return 1;
  }
  return 0;
}

void update_grid(std::size_t start_row, std::size_t end_row, const Grid& old_grid, Grid& new_grid) {
  for (std::size_t i = start_row; i < end_row; ++i) {
        for (std::size_t j = 1; j < old_grid.cols_ - 1; ++j) {

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

  // Top and bottom rows
  for (std::size_t j = 0; j < old_grid.cols_; ++j) {
    new_grid(0, j) = old_grid(0, j);
    new_grid(old_grid.rows_ - 1, j) =
        old_grid(old_grid.rows_ - 1, j);
  }

// Left and right columns
  for (std::size_t i = 1; i < old_grid.rows_ - 1; ++i) {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, old_grid.cols_ - 1) =
        old_grid(i, old_grid.cols_ - 1);
  }
  std::size_t current_row = 1;
  std::size_t extra_rows = old_grid.extra;
  std::vector<std::thread> threads;

  for (std::size_t t = 0; t < old_grid.num_threads_; ++t) {

      std::size_t rows_for_thread = old_grid.base_interval_ + additional_row(extra_rows);
      std::size_t start_row = current_row;
      std::size_t end_row = start_row + rows_for_thread;
      threads.emplace_back(update_grid, start_row, end_row, std::cref(old_grid), std::ref(new_grid));

      current_row = end_row;

  }
  for (auto& thread : threads) {
    thread.join();
  }
}

Grid::Grid(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), temps_(rows * cols, 0.0), interior_rows_(rows > 2 ? rows - 2 : 0) {

      num_threads_ = std::min(static_cast<std::size_t>(std::thread::hardware_concurrency()), interior_rows_);
      if (num_threads_ == 0)
        num_threads_ = 1;
      num_threads_ = 1;

      base_interval_ = interior_rows_ / num_threads_;
      extra = interior_rows_ % num_threads_;

     }

double& Grid::operator()(std::size_t i, std::size_t j) {
  return temps_[i * cols_ + j];
}

double Grid::operator()(std::size_t i, std::size_t j) const {
  return temps_[i * cols_ + j];
}