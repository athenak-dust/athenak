//========================================================================================
// AthenaK: composite AMR self-gravity. BSD-3-Clause; see LICENSE.
//========================================================================================
#ifndef GRAVITY_HOST_WORKERS_HPP_
#define GRAVITY_HOST_WORKERS_HPP_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gravity {

// Host worker threads kept between the parallel loops of the gravity setup: started when a
// loop first needs them and idle on a condition variable between loops, instead of a thread
// start and join for every loop.  Every thread that runs such loops has its own workers
// (ThreadWorkers), which end with that thread; one loop at a time per owner.
class HostWorkers {
 public:
  HostWorkers() = default;
  HostWorkers(const HostWorkers &) = delete;
  HostWorkers &operator=(const HostWorkers &) = delete;
  ~HostWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto &thread : threads_) thread.join();
  }
  // body(w) for w = 0..n-1: w = 0 on the calling thread, the others on the workers;
  // returns when all have returned (body must not throw)
  void Run(int n, const std::function<void(int)> &body) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (static_cast<int>(threads_.size()) < n-1) {
      const int id = static_cast<int>(threads_.size())+1;
      threads_.emplace_back([this,id]() { Loop(id); });
    }
    body_ = &body;
    active_ = n;
    pending_ = n-1;
    ++generation_;
    lock.unlock();
    wake_.notify_all();
    body(0);
    lock.lock();
    done_.wait(lock,[this]() { return pending_ == 0; });
    body_ = nullptr;
  }

 private:
  void Loop(int id) {
    std::uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      wake_.wait(lock,[&]() { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      if (id >= active_) continue;
      const auto *body = body_;
      lock.unlock();
      (*body)(id);
      lock.lock();
      if (--pending_ == 0) done_.notify_one();
    }
  }
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> threads_;
  const std::function<void(int)> *body_ = nullptr;
  int active_ = 0, pending_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;
};

// the workers of the calling thread
inline HostWorkers &ThreadWorkers() {
  thread_local HostWorkers workers;
  return workers;
}

// task(c) for c = 0..chunks-1 on up to `workers` host threads (the calling thread
// included), the chunks taken in any order; the first exception is rethrown
template <typename Task>
void ParallelChunks(int workers, int chunks, const Task &task) {
  workers = std::max(1,std::min(workers,chunks));
  if (workers == 1) {
    for (int c = 0; c < chunks; ++c) task(c);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::exception_ptr> errors(workers);
  const std::function<void(int)> run = [&](int w) {
    try {
      for (int c = next++; c < chunks; c = next++) task(c);
    } catch (...) { errors[w] = std::current_exception(); next = chunks; }
  };
  ThreadWorkers().Run(workers,run);
  for (const auto &error : errors) if (error) std::rethrow_exception(error);
}

}  // namespace gravity
#endif  // GRAVITY_HOST_WORKERS_HPP_
