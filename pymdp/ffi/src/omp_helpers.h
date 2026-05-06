// OpenMP threshold and team-size helpers shared by CPU FFI kernels.
//
// Header-only; mark helpers `inline` so multiple includers don't ODR-clash.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include <omp.h>

#include "cpu_capabilities.h"

#if defined(__linux__)
#include <dirent.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace pymdp_ffi {

// OMP fork/barrier threshold: below this, libomp overhead dominates.
// 32 on fast cores (M3/Apple/Cortex-A76+/x86); 8 on ARMv8.0 (Cortex-A57).
// Keyed by PYMDP_FFI_HAS_F16_FML; -DPYMDP_FFI_DISABLE_F16=1 forces v8.0 path.
#if PYMDP_FFI_HAS_AARCH64_NEON && !PYMDP_FFI_HAS_F16_FML
constexpr int64_t kOmpNodeThreshold = 8;
#else
constexpr int64_t kOmpNodeThreshold = 32;
#endif

// True if name matches cpuN convention (filters cpufreq, cpuidle, etc).
inline bool is_linux_cpu_dir_name(const char* name) {
  if (name[0] != 'c' || name[1] != 'p' || name[2] != 'u') return false;
  if (name[3] == '\0') return false;
  for (const char* p = name + 3; *p; ++p) {
    if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
  }
  return true;
}

inline bool read_first_token(const std::string& path, std::string& out) {
  std::ifstream f(path);
  return static_cast<bool>(f >> out);
}

// Returns host's physical core count, or -1 if unavailable.
// macOS: sysctl hw.physicalcpu. Linux: distinct topology groups.
inline int detect_physical_cores() {
#if defined(__APPLE__) && defined(__MACH__)
  int    value = 0;
  size_t size  = sizeof(value);
  if (sysctlbyname("hw.physicalcpu", &value, &size, nullptr, 0) == 0 && size == sizeof(value) && value > 0) {
    return value;
  }
  return -1;
#elif defined(__linux__)
  const std::string cpu_root = "/sys/devices/system/cpu";
  DIR*              dir      = opendir(cpu_root.c_str());
  if (!dir) return -1;
  std::set<std::string> physical_core_groups;
  while (dirent* ent = readdir(dir)) {
    if (!is_linux_cpu_dir_name(ent->d_name)) continue;
    const std::string topology = cpu_root + "/" + ent->d_name + "/topology/";
    std::string       group;
    if (read_first_token(topology + "core_cpus_list", group) ||
        read_first_token(topology + "thread_siblings_list", group)) {
      physical_core_groups.insert(group);
    }
  }
  closedir(dir);
  return physical_core_groups.empty() ? -1 : static_cast<int>(physical_core_groups.size());
#else
  return -1;
#endif
}

// Recommended OMP team size for FFI parallel regions.
// Capped to physical cores (avoids SMT pessimization on HT hosts).
// Override: if OMP_NUM_THREADS is set, respect the user's choice.
inline int omp_team_size() {
  static const int team = []() {
    const int   max_threads = omp_get_max_threads();
    const char* env         = std::getenv("OMP_NUM_THREADS");
    if (env != nullptr && env[0] != '\0') return max_threads;
    const int physical = detect_physical_cores();
    if (physical <= 0) return max_threads;
    return std::min(max_threads, physical);
  }();
  return team;
}

// Cap OpenMP team to number of work units (avoid idle workers on small calls).
inline int omp_team_size_for_work_units(int64_t work_units) {
  if (work_units <= 0) return 1;
  return static_cast<int>(std::min<int64_t>(work_units, omp_team_size()));
}

// FPI batch parallelism gate: batch >= kOmpNodeThreshold, or
// batch >= 2 && num_iter * total_ll >= kFpiBatchWorkThreshold.
constexpr int64_t kFpiBatchWorkThreshold = 100000;

// Neg-EFE per-level OMP gate: ~50K FMAs (fork/barrier breakeven for 12-thread team).
constexpr int64_t kNegEfeLevelOmpFlopThreshold = 50000;

inline bool should_parallelize_fpi_batch(int64_t batch, int64_t num_iter, int64_t total_ll) {
  return (batch >= kOmpNodeThreshold) || (batch >= 2 && num_iter * total_ll >= kFpiBatchWorkThreshold);
}

}  // namespace pymdp_ffi
