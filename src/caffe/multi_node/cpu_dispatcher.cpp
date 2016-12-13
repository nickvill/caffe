
#include <string>
#include <vector>
#include "caffe/multi_node/cpu_dispatcher.hpp"


namespace caffe {

void CPUDispatcher::Dispatch(vector<vector<int> > *pthread_arr, int num_threads) {
  ParseCPUInfo();

#if (defined USE_MKL) && (defined _OPENMP)
  int omp_threads = omp_get_max_threads();

  CHECK_LE(omp_threads * num_threads, num_cores_)
                << "too many threads to be scheduled";

  CHECK_GE(num_threads, 1)
                << "at least 1 workers";

  pthread_arr->clear();
  // reserve the last slot for free cores
  pthread_arr->resize(num_threads + 1);

  if (num_threads == 1) {
    // use all the cores if only 1 thread
    for (int i = 0; i < omp_threads; i++) {
      pthread_arr->at(0).push_back(i);
    }

    for (int i = omp_threads; i < num_cores_; i++) {
      pthread_arr->at(1).push_back(i);
    }
  } else {
    // evenly distribute the threads to each socket
    int thread_per_socket = num_threads / num_sockets_;
    CHECK_GE(cores_per_socket_, thread_per_socket * omp_threads)
          << "too many threads to be scheduled";

    vector<int> thrd_num_arr;
    thrd_num_arr.resize(num_sockets_);
    for (int i = 0; i < thrd_num_arr.size(); i++) {
      thrd_num_arr[i] = thread_per_socket;
    }
    int remain_threads = num_threads - thread_per_socket * num_sockets_;

    for (int i = 0; i < remain_threads; i++) {
      thrd_num_arr[i]++;
    }

    int thrd_idx = 0;
    for (int i = 0; i < num_sockets_; i++) {
      for (int j = 0; j < thrd_num_arr[i]; j++) {
        int core_idx = i * cores_per_socket_ + j * omp_threads;

        for (int k = 0; k < omp_threads; k++) {
          pthread_arr->at(thrd_idx).push_back(core_idx + k);
        }
        thrd_idx++;
      }

      // push the free cores to free list
      for (int j = omp_threads * thrd_num_arr[i]; j < cores_per_socket_; j++) {
        int core_idx = i * cores_per_socket_ + j;
        pthread_arr->at(num_threads).push_back(core_idx);
      }
    }  // end for
  }

#else
  LOG(ERROR) << "cpu dispatcher only works when MKL and OMP are enabled";
#endif
}

void CPUDispatcher::ParseCPUInfo() {
  cores_per_socket_ = NodeEnv::Instance()->cores_per_sock();
  num_sockets_ = NodeEnv::Instance()->num_sockets();

  // TODO: parse from cpu info
  num_cores_ = num_sockets_ * cores_per_socket_;
}

}  // end namespace caffe


