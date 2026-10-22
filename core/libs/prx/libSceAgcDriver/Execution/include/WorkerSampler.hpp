#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WORKERSAMPLER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WORKERSAMPLER_HPP

namespace AgcDriver {

// Starts sampling the calling thread when APS5_SAMPLE_WORKER names an output file.
void StartWorkerSampler();
// Starts sampling the calling thread (the draw committer) when APS5_SAMPLE_COMMITTER names a file.
void StartCommitterSampler();
void StopWorkerSampler();

}

#endif
