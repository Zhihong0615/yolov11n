// C++ KV260 three-stage BGR -> final-box diagnostic with two VART runners.
// Both target the same single continuous DPU graph on one physical DPU CU.
// No frame skip or host GPU. Audit outputs, traces and sensors independently.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <vart/experimental/runner_helper.hpp>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>

extern "C" {
struct LetterboxResult {
  double gain;
  int left, top, resized_width, resized_height;
  uint64_t resize_ns, pad_ns, convert_ns;
};
void* kv260_lb_create(int fix_point) noexcept;
void kv260_lb_destroy(void* context) noexcept;
int kv260_lb_prepare(void* context, const uint8_t* bgr, int width, int height,
                     int stride, int8_t* output, LetterboxResult* result) noexcept;
struct FastDecodeSixResult {
  float rows[300 * 6];
  int count;
  int stats[6];
};
int kv260_decode_six(const int8_t* const class_maps[3],
                     const int8_t* const box_maps[3],
                     const int class_fix[3], const int box_fix[3],
                     int clip8, float confidence, int image_height,
                     int image_width, double gain, int left, int top,
                     FastDecodeSixResult* out) noexcept;
int kv260_decode_six_offsets(const int8_t* const class_maps[3],
                             const int8_t* const box_maps[3],
                             const int class_fix[3], const int box_fix[3],
                             int clip8, float confidence, int image_height,
                             int image_width, double gain, int left, int top,
                             const float score_logit_offsets[6],
                             FastDecodeSixResult* out) noexcept;
}

namespace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using BufferList = std::vector<std::unique_ptr<vart::TensorBuffer>>;

template <typename T> class Queue {
 public:
  void push(T value) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      data_.push_back(std::move(value));
    }
    condition_.notify_one();
  }
  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return !data_.empty(); });
    T value = std::move(data_.front());
    data_.pop_front();
    return value;
  }
 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<T> data_;
};

struct Job {
  int input_slot = -1;
  int output_slot = -1;
  LetterboxResult geometry{};
  FastDecodeSixResult detection{};
  Time submitted, admitted, pre_start, pre_end, dpu_start, dpu_end, post_start, delivered;
};

double ms(Time a, Time b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
int64_t ns(Time value) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      value.time_since_epoch()).count();
}
double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}
double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double index = fraction * (values.size() - 1);
  const auto lower = static_cast<size_t>(index);
  const auto upper = std::min(lower + 1, values.size() - 1);
  return values[lower] * (upper - index) + values[upper] * (index - lower);
}
std::vector<cv::Mat> load_images(const std::string& list_path) {
  std::ifstream list(list_path);
  if (!list) throw std::runtime_error("Cannot open image list");
  std::vector<cv::Mat> images;
  for (std::string path; std::getline(list, path);) {
    if (path.empty() || path.front() == '#') continue;
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty() || image.type() != CV_8UC3)
      throw std::runtime_error("Cannot decode listed BGR image: " + path);
    images.push_back(std::move(image));
  }
  if (images.empty()) throw std::runtime_error("Empty image list");
  return images;
}
uint8_t* bytes(vart::TensorBuffer* buffer) {
  auto [address, capacity] = buffer->data({});
  if (!address || capacity < buffer->get_tensor()->get_data_size())
    throw std::runtime_error("VART buffer is not contiguous CPU memory");
  return reinterpret_cast<uint8_t*>(address);
}
void execute(vart::Runner& runner, BufferList& inputs, BufferList& outputs) {
  inputs[0]->sync_for_write(0, inputs[0]->get_tensor()->get_data_size());
  std::vector<vart::TensorBuffer*> in{inputs[0].get()};
  std::vector<vart::TensorBuffer*> out;
  for (auto& buffer : outputs) out.push_back(buffer.get());
  const auto job = runner.execute_async(in, out);
  if (job.second != 0 || runner.wait(static_cast<int>(job.first), -1) != 0)
    throw std::runtime_error("VART execute/wait failed");
  for (auto& buffer : outputs)
    buffer->sync_for_read(0, buffer->get_tensor()->get_data_size());
}
void decode(BufferList& outputs, const std::array<int, 3>& class_index,
            const std::array<int, 3>& box_index,
            const std::array<int, 3>& class_fix,
            const std::array<int, 3>& box_fix,
            const cv::Mat& image, const LetterboxResult& geometry,
            FastDecodeSixResult& result,
            const std::array<float, 6>* score_logit_offsets) {
  const int8_t* classes[3]{};
  const int8_t* boxes[3]{};
  for (int scale = 0; scale < 3; ++scale) {
    classes[scale] = reinterpret_cast<int8_t*>(bytes(outputs[class_index[scale]].get()));
    boxes[scale] = reinterpret_cast<int8_t*>(bytes(outputs[box_index[scale]].get()));
  }
  const int status = score_logit_offsets
      ? kv260_decode_six_offsets(classes, boxes, class_fix.data(), box_fix.data(),
                                 1, 0.25f, image.rows, image.cols,
                                 geometry.gain, geometry.left, geometry.top,
                                 score_logit_offsets->data(), &result)
      : kv260_decode_six(classes, boxes, class_fix.data(), box_fix.data(),
                         1, 0.25f, image.rows, image.cols,
                         geometry.gain, geometry.left, geometry.top, &result);
  if (status != 0 || result.count < 0 || result.count > 300)
    throw std::runtime_error("Final C++ decode failed");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 16)
      throw std::runtime_error("usage: bench_cpp_dual_runner_pipeline MODEL.xmodel IMAGE_LIST OUTPUT.bin WARMUP FRAMES OPENCV_THREADS [--score-logit-offsets SIX_FLOATS --score-offset-manifest-sha256 SHA256]");
    std::array<float, 6> offset_values{};
    std::string offset_manifest_sha;
    const std::array<float, 6>* score_logit_offsets = nullptr;
    if (argc == 16) {
      if (std::string(argv[7]) != "--score-logit-offsets" ||
          std::string(argv[14]) != "--score-offset-manifest-sha256")
        throw std::runtime_error("Calibrated decode requires six offsets and manifest SHA-256");
      for (int class_id = 0; class_id < 6; ++class_id) {
        const std::string token(argv[8 + class_id]);
        size_t consumed = 0;
        offset_values[class_id] = std::stof(token, &consumed);
        if (consumed != token.size())
          throw std::runtime_error("Class score offset contains trailing characters");
        if (!std::isfinite(offset_values[class_id]) || std::abs(offset_values[class_id]) > 2.0f)
          throw std::runtime_error("Invalid class score offset");
      }
      offset_manifest_sha = argv[15];
      if (offset_manifest_sha.size() != 64 ||
          !std::all_of(offset_manifest_sha.begin(), offset_manifest_sha.end(), [](char value) {
            return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
          }))
        throw std::runtime_error("Invalid score offset manifest SHA-256");
      score_logit_offsets = &offset_values;
    }
    const int warmup = std::stoi(argv[4]);
    const int count = std::stoi(argv[5]);
    const int cv_threads = std::stoi(argv[6]);
    if (warmup < 0 || count < 1 || count > 100000 || cv_threads < 1 || cv_threads > 8)
      throw std::runtime_error("Invalid warmup, frames or thread count");
    cv::setNumThreads(cv_threads);
    auto images = load_images(argv[2]);
    auto graph = xir::Graph::deserialize(argv[1]);
    auto* root = graph->get_root_subgraph();
    auto parts = root->children_topological_sort();
    xir::Subgraph* dpu = nullptr;
    for (auto* part : parts)
      if (part->has_attr("device") && part->get_attr<std::string>("device") == "DPU") {
        if (dpu) throw std::runtime_error("More than one DPU partition");
        dpu = part;
      }
    if (!dpu) throw std::runtime_error("No DPU partition");
    std::array<std::unique_ptr<vart::Runner>, 2> runners{
        vart::Runner::create_runner(dpu, "run"),
        vart::Runner::create_runner(dpu, "run")};
    auto in_tensors = runners[0]->get_input_tensors();
    auto out_tensors = runners[0]->get_output_tensors();
    if (in_tensors.size() != 1 || out_tensors.size() != 6 ||
        in_tensors[0]->get_shape() != std::vector<int>({1, 640, 640, 3}))
      throw std::runtime_error("Expected one 640x640 INT8 input and six outputs");
    const int input_fix = in_tensors[0]->get_attr<int>("fix_point");
    std::array<int, 3> class_index{-1, -1, -1}, box_index{-1, -1, -1};
    std::array<int, 3> class_fix{}, box_fix{};
    for (size_t index = 0; index < out_tensors.size(); ++index) {
      const auto shape = out_tensors[index]->get_shape();
      if (shape.size() != 4 || shape[0] != 1 || shape[1] != shape[2])
        throw std::runtime_error("Unexpected DPU output shape");
      const int scale = shape[1] == 80 ? 0 : (shape[1] == 40 ? 1 : (shape[1] == 20 ? 2 : -1));
      if (scale < 0) throw std::runtime_error("Unexpected DPU output scale");
      if (shape[3] == 6) {
        class_index[scale] = static_cast<int>(index);
        class_fix[scale] = out_tensors[index]->get_attr<int>("fix_point");
      } else if (shape[3] == 4) {
        box_index[scale] = static_cast<int>(index);
        box_fix[scale] = out_tensors[index]->get_attr<int>("fix_point");
      } else throw std::runtime_error("Unexpected DPU output channels");
    }
    if (std::any_of(class_index.begin(), class_index.end(), [](int value) { return value < 0; }) ||
        std::any_of(box_index.begin(), box_index.end(), [](int value) { return value < 0; }))
      throw std::runtime_error("Missing class/box output scale");
    std::array<BufferList, 2> inputs, outputs;
    for (int runner_id = 0; runner_id < 2; ++runner_id) {
      const auto actual_input = runners[runner_id]->get_input_tensors();
      const auto actual_output = runners[runner_id]->get_output_tensors();
      if (actual_input.size() != in_tensors.size() || actual_output.size() != out_tensors.size())
        throw std::runtime_error("Runner tensor count changed");
      for (size_t index = 0; index < in_tensors.size(); ++index)
        if (actual_input[index]->get_shape() != in_tensors[index]->get_shape() ||
            actual_input[index]->get_attr<int>("fix_point") != in_tensors[index]->get_attr<int>("fix_point"))
          throw std::runtime_error("Runner input contract changed");
      for (size_t index = 0; index < out_tensors.size(); ++index)
        if (actual_output[index]->get_shape() != out_tensors[index]->get_shape() ||
            actual_output[index]->get_attr<int>("fix_point") != out_tensors[index]->get_attr<int>("fix_point"))
          throw std::runtime_error("Runner output contract changed");
      inputs[runner_id] = vart::alloc_cpu_flat_tensor_buffers(actual_input);
      outputs[runner_id] = vart::alloc_cpu_flat_tensor_buffers(actual_output);
    }
    std::unique_ptr<void, decltype(&kv260_lb_destroy)> letterbox(kv260_lb_create(input_fix), kv260_lb_destroy);
    if (!letterbox) throw std::runtime_error("Cannot create letterbox context");
    for (int i = 0; i < warmup; ++i) {
      const int runner_id = i % 2;
      const auto& image = images[i % images.size()];
      LetterboxResult geometry{};
      const int code = kv260_lb_prepare(letterbox.get(), image.data, image.cols, image.rows,
                                        static_cast<int>(image.step[0]),
                                        reinterpret_cast<int8_t*>(bytes(inputs[runner_id][0].get())), &geometry);
      if (code != 0) throw std::runtime_error("Warmup preprocess failed");
      execute(*runners[runner_id], inputs[runner_id], outputs[runner_id]);
      FastDecodeSixResult detection{};
      decode(outputs[runner_id], class_index, box_index, class_fix, box_fix,
             image, geometry, detection, score_logit_offsets);
    }

    std::array<Queue<int>, 2> free_inputs, free_outputs, ready_dpu;
    Queue<int> inflight;
    std::mutex ready_mutex;
    std::condition_variable ready_condition;
    std::vector<bool> ready_post(count, false);
    for (int slot = 0; slot < 2; ++slot) {
      free_inputs[slot].push(0);
      free_outputs[slot].push(0);
    }
    for (int permit = 0; permit < 3; ++permit) inflight.push(permit);
    std::vector<Job> jobs(count);
    const auto start = Clock::now();
    std::thread pre([&] {
      for (int i = 0; i < count; ++i) {
        auto& job = jobs[i];
        job.submitted = Clock::now();
        inflight.pop();
        job.admitted = Clock::now();
        job.input_slot = i % 2;
        job.output_slot = job.input_slot;
        free_inputs[job.input_slot].pop();
        job.pre_start = Clock::now();
        const auto& image = images[i % images.size()];
        const int code = kv260_lb_prepare(letterbox.get(), image.data, image.cols, image.rows,
                                          static_cast<int>(image.step[0]),
                                          reinterpret_cast<int8_t*>(bytes(inputs[job.input_slot][0].get())),
                                          &job.geometry);
        if (code != 0) std::terminate();
        job.pre_end = Clock::now();
        ready_dpu[job.input_slot].push(i);
      }
    });
    std::array<std::thread, 2> dpu_threads;
    for (int runner_id = 0; runner_id < 2; ++runner_id) {
      dpu_threads[runner_id] = std::thread([&, runner_id] {
        for (int completed = runner_id; completed < count; completed += 2) {
          const int i = ready_dpu[runner_id].pop();
          auto& job = jobs[i];
          free_outputs[runner_id].pop();
          job.dpu_start = Clock::now();
          try { execute(*runners[runner_id], inputs[runner_id], outputs[runner_id]); }
          catch (...) { std::terminate(); }
          job.dpu_end = Clock::now();
          free_inputs[runner_id].push(0);
          {
            std::lock_guard<std::mutex> guard(ready_mutex);
            ready_post[i] = true;
          }
          ready_condition.notify_one();
        }
      });
    }
    std::thread post([&] {
      for (int i = 0; i < count; ++i) {
        {
          std::unique_lock<std::mutex> lock(ready_mutex);
          ready_condition.wait(lock, [&] { return ready_post[i]; });
        }
        auto& job = jobs[i];
        job.post_start = Clock::now();
        try {
          decode(outputs[job.output_slot], class_index, box_index, class_fix, box_fix,
                 images[i % images.size()], job.geometry, job.detection,
                 score_logit_offsets);
        } catch (...) { std::terminate(); }
        job.delivered = Clock::now();
        free_outputs[job.output_slot].push(0);
        inflight.push(0);
      }
    });
    pre.join();
    for (auto& thread : dpu_threads) thread.join();
    post.join();
    const auto end = Clock::now();
    std::vector<double> latencies, pre_ms, dpu_ms, post_ms;
    latencies.reserve(count); pre_ms.reserve(count); dpu_ms.reserve(count); post_ms.reserve(count);
    for (const auto& job : jobs) {
      latencies.push_back(ms(job.submitted, job.delivered));
      pre_ms.push_back(ms(job.pre_start, job.pre_end));
      dpu_ms.push_back(ms(job.dpu_start, job.dpu_end));
      post_ms.push_back(ms(job.post_start, job.delivered));
    }
    std::ofstream result(argv[3], std::ios::binary);
    if (!result) throw std::runtime_error("Cannot open output binary");
    for (const auto& job : jobs) {
      const auto rows = static_cast<uint32_t>(job.detection.count);
      result.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
      result.write(reinterpret_cast<const char*>(job.detection.rows), rows * 6 * sizeof(float));
    }
    result.close();
    if (!result) throw std::runtime_error("Cannot write complete output binary");
    std::ofstream trace(std::string(argv[3]) + ".trace.csv");
    if (!trace) throw std::runtime_error("Cannot open output trace");
    trace << "frame_id,source_index,submitted_ns,admitted_ns,pre_start_ns,pre_end_ns,dpu_start_ns,dpu_end_ns,post_start_ns,delivered_ns,detections\n";
    for (int i = 0; i < count; ++i) {
      const auto& job = jobs[i];
      trace << i << ',' << (i % images.size()) << ',' << ns(job.submitted) << ','
            << ns(job.admitted) << ',' << ns(job.pre_start) << ',' << ns(job.pre_end) << ','
            << ns(job.dpu_start) << ',' << ns(job.dpu_end) << ','
            << ns(job.post_start) << ',' << ns(job.delivered) << ','
            << job.detection.count << '\n';
    }
    trace.close();
    if (!trace) throw std::runtime_error("Cannot write complete trace");
    const double elapsed = ms(start, end) / 1000.0;
    std::ostringstream summary;
    std::ostringstream score_identity;
    if (score_logit_offsets) {
      score_identity << std::setprecision(9) << '[';
      for (size_t class_id = 0; class_id < offset_values.size(); ++class_id) {
        if (class_id) score_identity << ',';
        score_identity << offset_values[class_id];
      }
      score_identity << ']';
    }
    summary << std::fixed << std::setprecision(6)
              << "{\"scope\":\"diagnostic_complete_bgr_to_final_boxes\",\"batch\":1,\"input_hw\":[640,640],"
              << "\"images_unique\":" << images.size() << ",\"frames\":" << count
              << ",\"warmup\":" << warmup << ",\"inflight_limit\":3,\"input_slots\":2,\"output_slots\":2,\"runners\":2,\"opencv_threads\":" << cv::getNumThreads()
              << ",\"seconds\":" << elapsed << ",\"fps\":" << count / elapsed
              << ",\"latency_p50_ms\":" << percentile(latencies, 0.5)
              << ",\"latency_p95_ms\":" << percentile(latencies, 0.95)
              << ",\"pre_mean_ms\":" << mean(pre_ms)
              << ",\"dpu_mean_ms\":" << mean(dpu_ms)
              << ",\"post_mean_ms\":" << mean(post_ms)
              << ",\"score_offset_mode\":\"" << (score_logit_offsets ? "post_topk_manifest" : "raw") << "\""
              << ",\"score_logit_offsets\":" << (score_logit_offsets ? score_identity.str() : "null")
              << ",\"score_offset_manifest_sha256\":"
              << (score_logit_offsets ? "\"" + offset_manifest_sha + "\"" : "null")
              << ",\"output_binary\":\"" << argv[3] << "\",\"trace\":\""
              << argv[3] << ".trace.csv\"}\n";
    std::ofstream summary_file(std::string(argv[3]) + ".json");
    summary_file << summary.str();
    summary_file.close();
    if (!summary_file) throw std::runtime_error("Cannot write summary JSON");
    std::cout << summary.str();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bench_cpp_full_pipeline: " << error.what() << '\n';
    return 2;
  }
}
