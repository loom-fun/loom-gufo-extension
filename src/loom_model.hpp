// Loom package probing and the shared model handle (libloom_text_embed).
#ifndef GUFO_LOOM_MODEL_HPP_
#define GUFO_LOOM_MODEL_HPP_

#include <cstdlib>
#include <cstdio>
#include <loom_text_embed.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gufo/ext/generated/loom_constants.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/sha256.hpp>
#include <gufo/ext/text.hpp>

namespace gufo::loom {

namespace fs = std::filesystem;
using ext::TextRunnerToken;

inline constexpr std::uint64_t kExtentSlack = 1024;
// Retained row checkpoints, and host blocks naming them in Gufo's cache.
inline constexpr std::size_t kCheckpoints = 64;
inline constexpr std::size_t kMarkerBytes = 16;
// Marker blocks plus the cache's per-slot metadata.
inline constexpr std::size_t kHostCacheBytes = std::size_t{8} << 20;
inline constexpr std::size_t kTextBuffer = 1 << 16;

// ---------------------------------------------------------------- probe ---

inline std::optional<fs::path> ManifestPath(std::string_view model_root) {
  std::error_code error;
  fs::path path(model_root);
  if (fs::is_directory(path, error))
    path /= "model.json";
  if (path.filename() != "model.json" || !fs::is_regular_file(path, error))
    return std::nullopt;
  return path;
}

// A Loom model card with programs and a service of kind "text".
inline bool ProbeLoomText(std::string_view model_root) {
  const auto manifest = ManifestPath(model_root);
  if (!manifest)
    return false;
  std::error_code error;
  if (fs::file_size(*manifest, error) > (std::size_t{4} << 20) || error)
    return false;
  std::ifstream in(*manifest, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  try {
    const auto card = json::parse(text);
    const auto* services = card.find("services");
    if (!card.contains("schema_version") || !card.contains("programs") ||
        services == nullptr || !services->is_object())
      return false;
    for (const auto& [name, service] : services->members()) {
      (void)name;
      if (service.is_object() && service.member_str("kind") == "text")
        return true;
    }
  } catch (const std::exception&) {
  }
  return false;
}

// ---------------------------------------------------------------- model ---

// One opened package; every Loom call is serialized on its mutex (recursive:
// releasing a cache entry can release a checkpoint inside a runner call).
class LoomModel {
public:
  LoomModel(const std::string& path, const ext::ParsedOptions& options,
            std::size_t sessions) {
    std::string preset, device;
    std::uint64_t memory = 0;
    if (const auto* value =
            ext::FindOption<std::string>(options, "--loom-preset"))
      preset = *value;
    if (const auto* value =
            ext::FindOption<std::string>(options, "--loom-device"))
      device = *value;
    if (const auto* value =
            ext::FindOption<std::int64_t>(options, "--loom-memory-bytes"))
      memory = static_cast<std::uint64_t>(*value);
    loom_text_embed_options_t embed_options{
        .model_path = path.c_str(),
        .preset = preset.c_str(),
        .device_uri = device.empty() ? nullptr : device.c_str(),
        .memory_bytes = memory,
        .rows = static_cast<std::uint32_t>(std::max<std::size_t>(sessions, 1)),
        .checkpoints = kCheckpoints,
    };
    char error[2048] = {};
    if (loom_text_embed_open(&embed_options, &embed_, error, sizeof(error)))
      throw std::runtime_error(std::string("loom: ") + error);
    loom_text_embed_info(embed_, &info_);
    fingerprint_ = PackageFingerprint(path, preset);
  }
  ~LoomModel() {
    for (auto& [id, checkpoint] : checkpoints_)
      loom_text_embed_checkpoint_release(embed_, checkpoint);
    loom_text_embed_close(embed_);
  }
  LoomModel(const LoomModel&) = delete;
  LoomModel& operator=(const LoomModel&) = delete;

  [[nodiscard]] const loom_text_embed_info_t& info() const { return info_; }
  // The package's manifest and programs (and the preset): persisted state is
  // only valid for the same ones.
  [[nodiscard]] const std::string& Fingerprint() const { return fingerprint_; }
  [[nodiscard]] loom_text_embed_t* embed() const { return embed_; }
  std::recursive_mutex& mutex() const { return mutex_; }

  // Throws on a nonzero status other than `allowed`.
  int Check(int status, const char* what, int allowed = 0) const {
    if (status != 0 && status != allowed)
      throw std::runtime_error(std::string("loom: ") + what + ": " +
                               loom_text_embed_error(embed_));
    return status;
  }

  // Checkpoint handles named by cache payloads.
  std::uint64_t Keep(loom_text_embed_checkpoint_t* checkpoint) {
    const auto id = next_checkpoint_++;
    checkpoints_.emplace(id, checkpoint);
    return id;
  }
  loom_text_embed_checkpoint_t* Find(std::uint64_t id) const {
    const auto found = checkpoints_.find(id);
    return found == checkpoints_.end() ? nullptr : found->second;
  }
  void Release(std::uint64_t id) {
    std::lock_guard lock(mutex_);
    const auto found = checkpoints_.find(id);
    if (found == checkpoints_.end())
      return;
    loom_text_embed_checkpoint_release(embed_, found->second);
    checkpoints_.erase(found);
  }

private:
  loom_text_embed_t* embed_{nullptr};
  loom_text_embed_info_t info_{};
  mutable std::recursive_mutex mutex_;
  std::unordered_map<std::uint64_t, loom_text_embed_checkpoint_t*>
      checkpoints_;
  std::uint64_t next_checkpoint_{1};
  std::string fingerprint_;

  static std::string PackageFingerprint(const std::string& path,
                                        const std::string& preset) {
    namespace fs = std::filesystem;
    const fs::path root =
        fs::is_directory(path) ? fs::path(path) : fs::path(path).parent_path();
    std::vector<fs::path> files{root / "model.json"};
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(root / "program", error))
      if (entry.is_regular_file())
        files.push_back(entry.path());
    std::sort(files.begin() + 1, files.end());
    crypto::Sha256Hasher hasher;
    hasher.Update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(preset.data()), preset.size()));
    for (const auto& file : files) {
      std::ifstream in(file, std::ios::binary);
      const std::string bytes((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
      hasher.Update(std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string hex;
    for (const auto byte : hasher.Finish()) {
      hex += kHex[byte >> 4];
      hex += kHex[byte & 15];
    }
    return hex;
  }
};

// One request's output reader (the package's chat program and parser).
class LoomPromptContext final : public ext::TextPromptContext {
public:
  LoomPromptContext(std::shared_ptr<LoomModel> model,
                    loom_text_embed_request_t* request)
      : model(std::move(model)), request(request) {}
  ~LoomPromptContext() override {
    std::lock_guard lock(model->mutex());
    loom_text_embed_request_free(request);
  }
  std::shared_ptr<LoomModel> model;
  loom_text_embed_request_t* request;
  // Input ranges replaced by projected image features; images are projected
  // on the first prefill that needs them.
  std::vector<std::pair<std::size_t, std::size_t>> images;
  mutable bool projected{false};
  // Reasoning tokens allowed before the runner closes reasoning.
  std::optional<std::size_t> reasoning_budget;
  // The prompt ends inside an open reasoning block.
  bool opens_reasoning{false};
  [[nodiscard]] bool has_images() const { return !images.empty(); }
};


}  // namespace gufo::loom

#endif  // GUFO_LOOM_MODEL_HPP_
