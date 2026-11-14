// Loom model packages as Gufo text models.
//
// A fetched Loom package (model.json, its compiled programs and parameters)
// runs through Gufo's ordinary step contract, TextModelRunner. Gufo's server,
// scheduler, component cache, sampler policy and parsers stay in charge; Loom
// supplies the model's kernels, chat program, tokenizer, output parser and
// drafter through libloom_text_embed:
//
// - Each Gufo session slot is one resident Loom row. Gufo decides which
//   sessions step together; one decode batch is one packed Loom epoch, and a
//   package with a drafter decodes several verified tokens per step.
// - Gufo's component cache keeps Loom row checkpoints: a capture pins the
//   row's exact state on device (pages shared copy-on-write), a restore puts
//   it into any row, and evicting the cache entry releases it.
// - Output is read with the package's own chat program, as loom-serve reads
//   it, and streamed to Gufo in one canonical markup: reasoning inside
//   <think>...</think>, answers as text, tool calls as <tool_call> JSON
//   blocks. Gufo's parsers then serve every endpoint for every architecture.
//
// Sampling runs on device with the package's sampler (temperature, top-k,
// top-p, seed); Gufo's grammar constraints, logprobs, min-p and penalties
// need logits, which Loom packages do not expose.

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
#include <gufo/ext/text.hpp>

namespace gufo::loom {
namespace {

namespace fs = std::filesystem;
using ext::TextRunnerToken;

constexpr std::uint64_t kExtentSlack = 1024;
// Retained row checkpoints, and host blocks naming them in Gufo's cache.
constexpr std::size_t kCheckpoints = 64;
constexpr std::size_t kMarkerBytes = 16;
// Marker blocks plus the cache's per-slot metadata.
constexpr std::size_t kHostCacheBytes = std::size_t{8} << 20;
constexpr std::size_t kTextBuffer = 1 << 16;

// ---------------------------------------------------------------- probe ---

std::optional<fs::path> ManifestPath(std::string_view model_root) {
  std::error_code error;
  fs::path path(model_root);
  if (fs::is_directory(path, error))
    path /= "model.json";
  if (path.filename() != "model.json" || !fs::is_regular_file(path, error))
    return std::nullopt;
  return path;
}

// A Loom model card with programs and a service of kind "text".
bool ProbeLoomText(std::string_view model_root) {
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
  }
  ~LoomModel() {
    for (auto& [id, checkpoint] : checkpoints_)
      loom_text_embed_checkpoint_release(embed_, checkpoint);
    loom_text_embed_close(embed_);
  }
  LoomModel(const LoomModel&) = delete;
  LoomModel& operator=(const LoomModel&) = delete;

  [[nodiscard]] const loom_text_embed_info_t& info() const { return info_; }
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
};

// ---------------------------------------------------------------- cache ---

// Free Loom rows, shared by the adapter and its slots.
struct RowPool {
  std::mutex mutex;
  std::vector<bool> used;
};

class LoomSlot final : public cache::Slot {
public:
  LoomSlot(std::uint32_t row, std::shared_ptr<RowPool> pool)
      : row_(row), pool_(std::move(pool)) {}
  ~LoomSlot() override {
    std::lock_guard lock(pool_->mutex);
    pool_->used[row_] = false;
  }
  [[nodiscard]] bool IsValid() const noexcept override { return true; }
  [[nodiscard]] std::uint32_t row() const { return row_; }
  // Tokens Gufo counts as consumed, and the prompt token still deferred
  // (consumed by Gufo's count, not yet by the Loom row).
  std::size_t position{0};
  std::int64_t deferred{-1};
  // A restore in progress loaded its checkpoint.
  bool restore_ok{false};
  // The Loom row must be reset before reuse.
  bool stale{true};

private:
  std::uint32_t row_;
  std::shared_ptr<RowPool> pool_;
};

// The one component: the row endpoint, held on device by a Loom checkpoint
// and named in Gufo's cache by {checkpoint id, deferred token}.
constexpr cache::ComponentId kEndpoint{1};
const cache::ComponentDescriptor kEndpointDescriptor{
    .id = kEndpoint,
    .layout_version = 1,
    .kind = cache::ComponentKind::kPrivateState,
    .state_bytes = kMarkerBytes,
};

class DoneSignal final : public cache::CompletionSignal {
public:
  explicit DoneSignal(cache::TransferResult result) : result_(result) {}
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override {
    return result_;
  }

private:
  cache::TransferResult result_;
};

cache::Completion Done(bool ok) {
  return cache::Completion(std::make_unique<DoneSignal>(
      ok ? cache::TransferResult::kSucceeded : cache::TransferResult::kFailed));
}

struct Endpoint {
  std::uint64_t checkpoint;
  std::int64_t deferred;
};

// Owns the bytes of one cache payload; the checkpoint they name lives as
// long as Gufo keeps the payload.
struct EndpointPayload {
  std::shared_ptr<LoomModel> model;
  std::array<std::byte, kMarkerBytes> bytes{};
  ~EndpointPayload() {
    Endpoint endpoint{};
    std::memcpy(&endpoint, bytes.data(), sizeof(endpoint));
    if (endpoint.checkpoint != 0)
      model->Release(endpoint.checkpoint);
  }
};

class LoomAdapter final : public cache::Adapter {
public:
  LoomAdapter(std::shared_ptr<LoomModel> model, std::uint32_t rows)
      : model_(std::move(model)), pool_(std::make_shared<RowPool>()) {
    pool_->used.assign(rows, false);
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override {
    // Checkpoints live on device, so entries are never written to disk.
    return {.continuation = true, .persistent_encoding = false};
  }
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override {
    return std::span(&kEndpointDescriptor, 1);
  }
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override {
    const std::string id =
        std::string("loom-text-2:") + model_->info().model_id;
    return cache::Identity(id.begin(), id.end());
  }
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override {
    std::lock_guard lock(pool_->mutex);
    for (std::uint32_t row = 0; row < pool_->used.size(); ++row) {
      if (!pool_->used[row]) {
        pool_->used[row] = true;
        return std::make_unique<LoomSlot>(row, pool_);
      }
    }
    throw cache::ResourceExhausted();
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot& slot) const override {
    return {{.id = kEndpoint,
             .valid_rows = dynamic_cast<const LoomSlot&>(slot).position}};
  }
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override {
    std::vector<cache::Rows> boundaries;
    if (first >= end)
      return boundaries;
    const cache::Rows step =
        std::max<cache::Rows>(model_->info().prefill_capacity, 1);
    for (cache::Rows at = first + step; at < end; at += step)
      boundaries.push_back(at);
    boundaries.push_back(end);
    return boundaries;
  }
  void BeginRestore(
      cache::Slot& slot,
      std::span<const cache::ComponentPosition> positions) override {
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    loom.restore_ok = false;
    loom.position = positions.empty() ? 0 : positions.front().valid_rows;
  }
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot& slot,
                                                 cache::ComponentId,
                                                 std::span<std::byte> bytes,
                                                 cache::Stream&) override {
    const auto& loom = dynamic_cast<const LoomSlot&>(slot);
    std::lock_guard lock(model_->mutex());
    // Loom pins only a nonempty row.
    if (loom_text_embed_row_position(model_->embed(), loom.row()) == 0)
      return Done(false);
    loom_text_embed_checkpoint_t* checkpoint = nullptr;
    if (model_->Check(loom_text_embed_row_pin(model_->embed(), loom.row(),
                                              &checkpoint),
                      "pin", 1) != 0)
      return Done(false);
    const Endpoint endpoint{model_->Keep(checkpoint), loom.deferred};
    std::memcpy(bytes.data(), &endpoint,
                std::min(bytes.size(), sizeof(endpoint)));
    return Done(true);
  }
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&,
                                              cache::ComponentId, cache::Rows,
                                              cache::Rows, std::span<std::byte>,
                                              cache::Stream&) override {
    throw std::logic_error("loom endpoints have no rows");
  }
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows, cache::Rows,
                                             std::span<const std::byte>,
                                             cache::Stream&) override {
    throw std::logic_error("loom endpoints have no rows");
  }
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot& slot,
                                              cache::ComponentId,
                                              std::span<const std::byte> bytes,
                                              cache::Stream&) override {
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    Endpoint endpoint{};
    std::memcpy(&endpoint, bytes.data(),
                std::min(bytes.size(), sizeof(endpoint)));
    std::lock_guard lock(model_->mutex());
    const auto* checkpoint = model_->Find(endpoint.checkpoint);
    int restored = 0;
    if (checkpoint != nullptr)
      model_->Check(loom_text_embed_row_restore(model_->embed(), loom.row(),
                                                checkpoint, &restored),
                    "restore");
    loom.restore_ok = restored != 0;
    loom.deferred = endpoint.deferred;
    loom.stale = !loom.restore_ok;
    return Done(loom.restore_ok);
  }
  [[nodiscard]] bool Validate(
      cache::Slot& slot, std::span<const cache::ComponentPosition>) override {
    return dynamic_cast<LoomSlot&>(slot).restore_ok;
  }
  [[nodiscard]] bool Invalidate(cache::Slot& slot) noexcept override {
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    loom.position = 0;
    loom.deferred = -1;
    loom.restore_ok = false;
    loom.stale = true;
    return true;
  }

private:
  std::shared_ptr<LoomModel> model_;
  std::shared_ptr<RowPool> pool_;
};

class NullStream final : public cache::Stream {
public:
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};

// ---------------------------------------------------------------- state ---

class LoomState final : public ext::TextRunnerState {
public:
  LoomState(LoomSlot& slot, loom_text_embed_decoder_t* decoder)
      : slot(&slot), decoder(decoder) {}
  ~LoomState() override { loom_text_embed_decoder_free(decoder); }
  void Invalidate() noexcept override {
    slot->position = 0;
    slot->deferred = -1;
    slot->stale = true;
  }
  [[nodiscard]] std::uint32_t row() const { return slot->row(); }
  [[nodiscard]] std::size_t position() const { return slot->position; }

  LoomSlot* slot;
  // Per request: output reader, streaming detokenizer, the bound sampling,
  // the selection whose piece was produced, and whether anything was.
  std::shared_ptr<const LoomPromptContext> context;
  loom_text_embed_decoder_t* decoder;
  std::string bound;
  std::optional<std::pair<std::size_t, std::int32_t>> piece_key;
  std::string piece;
  std::size_t generated{0};
  bool produced{false};
  bool unconstrained{false};
  bool flushed{false};
};

LoomState& Require(ext::TextRunnerState& state) {
  auto* loom = dynamic_cast<LoomState*>(&state);
  if (loom == nullptr)
    throw std::invalid_argument("state does not belong to the Loom runner");
  return *loom;
}

// ------------------------------------------------------- chat requests ---

const char* RoleName(ext::ChatRole role) {
  switch (role) {
    case ext::ChatRole::kSystem:
    case ext::ChatRole::kDeveloper:
      return "system";
    case ext::ChatRole::kUser:
      return "user";
    case ext::ChatRole::kAssistant:
      return "assistant";
    case ext::ChatRole::kTool:
      return "tool";
  }
  return "user";
}

json::Value ParseOr(const std::string& text, json::Value fallback) {
  try {
    return json::parse(text);
  } catch (const std::exception&) {
    return fallback;
  }
}

// The OpenAI chat body the package's chat program renders.
std::string ChatBody(const std::string& model,
                     const ext::ChatRequest& request) {
  json::Value body = json::Value::object();
  body["model"] = model;
  json::Value messages = json::Value::array();
  for (const auto& message : request.messages) {
    if (!message.images.empty())
      throw std::invalid_argument("this Loom package does not accept images");
    json::Value out = json::Value::object();
    out["role"] = RoleName(message.role);
    out["content"] = message.content;
    if (message.role == ext::ChatRole::kAssistant) {
      if (!message.thought.empty())
        out["reasoning_content"] = message.thought;
      if (!message.tool_calls.empty()) {
        json::Value calls = json::Value::array();
        for (const auto& call : message.tool_calls) {
          json::Value arguments = json::Value::object();
          for (const auto& argument : call.arguments)
            arguments[argument.name] =
                argument.is_string
                    ? json::Value(argument.value)
                    : ParseOr(argument.value, json::Value(argument.value));
          json::Value function = json::Value::object();
          function["name"] = call.name;
          function["arguments"] = arguments.dump();
          json::Value entry = json::Value::object();
          entry["id"] = call.id;
          entry["type"] = "function";
          entry["function"] = std::move(function);
          calls.push_back(std::move(entry));
        }
        out["tool_calls"] = std::move(calls);
      }
    }
    if (message.role == ext::ChatRole::kTool && !message.tool_call_id.empty())
      out["tool_call_id"] = message.tool_call_id;
    messages.push_back(std::move(out));
  }
  body["messages"] = std::move(messages);
  if (!request.tools.empty()) {
    json::Value tools = json::Value::array();
    for (const auto& tool : request.tools) {
      auto definition = ParseOr(tool.definition_json, json::Value());
      if (!definition.is_object()) {
        json::Value function = json::Value::object();
        function["name"] = tool.name;
        if (!tool.description.empty())
          function["description"] = tool.description;
        function["parameters"] =
            ParseOr(tool.parameters_json, json::Value::object());
        definition = json::Value::object();
        definition["type"] = "function";
        definition["function"] = std::move(function);
      }
      tools.push_back(std::move(definition));
    }
    body["tools"] = std::move(tools);
    if (request.tool_choice == ext::ChatRequest::ToolChoice::kNone) {
      body["tool_choice"] = "none";
    } else if (request.tool_choice == ext::ChatRequest::ToolChoice::kRequired) {
      if (request.forced_tool_name.empty()) {
        body["tool_choice"] = "required";
      } else {
        json::Value function = json::Value::object();
        function["name"] = request.forced_tool_name;
        json::Value choice = json::Value::object();
        choice["type"] = "function";
        choice["function"] = std::move(function);
        body["tool_choice"] = std::move(choice);
      }
    }
  }
  if (request.reasoning.enabled.has_value()) {
    json::Value kwargs = json::Value::object();
    kwargs["enable_thinking"] = *request.reasoning.enabled;
    body["chat_template_kwargs"] = std::move(kwargs);
  }
  return body.dump();
}

// A request body that only carries sampling, for row binding.
std::string SamplingBody(const std::string& model,
                         const sampling::SamplingConfig& config) {
  json::Value body = json::Value::object();
  body["model"] = model;
  json::Value message = json::Value::object();
  message["role"] = "user";
  message["content"] = ".";
  json::Value messages = json::Value::array();
  messages.push_back(std::move(message));
  body["messages"] = std::move(messages);
  body["temperature"] = static_cast<double>(config.temperature);
  body["top_p"] = static_cast<double>(config.top_p);
  if (config.top_k > 0)
    body["top_k"] = config.top_k;
  if (config.seed >= 0)
    body["seed"] = config.seed;
  return body.dump();
}

// Tool calls as Gufo's JSON dialect reads them.
std::string ToolMarkup(std::string_view calls_json) {
  if (calls_json.empty())
    return {};
  std::string markup;
  try {
    const auto calls = json::parse(calls_json);
    for (const auto& call : calls.items()) {
      const auto* function = call.find("function");
      if (function == nullptr)
        continue;
      json::Value block = json::Value::object();
      block["name"] = function->member_str("name");
      const auto arguments = function->member_str("arguments");
      block["arguments"] = ParseOr(arguments, json::Value(arguments));
      markup += "<tool_call>\n" + block.dump() + "\n</tool_call>";
    }
  } catch (const std::exception&) {
  }
  return markup;
}

// --------------------------------------------------------------- runner ---

class LoomRunner final : public ext::TextModelRunner {
public:
  explicit LoomRunner(std::shared_ptr<LoomModel> model)
      : model_(std::move(model)),
        rows_(std::max<std::uint32_t>(model_->info().rows, 1)),
        width_(std::max<std::uint32_t>(
            1, std::min<std::uint32_t>(model_->info().max_epoch_spans,
                                       rows_))) {}

  [[nodiscard]] ext::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->info().model_id,
        .state_abi = "loom-text-2",
        .max_context = model_->info().context_capacity,
        .capabilities =
            {
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                // Every decode step goes through DecodeBatch, which sees the
                // next selection before publishing (end of output).
                .multi_token_decode = true,
                .batched_multi_token_decode = width_ > 1,
                .batched_multi_token_decode_max_width = width_,
                .prefix_reuse = true,
            },
        .scheduling =
            {
                .pass_tokens = std::max<std::size_t>(
                    model_->info().prefill_capacity, 1),
                .max_layer_groups = 1,
                .max_yield_decode_width = 1,
            },
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    // Loom accounts device memory itself; the host cache holds only
    // checkpoint names and its own metadata.
    return {.per_request_state_bytes = 0,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = kHostCacheBytes,
            .retained_snapshot_ceiling_bytes = kHostCacheBytes};
  }

  [[nodiscard]] std::vector<ext::TextExecutionPlan> SupportedPlans()
      const override {
    std::vector<ext::TextExecutionPlan> plans{
        {.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= width_; ++width)
      plans.push_back({.kind = ext::TextExecutionPlanKind::kBatched,
                       .physical_width = width});
    return plans;
  }

  [[nodiscard]] std::uint32_t VocabularySize() const override {
    std::lock_guard lock(model_->mutex());
    return static_cast<std::uint32_t>(
        loom_text_embed_vocabulary_size(model_->embed()));
  }

  // Gufo binds a grammar to every tool request; the package samples on
  // device, so the grammar is released (see Unconstrain).
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        VocabularySize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              Decode(std::span<const TextRunnerToken>(&id, 1)), false};
        });
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::lock_guard lock(model_->mutex());
    std::vector<std::int32_t> tokens(text.size() + 16);
    std::size_t count = 0;
    model_->Check(loom_text_embed_tokenize(model_->embed(), text.data(),
                                           text.size(), 1, tokens.data(),
                                           tokens.size(), &count),
                  "tokenize");
    return std::vector<TextRunnerToken>(tokens.begin(), tokens.begin() + count);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    auto prepared = PreparePrompt(request);
    if (!prepared)
      return std::nullopt;
    return std::move(prepared->tokens);
  }

  // The package's chat program renders the prompt; the request's output
  // reader travels with it to the state (SetPromptContext).
  [[nodiscard]] std::optional<ext::TextPreparedPrompt> PreparePrompt(
      const ext::ChatRequest& request) const override {
    const std::string body = ChatBody(model_->info().model_id, request);
    std::lock_guard lock(model_->mutex());
    std::vector<std::int32_t> tokens(model_->info().context_capacity);
    std::size_t count = 0;
    loom_text_embed_request_t* handle = nullptr;
    if (loom_text_embed_chat_prepare(model_->embed(), body.data(), body.size(),
                                     tokens.data(), tokens.size(), &count,
                                     &handle) != 0)
      throw std::invalid_argument(
          std::string("loom: ") + loom_text_embed_error(model_->embed()));
    auto context = std::make_shared<LoomPromptContext>(model_, handle);
    // The next turn renders this conversation plus the answer and a new user
    // message. Where that rendering departs from this prompt, the assistant
    // framing begins: Gufo checkpoints before it so the next turn reuses all
    // earlier work, whatever the package's template.
    std::size_t stable = 0;
    if (!request.messages.empty()) {
      ext::ChatRequest next = request;
      next.messages.emplace_back(ext::ChatRole::kAssistant, "\x01");
      next.messages.emplace_back(ext::ChatRole::kUser, "\x01");
      const std::string next_body = ChatBody(model_->info().model_id, next);
      std::vector<std::int32_t> next_tokens(model_->info().context_capacity);
      std::size_t next_count = 0;
      if (loom_text_embed_chat_prepare(model_->embed(), next_body.data(),
                                       next_body.size(), next_tokens.data(),
                                       next_tokens.size(), &next_count,
                                       nullptr) == 0) {
        const std::size_t limit = std::min(count, next_count);
        while (stable < limit && tokens[stable] == next_tokens[stable])
          ++stable;
      }
    }
    return ext::TextPreparedPrompt{
        .tokens = std::vector<TextRunnerToken>(tokens.begin(),
                                               tokens.begin() + count),
        .context = std::move(context),
        .cache_prefix_tokens = stable < count ? stable : 0,
    };
  }

  void SetPromptContext(
      ext::TextRunnerState& state,
      std::shared_ptr<const ext::TextPromptContext> context) const override {
    auto& loom = Require(state);
    auto loom_context =
        std::dynamic_pointer_cast<const LoomPromptContext>(context);
    if (context != nullptr && loom_context == nullptr)
      throw std::invalid_argument("prompt context is not from the Loom runner");
    // Called again after a restore within the same request.
    if (loom_context == loom.context && !loom.produced)
      return;
    std::lock_guard lock(model_->mutex());
    loom.context = std::move(loom_context);
    loom.bound.clear();
    loom.piece_key.reset();
    loom.generated = 0;
    loom.produced = false;
    loom.unconstrained = false;
    loom.flushed = false;
    model_->Check(loom_text_embed_decoder_reset(loom.decoder), "decoder");
  }

  // The package's chat program may open reasoning in the prompt; Gufo's
  // reasoning budget then counts from the prompt's opener.
  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ext::ChatRequest& request) const override {
    const auto prepared = PreparePrompt(request);
    if (!prepared || prepared->tokens.empty())
      return ext::InitialOutputState::kAuto;
    const auto& tokens = prepared->tokens;
    const std::size_t tail = std::min<std::size_t>(tokens.size(), 4);
    std::string text = Decode(std::span(tokens).last(tail));
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    const auto tags = ReasoningBudgetTags();
    const auto ends_with = [&](std::string_view suffix) {
      return !suffix.empty() && text.size() >= suffix.size() &&
             text.compare(text.size() - suffix.size(), suffix.size(),
                          suffix) == 0;
    };
    if (ends_with(tags.start))
      return ext::InitialOutputState::kThought;
    return ext::InitialOutputState::kAuto;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::lock_guard lock(model_->mutex());
    return DecodeLocked(std::vector<std::int32_t>(tokens.begin(), tokens.end()));
  }

  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t) const override {
    // One ledger block per retainable checkpoint: when Gufo's cache holds
    // kCheckpoints entries, a new capture makes it evict one first.
    struct Block {
      cache::ResourceCharge pool;
    };
    auto blocks = std::make_shared<std::vector<Block>>();
    for (std::size_t i = 0; i < kCheckpoints; ++i)
      blocks->push_back(
          {ledger.Reserve(cache::ResourceCategory::kBackingFree, 64).Convert()});
    auto resources = std::make_unique<ext::ComponentCacheResources>();
    resources->owner = blocks;
    resources->adapter = std::make_unique<LoomAdapter>(model_, rows_);
    resources->stream = [] { return std::make_unique<NullStream>(); };
    resources->allocate = [blocks, model = model_](
                              const cache::PayloadRequest& request)
        -> cache::ComponentCaptureBuffer {
      for (auto& block : *blocks) {
        if (request.bytes > kMarkerBytes ||
            block.pool.Info().category != cache::ResourceCategory::kBackingFree)
          continue;
        try {
          auto reservation = std::make_shared<cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          auto payload = std::make_shared<EndpointPayload>();
          payload->model = model;
          const std::span<std::byte> bytes(payload->bytes);
          return {bytes,
                  [reservation, payload] {
                    // Gufo reads a payload through its owner: alias the
                    // bytes, keeping the whole payload (and checkpoint) alive.
                    return cache::Payload::Committed(
                        reservation->Convert(),
                        std::shared_ptr<const void>(payload,
                                                    payload->bytes.data()));
                  },
                  payload.get()};
        } catch (const std::bad_alloc&) {
        }
      }
      throw cache::ResourceExhausted();
    };
    resources->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    return resources;
  }

  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter&, cache::Slot& slot) const override {
    std::lock_guard lock(model_->mutex());
    loom_text_embed_decoder_t* decoder = nullptr;
    model_->Check(loom_text_embed_decoder_open(model_->embed(), &decoder),
                  "decoder");
    return std::make_unique<LoomState>(dynamic_cast<LoomSlot&>(slot), decoder);
  }

  // After a restore the row holds the checkpoint's state at `position`.
  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    auto& loom = Require(state);
    std::lock_guard lock(model_->mutex());
    loom.slot->position = position;
    if (position == 0) {
      ResetRowLocked(loom);
      return;
    }
    loom.slot->stale = false;
  }

  [[nodiscard]] ext::TextPrefillStep Prefill(
      ext::TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& loom = Require(state);
    if (offset != loom.position() || offset >= prompt.size())
      throw std::invalid_argument("loom prefill must continue the row");
    const std::size_t count = std::min(
        prompt.size() - offset, std::max<std::size_t>(max_input_tokens, 1));
    const bool last = offset + count == prompt.size();
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    const std::uint64_t extent = std::min<std::uint64_t>(
        model_->info().context_capacity, prompt.size() + kExtentSlack);
    if (offset == 0 || loom.slot->stale) {
      if (offset != 0)
        throw std::logic_error("loom row lost its state");
      ResetRowLocked(loom);
      if (model_->Check(loom_text_embed_row_begin(embed, loom.row(), nullptr,
                                                  extent, 1),
                        "reserve", 1) != 0)
        throw cache::ResourceExhausted();
    } else if (loom_text_embed_row_extent(embed, loom.row()) < extent) {
      if (model_->Check(
              loom_text_embed_row_extend(embed, loom.row(), extent), "reserve",
              1) != 0)
        throw cache::ResourceExhausted();
      loom.bound.clear();
    }
    // A deferred token from the last prefill (or a restored checkpoint)
    // enters state first; this prompt's final token waits for its sampler.
    std::vector<std::int32_t> input;
    if (loom.slot->deferred >= 0)
      input.push_back(static_cast<std::int32_t>(loom.slot->deferred));
    input.insert(input.end(), prompt.begin() + offset,
                 prompt.begin() + offset + count - (last ? 1 : 0));
    loom.slot->deferred = last ? static_cast<std::int64_t>(prompt.back()) : -1;
    if (!input.empty())
      model_->Check(loom_text_embed_row_prefill(embed, loom.row(), input.data(),
                                                input.size()),
                    "prefill");
    loom.slot->position += count;
    return {.consumed_tokens = count, .decode_ready = last};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& loom = Require(state);
    Unconstrain(loom, sampler);
    std::lock_guard lock(model_->mutex());
    PrepareLocked(loom, sampler.config());
    if (const auto forced = sampler.ForcedToken())
      return {.token = *forced, .piece = PieceLocked(loom, *forced)};
    return SelectionLocked(loom);
  }

  void Advance(ext::TextRunnerState& state,
               TextRunnerToken token) const override {
    auto& loom = Require(state);
    std::lock_guard lock(model_->mutex());
    AdvanceLocked(loom, token);
  }

  void AdvanceBatch(
      std::span<const ext::TextRunnerAdvance> advances) const override {
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    std::vector<std::size_t> packed;
    std::vector<std::uint32_t> rows;
    for (std::size_t i = 0; i < advances.size(); ++i) {
      try {
        auto& loom = Require(advances[i].state.get());
        if (loom.slot->deferred >= 0 ||
            static_cast<TextRunnerToken>(loom_text_embed_row_token(
                embed, loom.row())) != advances[i].token) {
          AdvanceLocked(loom, advances[i].token);
          continue;
        }
        EnsureExtentLocked(loom, 2);
        packed.push_back(i);
        rows.push_back(loom.row());
      } catch (...) {
        if (advances[i].failure == nullptr)
          throw;
        *advances[i].failure = std::current_exception();
      }
    }
    if (rows.empty())
      return;
    try {
      model_->Check(
          loom_text_embed_decode_batch(embed, rows.data(), rows.size()),
          "decode");
      for (const auto i : packed)
        CommitLocked(Require(advances[i].state.get()));
    } catch (...) {
      for (const auto i : packed) {
        if (advances[i].failure == nullptr)
          throw;
        *advances[i].failure = std::current_exception();
      }
    }
  }

  [[nodiscard]] ext::TextDecodeStep DecodeStep(
      ext::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    const ext::TextRunnerDecode decode{state, max_tokens, sampler};
    auto steps = DecodeBatch(std::span(&decode, 1));
    if (steps.front().failure)
      std::rethrow_exception(steps.front().failure);
    return std::move(steps.front());
  }

  // One packed epoch for every session Gufo batches; drafted when the
  // package has a drafter. After the epoch each row's next selection is
  // known, so a response that ends here releases its withheld output (tool
  // calls) with its last token.
  [[nodiscard]] std::vector<ext::TextDecodeStep> DecodeBatch(
      std::span<const ext::TextRunnerDecode> decodes) const override {
    std::vector<ext::TextDecodeStep> steps(decodes.size());
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    const std::uint32_t depth = model_->info().draft_depth;
    std::vector<std::size_t> active;
    std::vector<std::uint32_t> rows, limits;
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      try {
        auto& loom = Require(decodes[i].state.get());
        const std::size_t remaining = decodes[i].max_tokens;
        if (remaining == 0)
          continue;
        auto& sampler = decodes[i].sampler.get();
        Unconstrain(loom, sampler);
        PrepareLocked(loom, sampler.config());
        if (const auto forced = sampler.ForcedToken()) {
          // Gufo's reasoning budget forces this token in place of the
          // row's own selection.
          steps[i].selections.push_back(
              {.token = *forced, .piece = PieceLocked(loom, *forced)});
          AdvanceLocked(loom, *forced);
          FinishLocked(loom, steps[i], remaining);
          continue;
        }
        auto selection = SelectionLocked(loom);
        if (selection.stop) {
          steps[i].stop = true;
          continue;
        }
        const std::uint32_t limit =
            depth == 0 || remaining < 2
                ? 0
                : static_cast<std::uint32_t>(
                      std::min<std::size_t>(depth + 1, remaining));
        EnsureExtentLocked(loom, limit ? depth + 2 : 2);
        steps[i].selections.push_back(std::move(selection));
        active.push_back(i);
        rows.push_back(loom.row());
        limits.push_back(limit);
      } catch (...) {
        steps[i].failure = std::current_exception();
      }
    }
    if (active.empty())
      return steps;
    try {
      const bool drafted =
          std::any_of(limits.begin(), limits.end(), [](auto l) { return l; });
      if (!drafted) {
        model_->Check(
            loom_text_embed_decode_batch(embed, rows.data(), rows.size()),
            "decode");
        for (std::size_t k = 0; k < active.size(); ++k) {
          auto& loom = Require(decodes[active[k]].state.get());
          CommitLocked(loom);
          FinishLocked(loom, steps[active[k]], decodes[active[k]].max_tokens);
        }
        return steps;
      }
      std::vector<std::int32_t> tokens(16 * rows.size());
      std::vector<std::uint32_t> counts(rows.size()), consumed(rows.size());
      model_->Check(loom_text_embed_verify_batch(
                        embed, rows.data(), rows.size(), limits.data(),
                        tokens.data(), counts.data(), consumed.data()),
                    "verify");
      for (std::size_t k = 0; k < active.size(); ++k) {
        auto& step = steps[active[k]];
        auto& loom = Require(decodes[active[k]].state.get());
        CommitLocked(loom);
        // Every output but the last entered state; the last stays pending.
        for (std::uint32_t j = 0; j + 1 < counts[k]; ++j) {
          const auto token = static_cast<TextRunnerToken>(tokens[16 * k + j]);
          step.selections.push_back(
              {.token = token, .piece = PieceLocked(loom, token)});
          CommitLocked(loom);
        }
        if (limits[k]) {
          step.draft_rounds = 1;
          step.draft_tokens = depth;
          step.draft_accepted_tokens = counts[k] ? counts[k] - 1 : 0;
        }
        FinishLocked(loom, step, decodes[active[k]].max_tokens);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto i : active)
        steps[i].failure = failure;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    return dynamic_cast<const LoomState&>(state).position();
  }

private:
  std::string DecodeLocked(const std::vector<std::int32_t>& tokens) const {
    if (tokens.empty())
      return {};
    std::string text(16 * tokens.size() + 64, '\0');
    for (int attempt = 0;; ++attempt) {
      std::size_t size = 0;
      if (loom_text_embed_detokenize(model_->embed(), tokens.data(),
                                     tokens.size(), text.data(), text.size(),
                                     &size) == 0) {
        text.resize(size);
        return text;
      }
      if (attempt == 6 ||
          std::string_view(loom_text_embed_error(model_->embed()))
                  .find("RESOURCE_EXHAUSTED") == std::string_view::npos)
        model_->Check(1, "detokenize");
      text.resize(text.size() * 4);
    }
  }

  // Releases the request's grammar once: the package samples on device, so
  // Gufo's token masks cannot apply; the package's parser reads the output.
  static void Unconstrain(LoomState& loom, sampling::SamplerState& sampler) {
    if (!loom.unconstrained) {
      sampler = sampler.WithoutConstraint();
      loom.unconstrained = true;
    }
  }

  void ResetRowLocked(LoomState& loom) const {
    loom_text_embed_row_end(model_->embed(), loom.row());
    model_->Check(loom_text_embed_row_reset(model_->embed(), loom.row()),
                  "reset");
    loom.slot->position = 0;
    loom.slot->deferred = -1;
    loom.slot->stale = false;
    loom.bound.clear();
  }

  // Binds the request's sampling and consumes a deferred prompt token.
  void PrepareLocked(LoomState& loom,
                     const sampling::SamplingConfig& config) const {
    auto* embed = model_->embed();
    EnsureExtentLocked(loom, 2);
    std::string body = SamplingBody(model_->info().model_id, config);
    if (body != loom.bound) {
      std::size_t count = 0;
      loom_text_embed_request_t* request = nullptr;
      model_->Check(loom_text_embed_chat_prepare(embed, body.data(),
                                                 body.size(), nullptr, 0,
                                                 &count, &request),
                    "sampling options");
      const int status = loom_text_embed_row_bind(embed, loom.row(), request);
      loom_text_embed_request_free(request);
      model_->Check(status, "bind sampling");
      loom.bound = std::move(body);
    }
    if (loom.slot->deferred >= 0) {
      const std::int32_t token = static_cast<std::int32_t>(loom.slot->deferred);
      loom.slot->deferred = -1;
      model_->Check(loom_text_embed_row_prefill(embed, loom.row(), &token, 1),
                    "prefill");
    }
  }

  ext::TextDecodeSelection SelectionLocked(LoomState& loom) const {
    auto* embed = model_->embed();
    const auto token = static_cast<TextRunnerToken>(
        loom_text_embed_row_token(embed, loom.row()));
    if (loom.stop_at_eos() && loom_text_embed_row_is_eos(embed, loom.row()))
      return {.stop = true, .token = token};
    return {.token = token, .piece = PieceLocked(loom, token)};
  }

  void AdvanceLocked(LoomState& loom, TextRunnerToken token) const {
    auto* embed = model_->embed();
    EnsureExtentLocked(loom, 2);
    if (loom.slot->deferred < 0 &&
        static_cast<TextRunnerToken>(
            loom_text_embed_row_token(embed, loom.row())) == token) {
      model_->Check(loom_text_embed_row_decode(embed, loom.row()), "decode");
    } else {
      // A token other than the pending selection replaces it, after any
      // prompt token still deferred.
      std::vector<std::int32_t> input;
      if (loom.slot->deferred >= 0)
        input.push_back(static_cast<std::int32_t>(loom.slot->deferred));
      loom.slot->deferred = -1;
      input.push_back(static_cast<std::int32_t>(token));
      model_->Check(loom_text_embed_row_prefill(embed, loom.row(), input.data(),
                                                input.size()),
                    "advance");
    }
    CommitLocked(loom);
  }

  void CommitLocked(LoomState& loom) const {
    ++loom.generated;
    ++loom.slot->position;
  }

  // Text this token settles: decoded by the row's streaming detokenizer,
  // then read by the request's output reader into canonical markup.
  std::string PieceLocked(LoomState& loom, std::int32_t token) const {
    const std::pair<std::size_t, std::int32_t> key{loom.generated, token};
    if (loom.piece_key == key)
      return loom.piece;
    if (loom.piece_key && loom.piece_key->first == key.first)
      throw std::logic_error("loom selection changed before it was consumed");
    std::string raw(256, '\0');
    std::size_t size = 0;
    model_->Check(loom_text_embed_decoder_feed(loom.decoder, &token, 1,
                                               raw.data(), raw.size(), &size),
                  "decode");
    raw.resize(size);
    loom.piece_key = key;
    loom.piece = OutputLocked(loom, raw, false);
    loom.produced = true;
    return loom.piece;
  }

  std::string OutputLocked(LoomState& loom, std::string_view raw,
                           bool final) const {
    if (loom.context == nullptr || loom.flushed)
      return std::string(raw);
    std::string text(raw.size() + kTextBuffer, '\0');
    std::size_t size = 0;
    model_->Check(loom_text_embed_output_feed(loom.context->request, raw.data(),
                                              raw.size(), final, text.data(),
                                              text.size(), &size),
                  "read output");
    text.resize(size);
    return text;
  }

  // Ends the response when the next selection is EOS or the token budget is
  // spent: releases withheld text and the parsed tool calls.
  void FinishLocked(LoomState& loom, ext::TextDecodeStep& step,
                    std::size_t remaining) const {
    if (loom.context == nullptr || loom.flushed || step.selections.empty())
      return;
    auto* embed = model_->embed();
    const bool eos =
        loom.stop_at_eos() && loom_text_embed_row_is_eos(embed, loom.row());
    const bool spent = step.selections.size() >= remaining;
    if (!eos && !spent)
      return;
    std::string tail = OutputLocked(loom, {}, true);
    std::string json(kTextBuffer, '\0');
    std::size_t size = 0;
    if (loom_text_embed_output_tools(loom.context->request, eos ? 0 : 1,
                                     json.data(), json.size(), &size) == 0) {
      json.resize(size);
      tail += ToolMarkup(json);
    }
    loom.flushed = true;
    step.selections.back().piece += tail;
  }

  // Grows the row's reservation before a step that consumes `step` tokens.
  void EnsureExtentLocked(LoomState& loom, std::uint64_t step) const {
    auto* embed = model_->embed();
    const std::uint64_t position =
        loom_text_embed_row_position(embed, loom.row()) +
        (loom.slot->deferred >= 0 ? 1 : 0);
    const std::uint64_t context = model_->info().context_capacity;
    if (position + step <= loom_text_embed_row_extent(embed, loom.row()))
      return;
    const std::uint64_t extent =
        std::min<std::uint64_t>(context, position + step + kExtentSlack);
    if (position + step > extent)
      throw std::length_error("loom row reached the context capacity");
    if (model_->Check(loom_text_embed_row_extend(embed, loom.row(), extent),
                      "reserve", 1) != 0)
      throw cache::ResourceExhausted();
    // The reservation restarted the request; bind its sampling again.
    loom.bound.clear();
  }

  std::shared_ptr<LoomModel> model_;
  std::uint32_t rows_;
  std::uint32_t width_;
};

std::shared_ptr<ext::TextModelRunner> CreateLoomRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  try {
    auto model = std::make_shared<LoomModel>(
        options.model_path, options.provider_options, options.session_count);
    return std::make_shared<LoomRunner>(std::move(model));
  } catch (const std::exception& exception) {
    if (error != nullptr)
      *error = exception.what();
    return nullptr;
  }
}

}  // namespace
}  // namespace gufo::loom

void loom_text_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_loom::kProvider_loom_text);
  registry.AddTextProvider({
      .id = "loom-text",
      .probe = &gufo::loom::ProbeLoomText,
      .create = &gufo::loom::CreateLoomRunner,
  });
}

void loom_register(gufo::ext::Registry& registry) {
  (void)registry;
}
