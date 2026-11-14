// Loom model packages as Gufo text services.
//
// A Loom package (model.json, its compiled programs and parameters, as
// `loom-fetch` installs them) is a complete text service: its programs own the
// chat template, tokenizer, tool syntax, sampling, speculative decoding and
// batching. This provider runs the package in `loom-serve`, a child process
// on a loopback port, and adapts each Gufo chat request to it. Gufo keeps its
// HTTP APIs (OpenAI chat and responses, Anthropic messages, llama-server
// routes), logging and admission in front.

#include <curl/curl.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gufo/ext/generated/loom_constants.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text_service.hpp>

namespace gufo::loom {
namespace {

namespace fs = std::filesystem;
using server::TextGenerationBackend;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kDefaultDevice =
    "amdgpu://?completion_spin_ns=4294967295";
constexpr std::int64_t kDefaultMemoryBytes = std::int64_t{32} << 30;
constexpr auto kStartTimeout = std::chrono::minutes(30);

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

std::optional<json::Value> ReadJson(const fs::path& path,
                                    std::size_t max_bytes) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error || size > max_bytes)
    return std::nullopt;
  std::ifstream in(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)), {});
  try {
    return json::parse(text);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// A Loom model card: schema_version, programs and a service of kind "text".
bool IsLoomTextManifest(const json::Value& card) {
  if (!card.is_object() || !card.contains("schema_version") ||
      !card.contains("programs"))
    return false;
  const auto* services = card.find("services");
  if (services == nullptr || !services->is_object())
    return false;
  for (const auto& [name, service] : services->members()) {
    (void)name;
    if (service.is_object() && service.member_str("kind") == "text")
      return true;
  }
  return false;
}

bool ProbeLoomText(std::string_view model_root) {
  const auto manifest = ManifestPath(model_root);
  if (!manifest)
    return false;
  const auto card = ReadJson(*manifest, std::size_t{4} << 20);
  return card && IsLoomTextManifest(*card);
}

// ----------------------------------------------------------------- http ---

struct HttpReply {
  long status{0};
  std::string body;
};

struct CurlGlobal {
  CurlGlobal() { curl_global_init(CURL_GLOBAL_DEFAULT); }
};

struct CurlEasy {
  CURL* handle{curl_easy_init()};
  curl_slist* headers{nullptr};
  ~CurlEasy() {
    curl_slist_free_all(headers);
    curl_easy_cleanup(handle);
  }
};

std::size_t AppendBody(char* data, std::size_t size, std::size_t count,
                       void* user) {
  static_cast<std::string*>(user)->append(data, size * count);
  return size * count;
}

std::optional<HttpReply> Request(const std::string& url,
                                 const std::string* post_body,
                                 long timeout_ms) {
  static CurlGlobal global;
  CurlEasy curl;
  if (curl.handle == nullptr)
    return std::nullopt;
  HttpReply reply;
  curl_easy_setopt(curl.handle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl.handle, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl.handle, CURLOPT_TIMEOUT_MS, timeout_ms);
  curl_easy_setopt(curl.handle, CURLOPT_WRITEFUNCTION, &AppendBody);
  curl_easy_setopt(curl.handle, CURLOPT_WRITEDATA, &reply.body);
  if (post_body != nullptr) {
    curl.headers =
        curl_slist_append(curl.headers, "Content-Type: application/json");
    curl_easy_setopt(curl.handle, CURLOPT_HTTPHEADER, curl.headers);
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDS, post_body->c_str());
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(post_body->size()));
  }
  if (curl_easy_perform(curl.handle) != CURLE_OK)
    return std::nullopt;
  curl_easy_getinfo(curl.handle, CURLINFO_RESPONSE_CODE, &reply.status);
  return reply;
}

std::string ErrorMessage(const HttpReply& reply) {
  try {
    const auto body = json::parse(reply.body);
    if (const auto* error = body.find("error"); error && error->is_object())
      return error->member_str("message", reply.body);
  } catch (const std::exception&) {
  }
  return reply.body;
}

// ---------------------------------------------------------------- child ---

int FreeLoopbackPort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    throw std::runtime_error("cannot open a socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t length = sizeof(address);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot reserve a loopback port");
  }
  ::close(fd);
  return ntohs(address.sin_port);
}

fs::path FindLoomServe(const ext::ParsedOptions& options) {
  if (const auto* path = ext::FindOption<std::string>(options, "--loom-serve"))
    return *path;
  if (const char* env = std::getenv("LOOM_SERVE"); env && *env)
    return env;
  if (const char* path = std::getenv("PATH")) {
    std::stringstream dirs(path);
    for (std::string dir; std::getline(dirs, dir, ':');) {
      const auto candidate = fs::path(dir) / "loom-serve";
      if (::access(candidate.c_str(), X_OK) == 0)
        return candidate;
    }
  }
  throw std::runtime_error(
      "loom-serve not found: install hrx-loom (pip install hrx-loom, or "
      "`nix build github:gufo-org/gufo#hrx-loom`), or pass --loom-serve");
}

// loom-serve on a loopback port; it dies with Gufo.
class LoomServe {
public:
  LoomServe(const fs::path& program, std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(program.c_str()));
    for (auto& arg : args)
      argv.push_back(arg.data());
    argv.push_back(nullptr);
    const pid_t parent = ::getpid();
    pid_ = ::fork();
    if (pid_ < 0)
      throw std::runtime_error("cannot start loom-serve");
    if (pid_ == 0) {
      ::prctl(PR_SET_PDEATHSIG, SIGTERM);
      if (::getppid() != parent)
        ::_exit(1);
      // loom-serve logs JSON lines on stdout; keep Gufo's stdout clean.
      ::dup2(STDERR_FILENO, STDOUT_FILENO);
      // Gufo's sockets (its listening port among them) stay with Gufo.
      ::close_range(3, ~0U, 0);
      ::execv(program.c_str(), argv.data());
      ::_exit(127);
    }
  }
  ~LoomServe() {
    if (pid_ <= 0)
      return;
    ::kill(pid_, SIGTERM);
    for (int i = 0; i < 100 && !Exited(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (!Exited()) {
      ::kill(pid_, SIGKILL);
      ::waitpid(pid_, nullptr, 0);
    }
  }
  LoomServe(const LoomServe&) = delete;
  LoomServe& operator=(const LoomServe&) = delete;

  bool Running() { return !Exited(); }

private:
  bool Exited() {
    if (exited_)
      return true;
    int status = 0;
    exited_ = ::waitpid(pid_, &status, WNOHANG) == pid_;
    return exited_;
  }
  pid_t pid_{-1};
  bool exited_{false};
};

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

std::string ArgumentsJson(const ext::ChatMessage::ToolCall& call) {
  json::Value arguments = json::Value::object();
  for (const auto& argument : call.arguments) {
    arguments[argument.name] =
        argument.is_string ? json::Value(argument.value)
                           : ParseOr(argument.value, json::Value(argument.value));
  }
  return arguments.dump();
}

json::Value Messages(const ext::ChatRequest& request) {
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
          json::Value function = json::Value::object();
          function["name"] = call.name;
          function["arguments"] = ArgumentsJson(call);
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
  return messages;
}

json::Value Tools(const ext::ChatRequest& request) {
  json::Value tools = json::Value::array();
  for (const auto& tool : request.tools) {
    if (!tool.definition_json.empty()) {
      auto definition = ParseOr(tool.definition_json, json::Value());
      if (definition.is_object()) {
        tools.push_back(std::move(definition));
        continue;
      }
    }
    json::Value function = json::Value::object();
    function["name"] = tool.name;
    if (!tool.description.empty())
      function["description"] = tool.description;
    function["parameters"] =
        ParseOr(tool.parameters_json, json::Value::object());
    json::Value entry = json::Value::object();
    entry["type"] = "function";
    entry["function"] = std::move(function);
    tools.push_back(std::move(entry));
  }
  return tools;
}

// The OpenAI chat body loom-serve accepts (it rejects unknown fields).
std::string ChatBody(const std::string& model, const ext::ChatRequest& request,
                     std::size_t max_tokens,
                     const sampling::SamplingConfig& sampling, bool stream) {
  json::Value body = json::Value::object();
  body["model"] = model;
  body["messages"] = Messages(request);
  if (!request.tools.empty()) {
    body["tools"] = Tools(request);
    switch (request.tool_choice) {
      case ext::ChatRequest::ToolChoice::kAuto:
        break;
      case ext::ChatRequest::ToolChoice::kNone:
        body["tool_choice"] = "none";
        break;
      case ext::ChatRequest::ToolChoice::kRequired:
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
        break;
    }
  }
  if (request.reasoning.enabled.has_value()) {
    json::Value kwargs = json::Value::object();
    kwargs["enable_thinking"] = *request.reasoning.enabled;
    body["chat_template_kwargs"] = std::move(kwargs);
  }
  if (max_tokens > 0)
    body["max_tokens"] = max_tokens;
  body["temperature"] = static_cast<double>(sampling.temperature);
  body["top_p"] = static_cast<double>(sampling.top_p);
  if (sampling.top_k > 0)
    body["top_k"] = sampling.top_k;
  if (sampling.seed >= 0)
    body["seed"] = sampling.seed;
  if (stream) {
    body["stream"] = true;
    json::Value options = json::Value::object();
    options["include_usage"] = true;
    body["stream_options"] = std::move(options);
  }
  return body.dump();
}

// ---------------------------------------------------- streamed answers ---

// Turns loom-serve's parsed deltas back into text Gufo's response parsers
// read: reasoning inside <think></think>, tool calls as <tool_call> JSON
// blocks (Gufo's JSON tool dialect), content as is. Applies stop sequences,
// which loom-serve does not take.
class Stream {
public:
  Stream(const ext::ChatRequest& request,
         const TextGenerationBackend::CancellationCheck& is_cancelled,
         const TextGenerationBackend::TokenCallback& on_token)
      : stops_(request.stop_sequences),
        is_cancelled_(is_cancelled),
        on_token_(on_token),
        start_(Clock::now()) {
    for (const auto& stop : stops_)
      hold_ = std::max(hold_, stop.empty() ? 0 : stop.size() - 1);
  }

  // Bytes from curl; false aborts the transfer.
  bool Feed(std::string_view bytes) {
    if (raw.size() < 8192)
      raw.append(bytes.substr(0, 8192 - raw.size()));
    buffer_.append(bytes);
    for (std::size_t end; (end = buffer_.find('\n')) != std::string::npos;) {
      std::string line = buffer_.substr(0, end);
      buffer_.erase(0, end + 1);
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (line.rfind("data:", 0) != 0)
        continue;
      std::string_view data(line);
      data.remove_prefix(5);
      while (!data.empty() && data.front() == ' ')
        data.remove_prefix(1);
      if (data == "[DONE]")
        continue;
      if (!Event(data))
        return false;
    }
    return !Cancelled();
  }

  bool Cancelled() {
    if (!result.cancelled && is_cancelled_ && is_cancelled_())
      result.cancelled = true;
    return result.cancelled;
  }

  void Finish() {
    if (!stopped_) {
      FlushContent(/*final=*/true);
      CloseReasoning();
      for (const auto& [index, call] : calls_) {
        (void)index;
        auto arguments = ParseOr(call.arguments, json::Value::object());
        json::Value block = json::Value::object();
        block["name"] = call.name;
        block["arguments"] = std::move(arguments);
        Emit("<tool_call>\n" + block.dump() + "\n</tool_call>");
      }
    }
    const double total = Ms(Clock::now());
    result.ttft_ms = first_ms_.value_or(total);
    result.prefill_ms = result.ttft_ms;
    result.decode_ms = total - result.ttft_ms;
    if (result.completion_tokens > 1)
      result.mean_inter_token_ms =
          result.decode_ms / static_cast<double>(result.completion_tokens - 1);
    result.execution_plan = "loom";
    if (result.cancelled)
      result.finish_reason = TextGenerationBackend::FinishReason::kCancelled;
  }

  std::string error;
  // The first bytes of the response, for an error body that is not events.
  std::string raw;
  TextGenerationBackend::Result result;

private:
  struct Call {
    std::string name;
    std::string arguments;
  };

  bool Event(std::string_view data) {
    json::Value event;
    try {
      event = json::parse(data);
    } catch (const std::exception&) {
      return true;
    }
    if (const auto* failure = event.find("error"); failure != nullptr) {
      error = failure->is_object() ? failure->member_str("message") : data;
      return false;
    }
    if (const auto* usage = event.find("usage"); usage && usage->is_object()) {
      result.prompt_tokens = usage->member_size("prompt_tokens");
      result.completion_tokens = usage->member_size("completion_tokens");
      if (const auto* details = usage->find("prompt_tokens_details"))
        result.cached_prompt_tokens = details->member_size("cached_tokens");
    }
    const auto* choices = event.find("choices");
    if (choices == nullptr || !choices->is_array())
      return true;
    for (const auto& choice : choices->items()) {
      if (const auto* delta = choice.find("delta"); delta && !stopped_) {
        if (const auto* text = delta->find("reasoning_content");
            text && text->is_string() && !text->str().empty()) {
          if (!reasoning_open_) {
            reasoning_open_ = true;
            Emit("<think>");
          }
          Emit(text->str());
        }
        if (const auto* text = delta->find("content");
            text && text->is_string() && !text->str().empty()) {
          CloseReasoning();
          content_ += text->str();
          if (!FlushContent(/*final=*/false))
            return false;
        }
        if (const auto* calls = delta->find("tool_calls");
            calls && calls->is_array()) {
          CloseReasoning();
          for (const auto& call : calls->items()) {
            auto& entry = calls_[call.member_size("index")];
            if (const auto* function = call.find("function")) {
              entry.name += function->member_str("name");
              entry.arguments += function->member_str("arguments");
            }
          }
          MarkFirst();
        }
      }
      const std::string finish = choice.member_str("finish_reason");
      if (finish == "length")
        result.finish_reason = TextGenerationBackend::FinishReason::kLength;
    }
    return !stopped_ && !Cancelled();
  }

  void CloseReasoning() {
    if (reasoning_open_) {
      reasoning_open_ = false;
      Emit("</think>");
    }
  }

  // Emits content up to a stop sequence, holding back a possible prefix.
  bool FlushContent(bool final) {
    std::size_t cut = std::string::npos;
    std::string_view matched;
    for (const auto& stop : stops_) {
      if (stop.empty())
        continue;
      const auto at = content_.find(stop);
      if (at != std::string::npos && at < cut) {
        cut = at;
        matched = stop;
      }
    }
    if (cut != std::string::npos) {
      Emit(content_.substr(0, cut));
      content_.clear();
      stopped_ = true;
      result.finish_reason = TextGenerationBackend::FinishReason::kStopSequence;
      result.stop_sequence = std::string(matched);
      return false;
    }
    const std::size_t keep = final ? 0 : std::min(hold_, content_.size());
    Emit(content_.substr(0, content_.size() - keep));
    content_.erase(0, content_.size() - keep);
    return true;
  }

  void Emit(const std::string& text) {
    if (text.empty())
      return;
    MarkFirst();
    result.text += text;
    if (on_token_ && !on_token_(server::TokenPiece(text)))
      result.cancelled = true;
  }

  void MarkFirst() {
    if (!first_ms_)
      first_ms_ = Ms(Clock::now());
  }

  double Ms(Clock::time_point at) const {
    return std::chrono::duration<double, std::milli>(at - start_).count();
  }

  std::vector<std::string> stops_;
  std::size_t hold_{0};
  const TextGenerationBackend::CancellationCheck& is_cancelled_;
  const TextGenerationBackend::TokenCallback& on_token_;
  Clock::time_point start_;
  std::optional<double> first_ms_;
  std::string buffer_;
  std::string content_;
  std::map<std::size_t, Call> calls_;
  bool reasoning_open_{false};
  bool stopped_{false};
};

std::size_t StreamBytes(char* data, std::size_t size, std::size_t count,
                        void* user) {
  auto* stream = static_cast<Stream*>(user);
  return stream->Feed(std::string_view(data, size * count)) ? size * count : 0;
}

int StreamProgress(void* user, curl_off_t, curl_off_t, curl_off_t,
                   curl_off_t) {
  return static_cast<Stream*>(user)->Cancelled() ? 1 : 0;
}

// --------------------------------------------------------------- service ---

class LoomTextService final : public TextGenerationBackend {
public:
  explicit LoomTextService(const ext::TextServiceOptions& options)
      : served_names_(options.served_model_names) {
    const auto manifest = ManifestPath(options.model_path);
    if (!manifest)
      throw std::invalid_argument("not a Loom package: " + options.model_path);
    manifest_ = fs::absolute(*manifest);
    const auto& provider = options.provider_options;
    const int port = FreeLoopbackPort();
    base_url_ = "http://127.0.0.1:" + std::to_string(port);
    std::string device(kDefaultDevice);
    if (const auto* value = ext::FindOption<std::string>(provider,
                                                         "--loom-device"))
      device = *value;
    std::int64_t memory = kDefaultMemoryBytes;
    if (const auto* value = ext::FindOption<std::int64_t>(
            provider, "--loom-memory-bytes"))
      memory = *value;
    std::vector<std::string> args{
        "--device=" + device,
        "--listen=127.0.0.1:" + std::to_string(port),
        "--model=" + manifest_.string(),
        "--memory_bytes=" + std::to_string(memory),
    };
    if (const auto* value = ext::FindOption<std::string>(provider,
                                                         "--loom-preset"))
      args.push_back("--preset=" + *value);
    const auto program = FindLoomServe(provider);
    std::cerr << "loom: starting " << program.string() << " for "
              << manifest_.string() << '\n';
    child_ = std::make_unique<LoomServe>(program, std::move(args));
    WaitReady();
  }

  [[nodiscard]] std::string model_id() const override {
    return served_names_.empty() ? loom_id_ : served_names_.front();
  }
  [[nodiscard]] std::vector<std::string> model_ids() const override {
    return served_names_.empty() ? std::vector<std::string>{loom_id_}
                                 : served_names_;
  }
  [[nodiscard]] bool ready() const override {
    return child_ != nullptr && child_->Running();
  }
  [[nodiscard]] std::uint32_t max_context() const override {
    return context_;
  }
  [[nodiscard]] std::string model_path() const override {
    return manifest_.string();
  }
  [[nodiscard]] std::string build_info() const override { return "loom"; }

  Result chat(const ext::ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled,
              const TokenCallback& on_token) override {
    const std::string body =
        ChatBody(loom_id_, request, max_tokens, sampling, /*stream=*/true);
    Stream stream(request, is_cancelled, on_token);
    stream.result.client_id = request.client_id;
    CurlEasy curl;
    if (curl.handle == nullptr)
      throw std::runtime_error("loom: cannot create an HTTP request");
    const std::string url = base_url_ + "/v1/chat/completions";
    curl.headers =
        curl_slist_append(curl.headers, "Content-Type: application/json");
    curl_easy_setopt(curl.handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.handle, CURLOPT_HTTPHEADER, curl.headers);
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body.size()));
    curl_easy_setopt(curl.handle, CURLOPT_WRITEFUNCTION, &StreamBytes);
    curl_easy_setopt(curl.handle, CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(curl.handle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl.handle, CURLOPT_XFERINFOFUNCTION, &StreamProgress);
    curl_easy_setopt(curl.handle, CURLOPT_XFERINFODATA, &stream);
    const CURLcode code = curl_easy_perform(curl.handle);
    long status = 0;
    curl_easy_getinfo(curl.handle, CURLINFO_RESPONSE_CODE, &status);
    if (status >= 400) {
      // An error body is one JSON object, not events.
      throw std::invalid_argument(
          "loom: " + ErrorMessage(HttpReply{.status = status, .body = stream.raw}));
    }
    if (!stream.error.empty())
      throw std::runtime_error("loom: " + stream.error);
    if (code != CURLE_OK && code != CURLE_WRITE_ERROR &&
        code != CURLE_ABORTED_BY_CALLBACK)
      throw std::runtime_error(std::string("loom: ") +
                               curl_easy_strerror(code));
    stream.Finish();
    return std::move(stream.result);
  }

  Result complete(std::string_view, std::size_t,
                  const sampling::SamplingConfig&, const CancellationCheck&,
                  const TokenCallback&, std::string_view,
                  const std::vector<std::string>&, bool, bool) override {
    throw std::invalid_argument(
        "Loom packages serve chat requests; raw completions are unsupported");
  }

  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override {
    const auto tokens = tokenize(text);
    return tokens ? tokens->size() : 0;
  }

  [[nodiscard]] std::optional<std::vector<std::uint32_t>> tokenize(
      std::string_view text) const override {
    json::Value body = json::Value::object();
    body["model"] = loom_id_;
    body["content"] = std::string(text);
    const std::string payload = body.dump();
    const auto reply = Request(base_url_ + "/tokenize", &payload, 60000);
    if (!reply || reply->status != 200)
      return std::nullopt;
    std::vector<std::uint32_t> tokens;
    try {
      const auto parsed = json::parse(reply->body);
      if (const auto* list = parsed.find("tokens"); list && list->is_array())
        for (const auto& token : list->items())
          tokens.push_back(static_cast<std::uint32_t>(token.as_size()));
    } catch (const std::exception&) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::size_t> count_chat_tokens(
      const ext::ChatRequest& request) const override {
    const std::string payload =
        ChatBody(loom_id_, request, 0, sampling::SamplingConfig{}, false);
    const auto reply = Request(
        base_url_ + "/v1/chat/completions/input_tokens", &payload, 60000);
    if (!reply || reply->status != 200)
      return std::nullopt;
    try {
      return json::parse(reply->body).member_size("input_tokens");
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }

private:
  // Until the package compiles and registers; the first run compiles for
  // tens of seconds, later runs reuse Loom's cache.
  void WaitReady() {
    const auto deadline = Clock::now() + kStartTimeout;
    while (Clock::now() < deadline) {
      if (!child_->Running())
        throw std::runtime_error("loom-serve exited during startup");
      if (const auto reply = Request(base_url_ + "/v1/status", nullptr, 5000);
          reply && reply->status == 200) {
        const auto status = json::parse(reply->body);
        const auto* data = status.find("data");
        if (data && data->is_array() && !data->items().empty()) {
          const auto& model = data->items().front();
          const auto* loom = model.find("loom");
          const std::string registration =
              loom ? loom->member_str("registration") : "";
          if (registration == "failed")
            throw std::runtime_error("loom-serve failed to register " +
                                     manifest_.string() + ": " +
                                     reply->body.substr(0, 4000));
          if (registration == "ready") {
            loom_id_ = model.member_str("id");
            context_ = static_cast<std::uint32_t>(
                model.member_size("context_length"));
            return;
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    throw std::runtime_error("loom-serve did not become ready in time");
  }

  std::vector<std::string> served_names_;
  fs::path manifest_;
  std::string base_url_;
  std::string loom_id_;
  std::uint32_t context_{0};
  std::unique_ptr<LoomServe> child_;
};

std::shared_ptr<TextGenerationBackend> CreateLoomTextService(
    const ext::TextServiceOptions& options, std::string* error) {
  try {
    return std::make_shared<LoomTextService>(options);
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
      .create_service = &gufo::loom::CreateLoomTextService,
  });
}

void loom_register(gufo::ext::Registry& registry) {
  (void)registry;
}
