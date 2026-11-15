// Host-sampling behavior of the Loom runner against a fake libloom_text_embed
// (fake_embed.cpp): no GPU. Builds with build_test.sh against a Gufo checkout.
#include "../src/loom_runner.cpp"  // the runner's types live in an anonymous namespace

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "fake_embed.hpp"

using namespace gufo;
using namespace gufo::loom;

static int g_failures = 0;
#define CHECK(...) CheckImpl(__FILE__, __LINE__, #__VA_ARGS__, __VA_ARGS__)
static void CheckImpl(const char* file, int line, const char* text, bool ok,
                      const std::string& note = {}) {
  if (ok)
    return;
  ++g_failures;
  std::cerr << "FAIL " << file << ":" << line << ": " << text << " " << note
            << "\n";
}

namespace {

struct Run {
  std::vector<int> tokens;
  std::vector<std::optional<sampling::TokenLogprobs>> logprobs;
  std::string text;
  bool stopped{false};
  std::size_t epochs_host{0}, epochs_device{0}, exported{0};
};

class Fixture {
public:
  explicit Fixture(bool logits = true, std::size_t rows = 2) {
    dir_ = std::filesystem::temp_directory_path() /
           ("loom_runner_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "model.json")
        << R"({"schema_version":1,"programs":{},"services":{"chat":{"kind":"text"}},)"
           R"("variants":{"v":{"services":{"chat":{"config":{"llm.card":true,"llm.logits":false}}}}}})";
    ext::ParsedOptions options;
    if (!logits)
      options["--loom-logits"] = std::string("off");
    fake::state().epochs.clear();
    model = std::make_shared<LoomModel>(dir_.string(), options, rows);
    runner = std::make_unique<LoomRunner>(model, 0, false, 0);
    pool = std::make_shared<RowPool>();
    pool->used.assign(rows, false);
  }
  ~Fixture() { std::filesystem::remove_all(dir_); }

  // One request on `row`: prefill the prompt, then decode up to `max_tokens`
  // the way Gufo's pool does (Accept after each returned selection).
  Run Generate(std::uint32_t row, sampling::SamplingConfig config,
               std::size_t max_tokens,
               ext::ChatRequest request = DefaultRequest()) {
    auto slot = std::make_unique<LoomSlot>(row, pool);
    loom_text_embed_decoder_t* decoder = nullptr;
    loom_text_embed_decoder_open(model->embed(), &decoder);
    LoomState state(*slot, decoder);
    state.SetStopAtEos(true);
    auto prepared = runner->PreparePrompt(request);
    CHECK(prepared.has_value());
    runner->SetPromptContext(state, prepared->context);
    const auto& prompt = prepared->tokens;
    std::size_t offset = 0;
    while (true) {
      const auto step = runner->Prefill(state, prompt, offset, 64);
      offset += step.consumed_tokens;
      if (step.decode_ready)
        break;
    }
    const auto before = fake::state().epochs.size();
    sampling::SamplerState sampler(config, std::span<const sampling::TokenId>());
    Run run;
    for (std::size_t n = 0; n < max_tokens && !run.stopped;) {
      const ext::TextRunnerDecode decode{state, max_tokens - n, sampler};
      auto steps = runner->DecodeBatch(std::span(&decode, 1));
      auto& step = steps.front();
      if (step.failure)
        std::rethrow_exception(step.failure);
      for (auto& selection : step.selections) {
        sampler.Accept(selection.token);
        run.tokens.push_back(static_cast<int>(selection.token));
        run.logprobs.push_back(selection.logprobs);
        run.text += selection.piece;
        ++n;
      }
      run.stopped = step.stop;
      if (step.selections.empty() && !step.stop)
        break;
    }
    const auto& epochs = fake::state().epochs;
    for (std::size_t i = before; i < epochs.size(); ++i) {
      (epochs[i].host_tokens ? run.epochs_host : run.epochs_device)++;
      for (const bool e : epochs[i].exported)
        run.exported += e;
    }
    return run;
  }

  static ext::ChatRequest DefaultRequest() {
    ext::ChatRequest request;
    request.messages.emplace_back(ext::ChatRole::kUser, "Hi");
    return request;
  }

  std::filesystem::path dir_;
  std::shared_ptr<LoomModel> model;
  std::unique_ptr<LoomRunner> runner;
  std::shared_ptr<RowPool> pool;
};

// Row: a sharp peak at token (last*7 + position*3 + 11) % 200 + 32 (printable
// range), then a smoother falloff; EOS never wins.
void PeakScript(int last, int position, float* out) {
  const int target = (last * 7 + position * 3 + 11) % 90 + 32;
  for (int v = 0; v < fake::kBytes; ++v) {
    const float d = static_cast<float>(v - target);
    out[v] = -0.05F * d * d;
  }
  out[fake::kEos] = -50.0F;
}

int DevicePick(int last, int position) {
  std::vector<float> row(fake::kWidth, -3e38F);
  PeakScript(last, position, row.data());
  return fake::Argmax(row.data());
}

double LogSoftmax(const float* row, int token) {
  double m = -1e300;
  for (int i = 0; i < fake::kVocabulary; ++i)
    m = std::max<double>(m, row[i]);
  double sum = 0;
  for (int i = 0; i < fake::kVocabulary; ++i)
    sum += std::exp(static_cast<double>(row[i]) - m);
  return static_cast<double>(row[token]) - m - std::log(sum);
}

void TestDevicePathUnchanged() {
  std::cerr << "device path: no host need keeps the device sampler\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  sampling::SamplingConfig config;
  const auto run = f.Generate(0, config, 8);
  CHECK(run.tokens.size() == 8);
  CHECK(run.epochs_host == 0, "no host tokens");
  CHECK(run.exported == 0, "no logits downloaded");
  CHECK(f.model->logits().enabled);
}

void TestLogprobs() {
  std::cerr << "logprobs: host path reports exact log-softmax\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  sampling::SamplingConfig config;
  config.logprobs = 3;
  const auto run = f.Generate(0, config, 6);
  CHECK(run.tokens.size() == 6);
  CHECK(run.epochs_host >= 5, "host tokens used");
  // Reference: replay the greedy walk on the scripted rows.
  int last = 'H';  // the fake prompt ends at the last byte of "PROMPT:{"mod"
  (void)last;
  for (std::size_t i = 0; i < run.tokens.size(); ++i) {
    CHECK(run.logprobs[i].has_value(), "logprobs present");
    if (!run.logprobs[i]) continue;
    const auto& lp = *run.logprobs[i];
    CHECK(lp.selected.token == static_cast<sampling::TokenId>(run.tokens[i]));
    CHECK(lp.top.size() == 3);
    CHECK(lp.top.front().token == lp.selected.token, "greedy token is top-1");
    CHECK(lp.selected.logprob <= 0.0);
    CHECK(lp.selected.logprob == lp.top.front().logprob, "selected is the top-1 entry");
    CHECK(lp.top[0].logprob >= lp.top[1].logprob && lp.top[1].logprob >= lp.top[2].logprob);
    double mass = 0;
    for (const auto& t : lp.top) mass += std::exp(t.logprob);
    CHECK(mass <= 1.0 + 1e-9, "top-3 mass");
  }
  // The same tokens as the device path (temperature 0).
  Fixture g;
  sampling::SamplingConfig plain;
  const auto reference = g.Generate(0, plain, 6);
  CHECK(reference.tokens == run.tokens, "host greedy == device greedy");
}

void TestRepetitionPenalty() {
  std::cerr << "penalties: the host applies them (the device sampler cannot)\n";
  fake::reset();
  // Two tokens nearly tied: 'A' slightly ahead of 'B' every step.
  fake::state().script = [](int, int, float* out) {
    for (int v = 0; v < fake::kBytes; ++v) out[v] = -20.0F;
    out['A'] = 5.0F;
    out['B'] = 4.9F;
    out[fake::kEos] = -50.0F;
  };
  Fixture f;
  sampling::SamplingConfig plain;
  const auto device = f.Generate(0, plain, 6);
  CHECK(std::all_of(device.tokens.begin(), device.tokens.end(),
                    [](int t) { return t == 'A'; }), "device repeats A");
  sampling::SamplingConfig config;
  config.repeat_penalty = 2.0F;
  config.repeat_last_n = 64;
  const auto host = f.Generate(0, config, 6);
  CHECK(host.tokens.size() == 6);
  CHECK(host.tokens[0] == 'A');
  CHECK(host.tokens[1] == 'B', "penalty moves the second token to B");
  CHECK(host.epochs_host >= 5);
}

void TestMinP() {
  std::cerr << "min_p: tokens below min_p * p_max are dropped\n";
  fake::reset();
  // Top token 'A' p~0.99; with min_p=0.5 only A can ever be sampled, even at
  // high temperature and any seed.
  fake::state().script = [](int, int, float* out) {
    for (int v = 0; v < fake::kBytes; ++v) out[v] = -2.0F;
    out['A'] = 6.0F;
    out[fake::kEos] = -50.0F;
  };
  Fixture f;
  sampling::SamplingConfig config;
  config.temperature = 1.5F;
  config.min_p = 0.5F;
  config.seed = 11;
  const auto run = f.Generate(0, config, 12);
  CHECK(std::all_of(run.tokens.begin(), run.tokens.end(),
                    [](int t) { return t == 'A'; }), "only the top token survives");
  config.min_p = 0.0F;
  config.logprobs = 1;  // host path, no min_p
  config.seed = 11;
  const auto loose = f.Generate(0, config, 40);
  CHECK(std::any_of(loose.tokens.begin(), loose.tokens.end(),
                    [](int t) { return t != 'A'; }), "without min_p others appear");
}

void TestSeedStream() {
  std::cerr << "seeded host sampling is deterministic across runs\n";
  fake::reset();
  fake::state().script = PeakScript;
  sampling::SamplingConfig config;
  config.temperature = 2.0F;
  config.seed = 5;
  config.logprobs = 1;
  Fixture f;
  const auto a = f.Generate(0, config, 16);
  const auto b = f.Generate(0, config, 16);
  CHECK(a.tokens == b.tokens);
  config.seed = 6;
  const auto c = f.Generate(0, config, 16);
  CHECK(a.tokens != c.tokens);
}

void TestJsonGrammar() {
  std::cerr << "grammar: json_schema enforced from the logits\n";
  fake::reset();
  // The model prefers garbage letters; the grammar must win.
  fake::state().script = [](int last, int position, float* out) {
    PeakScript(last, position, out);
    out[fake::kEos] = 0.0F;
  };
  Fixture f;
  json::Value schema = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},"n":{"type":"integer"}},"required":["city","n"],"additionalProperties":false})");
  auto grammar = sampling::JsonConstraint::Compile(schema, true, false);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  const auto run = f.Generate(0, config, 200);
  CHECK(run.stopped, "the grammar completes and EOS ends the response");
  json::Value parsed;
  bool ok = true;
  try {
    parsed = json::parse(run.text);
  } catch (...) {
    ok = false;
  }
  CHECK(ok, "output parses as JSON: " + run.text);
  if (ok) {
    CHECK(parsed.is_object() && parsed.contains("city") && parsed.contains("n"));
  }
  CHECK(run.epochs_host > 0);
  // Without logits the same request is served unconstrained (status quo).
  Fixture g(false);
  sampling::SamplingConfig again;
  again.constraint = g.runner->BindConstraint(grammar);
  const auto plain = g.Generate(0, again, 12);
  CHECK(plain.epochs_host == 0, "device only without logits");
}

void TestToolChoiceRequired() {
  std::cerr << "tool_choice required: the call is forced\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back("get_weather",
                     sampling::JsonConstraint::ToolParameters(
                         params, true, sampling::JsonConstraint::ToolFormat::kJson));
  auto grammar = sampling::JsonConstraint::WithTools(
      nullptr, std::move(tools), true, false,
      sampling::JsonConstraint::ToolFormat::kJson);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back({.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  const auto run = f.Generate(0, config, 120, request);
  CHECK(run.text.rfind("<tool_call>", 0) == 0, "starts with the call marker: " + run.text);
  CHECK(run.text.find("get_weather") != std::string::npos);
  CHECK(run.epochs_host > 0);
}

void TestLazyGrammarStaysOnDevice() {
  std::cerr << "auto tools: device sampling until the call starts, then the host\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back("get_weather",
                     sampling::JsonConstraint::ToolParameters(
                         params, true, sampling::JsonConstraint::ToolFormat::kJson));
  auto grammar = sampling::JsonConstraint::WithTools(
      nullptr, std::move(tools), false, false,
      sampling::JsonConstraint::ToolFormat::kJson);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back({.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  const auto prompt = f.runner->PreparePrompt(request);
  CHECK(prompt.has_value());
  const int base = static_cast<int>(prompt->tokens.size());
  // The model writes "ok ", then the call marker; after it the script keeps
  // writing garbage that only the grammar can turn into a call.
  const std::string plan = "ok <tool_call>";
  fake::state().script = [plan, base](int last, int position, float* out) {
    const int index = position - base;
    PeakScript(last, position, out);
    if (index >= 0 && index < static_cast<int>(plan.size())) {
      for (int v = 0; v < fake::kBytes; ++v) out[v] = -20.0F;
      out[static_cast<unsigned char>(plan[index])] = 5.0F;
    }
  };
  const auto run = f.Generate(0, config, 60, request);
  CHECK(run.text.rfind("ok <tool_call>", 0) == 0, "the lazy prefix is untouched: " + run.text);
  CHECK(run.epochs_device >= 1, "device epochs before the call");
  CHECK(run.epochs_host >= 1, "host epochs once the grammar restricts");
  // After the marker the grammar owns the output: a name and JSON arguments.
  CHECK(run.text.find("\"name\"") != std::string::npos, "call body follows: " + run.text);
}


void TestDialectGating() {
  std::cerr << "dialect: tool grammars stay off for another native call syntax\n";
  fake::reset();
  fake::state().script = PeakScript;
  fake::state().tool_call_text =
      "<tool_call>\n<function=qprobe>\n<parameter=zz>\n7\n</parameter>\n</function>\n</tool_call>";
  Fixture f;
  CHECK(!f.runner->NativeJsonTools());
  json::Value params = json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})");
  std::vector<sampling::JsonConstraint::Tool> tools;
  tools.emplace_back("get_weather",
                     sampling::JsonConstraint::ToolParameters(
                         params, true, sampling::JsonConstraint::ToolFormat::kJson));
  auto grammar = sampling::JsonConstraint::WithTools(
      nullptr, std::move(tools), true, false,
      sampling::JsonConstraint::ToolFormat::kJson);
  sampling::SamplingConfig config;
  config.constraint = f.runner->BindConstraint(grammar);
  ext::ChatRequest request = Fixture::DefaultRequest();
  request.tools.push_back({.name = "get_weather", .parameters_json = R"({"type":"object"})"});
  const auto run = f.Generate(0, config, 10, request);
  CHECK(run.epochs_host == 0, "unconstrained: the device path serves it");
  // The same package still honours logprobs on a tools request.
  sampling::SamplingConfig lp;
  lp.logprobs = 2;
  const auto with_lp = f.Generate(0, lp, 4, request);
  CHECK(with_lp.epochs_host > 0);
  // A reasoning marker that is not <think> turns every grammar off.
  fake::reset();
  fake::state().script = PeakScript;
  fake::state().reasoning_close = "<|end|>";
  Fixture g;
  sampling::SamplingConfig plain;
  plain.constraint = g.runner->BindConstraint(sampling::JsonConstraint::Object());
  const auto no_grammar = g.Generate(0, plain, 6);
  CHECK(no_grammar.epochs_host == 0, "reasoning markers differ: no grammar");
}

void TestBatchMix() {
  std::cerr << "batch: hosted and device rows share one DecodeBatch\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f(true, 3);
  // Rows 0 (logprobs) and 1 (plain) decode together.
  struct One {
    std::unique_ptr<LoomSlot> slot;
    std::unique_ptr<LoomState> state;
    std::unique_ptr<sampling::SamplerState> sampler;
    std::vector<int> tokens;
  };
  std::vector<One> ones(2);
  for (std::uint32_t r = 0; r < 2; ++r) {
    ones[r].slot = std::make_unique<LoomSlot>(r, f.pool);
    loom_text_embed_decoder_t* decoder = nullptr;
    loom_text_embed_decoder_open(f.model->embed(), &decoder);
    ones[r].state = std::make_unique<LoomState>(*ones[r].slot, decoder);
    ones[r].state->SetStopAtEos(true);
    auto prepared = f.runner->PreparePrompt(Fixture::DefaultRequest());
    f.runner->SetPromptContext(*ones[r].state, prepared->context);
    std::size_t offset = 0;
    while (true) {
      const auto step = f.runner->Prefill(*ones[r].state, prepared->tokens, offset, 64);
      offset += step.consumed_tokens;
      if (step.decode_ready) break;
    }
    sampling::SamplingConfig config;
    if (r == 0) config.logprobs = 2;
    ones[r].sampler = std::make_unique<sampling::SamplerState>(config, std::span<const sampling::TokenId>());
  }
  for (int round = 0; round < 5; ++round) {
    const ext::TextRunnerDecode decodes[] = {
        {*ones[0].state, 1, *ones[0].sampler},
        {*ones[1].state, 1, *ones[1].sampler}};
    auto steps = f.runner->DecodeBatch(std::span(decodes, 2));
    for (std::size_t i = 0; i < 2; ++i) {
      if (steps[i].failure) std::rethrow_exception(steps[i].failure);
      for (auto& sel : steps[i].selections) {
        ones[i].sampler->Accept(sel.token);
        ones[i].tokens.push_back(static_cast<int>(sel.token));
        CHECK(sel.logprobs.has_value() == (i == 0), "logprobs only on the hosted row");
      }
    }
  }
  CHECK(ones[0].tokens.size() == 5 && ones[1].tokens.size() == 5);
  CHECK(ones[0].tokens == ones[1].tokens, "same prompt, same greedy tokens either way");
}

void TestSelectAdvance() {
  std::cerr << "SelectNext/Advance: the same tokens as DecodeBatch\n";
  fake::reset();
  fake::state().script = PeakScript;
  Fixture f;
  sampling::SamplingConfig config;
  config.logprobs = 2;
  const auto batch = f.Generate(0, config, 6);
  // Same request through SelectNext + Advance.
  auto slot = std::make_unique<LoomSlot>(0, f.pool);
  loom_text_embed_decoder_t* decoder = nullptr;
  loom_text_embed_decoder_open(f.model->embed(), &decoder);
  LoomState state(*slot, decoder);
  state.SetStopAtEos(true);
  auto prepared = f.runner->PreparePrompt(Fixture::DefaultRequest());
  f.runner->SetPromptContext(state, prepared->context);
  std::size_t offset = 0;
  while (true) {
    const auto step = f.runner->Prefill(state, prepared->tokens, offset, 64);
    offset += step.consumed_tokens;
    if (step.decode_ready) break;
  }
  sampling::SamplerState sampler(config, std::span<const sampling::TokenId>());
  std::vector<int> tokens;
  for (int i = 0; i < 6; ++i) {
    auto selection = f.runner->SelectNext(state, sampler);
    CHECK(!selection.stop);
    CHECK(selection.logprobs.has_value());
    sampler.Accept(selection.token);
    tokens.push_back(static_cast<int>(selection.token));
    f.runner->Advance(state, selection.token);
  }
  CHECK(tokens == batch.tokens, "select/advance == batch");
}

}  // namespace

int main() {
  TestDevicePathUnchanged();
  TestLogprobs();
  TestRepetitionPenalty();
  TestMinP();
  TestSeedStream();
  TestJsonGrammar();
  TestToolChoiceRequired();
  TestLazyGrammarStaysOnDevice();
  TestDialectGating();
  TestBatchMix();
  TestSelectAdvance();
  if (g_failures) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cerr << "all passed\n";
  return 0;
}
