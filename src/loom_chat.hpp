// OpenAI request bodies for a package's chat program, and the canonical
// tool-call markup Gufo's parsers read.
#ifndef GUFO_LOOM_CHAT_HPP_
#define GUFO_LOOM_CHAT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <gufo/ext/json.hpp>
#include <gufo/ext/text.hpp>

namespace gufo::loom {

const char* RoleName(ext::ChatRole role);
json::Value ParseOr(const std::string& text, json::Value fallback);
std::string Base64(std::span<const std::uint8_t> bytes);
std::string_view MediaType(std::span<const std::uint8_t> bytes);
json::Value ContentParts(const ext::ChatMessage& message);
std::string ChatBody(const std::string& model, const ext::ChatRequest& request,
                     bool with_effort = true);
std::string SamplingBody(const std::string& model,
                         const sampling::SamplingConfig& config);
std::string ToolMarkup(std::string_view calls_json);

}  // namespace gufo::loom

#endif  // GUFO_LOOM_CHAT_HPP_
