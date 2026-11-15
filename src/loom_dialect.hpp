// The model's own spelling of tool calls and reasoning, for Gufo's grammars.
//
// A Loom package's chat program reads the model's native output (its tool-call
// syntax, its reasoning markers) and the runner publishes it as the canonical
// markup Gufo parses. Grammars constrain the model's tokens, so on the host-sampling path they
// must be written in the native spelling. The package says which one it
// speaks in llm_config.json (the generator's input, shipped in the package):
// chat.call_format, chat.think_open / think_end. This file maps those to
// Gufo's JsonConstraint::ToolSyntax and to the reasoning markers a grammar
// uses; a format Gufo's descriptors cannot express (pythonic calls, ATEM)
// has no syntax, and its tool requests stay unconstrained.
#ifndef GUFO_LOOM_DIALECT_HPP_
#define GUFO_LOOM_DIALECT_HPP_

#include <optional>
#include <string>

#include <gufo/ext/json.hpp>
#include <gufo/ext/text.hpp>

namespace gufo::loom {

struct NativeDialect {
  // llm_config.json named the dialect (otherwise the runner probes the chat
  // program for the one JSON dialect it can recognize).
  bool known{false};
  std::string call_format;
  // The native call syntax; nullopt when Gufo's grammars cannot spell it.
  std::optional<sampling::JsonConstraint::ToolSyntax> tools;
  // The raw markers around reasoning; an empty open means the package does
  // not reason (no reasoning phase in grammars).
  std::string reasoning_open;
  std::string reasoning_close;
  // Framing the model writes between the end of reasoning and the next
  // channel: OpenAI harmony starts a new assistant message, and a JSON answer
  // lives in its `final` channel.
  std::string after_reasoning;
  std::string answer_channel;
};

// `chat` is the "chat" object of the package's llm_config.json.
inline NativeDialect NativeDialectOf(const json::Value& chat) {
  using Syntax = sampling::JsonConstraint::ToolSyntax;
  NativeDialect dialect;
  dialect.known = true;
  dialect.call_format = chat.member_str("call_format");
  const auto& format = dialect.call_format;
  if (format == "qwen_xml") {
    // <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
    dialect.tools = sampling::JsonConstraint::Syntax(
        sampling::JsonConstraint::ToolFormat::kQwen);
  } else if (format == "hermes") {
    // <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>
    dialect.tools = Syntax{
        .open = "<tool_call>",
        .name_prefix = "\n{\"name\": \"",
        .name_suffix = "\", \"arguments\": ",
        .close = "}\n</tool_call>",
        .name_delimiters = "\"",
    };
  } else if (format == "glm") {
    // <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
    // as the package's template (and so its history) spells it: no line breaks.
    dialect.tools = Syntax{
        .open = "<tool_call>",
        .close = "</tool_call>",
        .name_delimiters = "<",
        .tagged =
            sampling::JsonConstraint::TaggedArguments{
                .parameter_open = "<arg_key>",
                .raw_value_open = "</arg_key><arg_value>",
                .typed_value_open = "</arg_key><arg_value>",
                .value_close = "</arg_value>",
            },
    };
  } else if (format == "gemma4") {
    // <|tool_call>call:NAME{key:<|"|>v<|"|>,n:3}<tool_call|> (keys sorted, bare)
    dialect.tools = Syntax{
        .open = "<|tool_call>",
        .name_prefix = "call:",
        .close = "<tool_call|>",
        .name_delimiters = "{",
        .values = {.string_open = "<|\"|>",
                   .string_close = "<|\"|>",
                   .bare_keys = true,
                   .whitespace = false,
                   .sorted_keys = true},
    };
  } else if (format == "harmony") {
    // After the analysis channel (and its <|end|>), a new assistant message:
    //   <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>
    // A model that skips analysis starts at <|channel|>, as the prompt ends in
    // <|start|>assistant, so <|start|>assistant belongs to the reasoning close.
    dialect.tools = Syntax{
        .open = "<|channel|>commentary",
        .name_prefix = " to=functions.",
        .name_suffix = " <|constrain|>json<|message|>",
        .close = "<|call|>",
        .name_delimiters = " ",
    };
    dialect.after_reasoning = "<|start|>assistant";
    dialect.answer_channel = "<|channel|>final<|message|>";
  } else if (format == "json") {
    // Llama 3.x: {"name": "NAME", "parameters": {...}} (chat.call_arguments_key)
    dialect.tools = Syntax{
        .open = "{\"name\": \"",
        .name_suffix = "\", \"" + chat.member_str("call_arguments_key", "arguments") + "\": ",
        .close = "}",
        .name_delimiters = "\"",
    };
  }
  // pythonic (LFM2: [name(k='v')]), atem (Glimmer) and none (no tool calls):
  // no descriptor.
  const std::string open = chat.member_str("think_open");
  if (chat.member_str("thinking", "never") != "never" && !open.empty()) {
    dialect.reasoning_open = open;
    dialect.reasoning_close = chat.member_str("think_end", "</think>");
  }
  return dialect;
}

}  // namespace gufo::loom

#endif  // GUFO_LOOM_DIALECT_HPP_
