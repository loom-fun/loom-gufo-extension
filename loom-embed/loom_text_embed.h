// Loom text models embedded in another serving process.
//
// Loads a Loom model package (model.json, its programs and parameters) the
// way loom-serve registers it, and exposes the package's text model layer one
// step at a time: chat rendering and tokenization from the package's own
// programs, and resident rows advanced by prefill, decode and packed epochs on
// the package's kernels. The caller owns scheduling, admission and caching;
// no Loom service, session cache or HTTP is involved.
//
// Calls on one handle must be serialized by the caller. Functions return 0 on
// success; on failure they return nonzero and loom_text_embed_error() names
// the cause.

#ifndef LOOM_TEXT_EMBED_H_
#define LOOM_TEXT_EMBED_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TEXT_EMBED_ABI_VERSION 1

typedef struct loom_text_embed_t loom_text_embed_t;
typedef struct loom_text_embed_request_t loom_text_embed_request_t;

typedef struct loom_text_embed_options_t {
  // model.json of a fetched Loom package, or its directory.
  const char* model_path;
  // Named preset; NULL or "" selects the card's unambiguous preset.
  const char* preset;
  // Loom device URI; NULL selects amdgpu with an active completion wait.
  const char* device_uri;
  // Shared parameter/state budget in bytes; zero imposes no limit.
  uint64_t memory_bytes;
  // Resident rows (concurrent sessions); zero keeps the preset's count.
  uint32_t rows;
  // Retained row checkpoints (shared device state); zero selects 64.
  uint32_t checkpoints;
} loom_text_embed_options_t;

typedef struct loom_text_embed_info_t {
  // The card's model identity, NUL-terminated.
  char model_id[256];
  uint32_t context_capacity;
  // Largest token count of one prefill call's packed shape.
  uint32_t prefill_capacity;
  uint32_t rows;
  // Proposed tokens per drafted step; zero when the package has no drafter.
  uint32_t draft_depth;
  // Widest packed decode epoch (rows advanced together).
  uint32_t max_epoch_spans;
  // Most images one request may carry; zero when the package has no vision.
  uint32_t max_images;
} loom_text_embed_info_t;

int loom_text_embed_open(const loom_text_embed_options_t* options,
                         loom_text_embed_t** out_embed, char* error,
                         size_t error_size);
void loom_text_embed_close(loom_text_embed_t* embed);
const char* loom_text_embed_error(const loom_text_embed_t* embed);
void loom_text_embed_info(const loom_text_embed_t* embed,
                          loom_text_embed_info_t* out_info);

// Renders an OpenAI chat request body with the package's chat program and
// returns its prompt tokens (tokens may be NULL to count only), plus the
// request's source-validated options (sampling, thinking) for row binding.
int loom_text_embed_chat_prepare(loom_text_embed_t* embed, const char* body,
                                 size_t body_size, int32_t* tokens,
                                 size_t capacity, size_t* out_count,
                                 loom_text_embed_request_t** out_request);
void loom_text_embed_request_free(loom_text_embed_request_t* request);

// Images of a prepared request (image_url parts in the body). media returns
// the input ranges [offset, offset + count) that each image's projected
// features replace, in order, and their number. project runs the package's
// image component on them; prefill_request then consumes request input
// [offset, offset + count) on a row, features included, selecting the next
// token when select is nonzero.
size_t loom_text_embed_request_media(const loom_text_embed_request_t* request,
                                     uint32_t* offsets, uint32_t* counts,
                                     size_t capacity);
int loom_text_embed_request_project(loom_text_embed_request_t* request);
int loom_text_embed_row_prefill_request(loom_text_embed_t* embed, uint32_t row,
                                        loom_text_embed_request_t* request,
                                        size_t offset, size_t count,
                                        int select);

// Output of one prepared chat request, read with the package's own chat
// program, as loom-serve reads it. feed appends generated text and returns
// what it settles, normalized: reasoning inside <think>...</think>, answer
// text as is, framing and tool syntax withheld. final releases the remainder
// and closes reasoning. tools then parses the withheld tool calls into a JSON
// array of {"id","type":"function","function":{"name","arguments"}}
// (truncated: the output hit its token limit; an unfinished call is dropped).
int loom_text_embed_output_feed(loom_text_embed_request_t* request,
                                const char* text, size_t text_size, int final,
                                char* out, size_t capacity, size_t* out_size);
int loom_text_embed_output_tools(loom_text_embed_request_t* request,
                                 int truncated, char* json, size_t capacity,
                                 size_t* out_size);

// Tokenizes raw text; special-token spellings become special tokens when
// match_special is nonzero. Tokens may be NULL to count only.
int loom_text_embed_tokenize(loom_text_embed_t* embed, const char* text,
                             size_t text_size, int match_special,
                             int32_t* tokens, size_t capacity,
                             size_t* out_count);
// Decodes tokens to text, special tokens included. text may be NULL to size.
int loom_text_embed_detokenize(loom_text_embed_t* embed, const int32_t* tokens,
                               size_t count, char* text, size_t capacity,
                               size_t* out_size);
int32_t loom_text_embed_vocabulary_size(loom_text_embed_t* embed);

// Streaming detokenization: each feed returns exactly the text its tokens
// settle (a character split across tokens appears once it completes), so the
// concatenated output equals decoding all tokens at once. Special tokens are
// kept.
typedef struct loom_text_embed_decoder_t loom_text_embed_decoder_t;
int loom_text_embed_decoder_open(loom_text_embed_t* embed,
                                 loom_text_embed_decoder_t** out_decoder);
int loom_text_embed_decoder_feed(loom_text_embed_decoder_t* decoder,
                                 const int32_t* tokens, size_t count,
                                 char* text, size_t capacity,
                                 size_t* out_size);
int loom_text_embed_decoder_reset(loom_text_embed_decoder_t* decoder);
void loom_text_embed_decoder_free(loom_text_embed_decoder_t* decoder);

// Starts a request on a row: reserves state through `extent` consumed tokens
// (fresh replaces the row's history, otherwise it continues it) and binds the
// request's options. Returns 1 (and no error) when capacity is unavailable.
int loom_text_embed_row_begin(loom_text_embed_t* embed, uint32_t row,
                              const loom_text_embed_request_t* request,
                              uint64_t extent, int fresh);
// Grows the row's reservation, keeping its history. 1 means no capacity.
int loom_text_embed_row_extend(loom_text_embed_t* embed, uint32_t row,
                               uint64_t extent);
uint64_t loom_text_embed_row_extent(loom_text_embed_t* embed, uint32_t row);
// Rebinds request options (sampling) to a row between steps.
int loom_text_embed_row_bind(loom_text_embed_t* embed, uint32_t row,
                             const loom_text_embed_request_t* request);
void loom_text_embed_row_end(loom_text_embed_t* embed, uint32_t row);
int loom_text_embed_row_reset(loom_text_embed_t* embed, uint32_t row);

// Consumes known tokens (in prefill-capacity chunks) and selects the next.
int loom_text_embed_row_prefill(loom_text_embed_t* embed, uint32_t row,
                                const int32_t* tokens, size_t count);
// Consumes the pending selection and selects its successor.
int loom_text_embed_row_decode(loom_text_embed_t* embed, uint32_t row);
int32_t loom_text_embed_row_token(loom_text_embed_t* embed, uint32_t row);
int loom_text_embed_row_is_eos(loom_text_embed_t* embed, uint32_t row);
uint64_t loom_text_embed_row_position(loom_text_embed_t* embed, uint32_t row);

// One packed epoch: every row consumes its pending selection and selects its
// successor. Rows must be distinct and have pending, non-EOS selections.
int loom_text_embed_decode_batch(loom_text_embed_t* embed, const uint32_t* rows,
                                 size_t count);

// One drafted epoch (draft_depth > 0): each row proposes draft_depth tokens on
// device after its pending selection and verifies them greedily, accepting at
// most limits[i] new selections. tokens receives 16 IDs per row, of which
// counts[i] are valid: the accepted continuation, ending in the row's new
// pending selection (consumed[i] tokens entered state).
int loom_text_embed_verify_batch(loom_text_embed_t* embed, const uint32_t* rows,
                                 size_t count, const uint32_t* limits,
                                 int32_t* tokens, uint32_t* counts,
                                 uint32_t* consumed);

// Row checkpoints: the exact consumed state of a row, sharing its device
// pages (copy-on-write) until released. Restoring replaces a row's state.
// pin returns 1 when checkpoint capacity is exhausted; restore sets
// *restored to 0 when pool capacity is short.
typedef struct loom_text_embed_checkpoint_t loom_text_embed_checkpoint_t;
int loom_text_embed_row_pin(loom_text_embed_t* embed, uint32_t row,
                            loom_text_embed_checkpoint_t** out_checkpoint);
int loom_text_embed_row_restore(loom_text_embed_t* embed, uint32_t row,
                                const loom_text_embed_checkpoint_t* checkpoint,
                                int* restored);
void loom_text_embed_checkpoint_release(loom_text_embed_t* embed,
                                        loom_text_embed_checkpoint_t* checkpoint);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TEXT_EMBED_H_
