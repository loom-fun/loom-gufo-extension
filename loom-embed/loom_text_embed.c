// Loom text models embedded in another serving process. See loom_text_embed.h.

#include "loom_text_embed.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "iree/base/api.h"
#include "iree/base/internal/path.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/device_util.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/device_configuration.h"
#include "loom/serve/service/finite/component.h"
#include "loom/serve/service/kind.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/media.h"
#include "loom/serve/service/text/media_input.h"
#include "loom/serve/service/text/media_projector.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/runtime/storage/snapshot.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/service/text/schedule.h"
#include "loom/serve/service/text/tool_calls.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/configured.h"

#define LOOM_TEXT_EMBED_EXPORT __attribute__((visibility("default")))

struct loom_text_embed_t {
  iree_allocator_t allocator;
  loom_serve_host_t* host;
  loom_serve_device_configuration_t devices;
  loom_serve_device_group_t* group;
  loom_serve_model_configuration_t configuration;
  loom_serve_model_configuration_t resolved;
  bool has_resolved;
  // Reserved consumed-position high-water per row; zero when unreserved.
  uint64_t* extents;
  loom_serve_text_model_t* model;
  // The package's image component and its projector, when it has vision.
  loom_serve_finite_component_t* image;
  loom_serve_text_media_projector_t* projector;
  iree_host_size_t shape_count;
  loom_serve_packing_shape_t* shapes;
  iree_host_size_t prefill_capacity;
  uint32_t rows;
  uint32_t draft_depth;
  uint64_t request_serial;
  char model_id[256];
  char error[2048];
};

struct loom_text_embed_request_t {
  loom_text_embed_t* embed;
  char* body;
  uint64_t serial;
  loom_serve_text_chat_t chat;
  // Checked input with image spans, and its projected features.
  loom_serve_text_input_t input;
  bool has_input;
  loom_serve_text_media_request_t* media;
  // Generated text so far, and whether a reasoning span is open on output.
  iree_string_builder_t response;
  bool reasoning_open;
  bool finished;
  // Response offsets: where the last reasoning span ended, and the framing
  // between reasoning and the first answer text (the model's own close).
  size_t reasoning_end;
  size_t close_begin;
  size_t close_end;
  bool reasoned;
  bool closed;
};

static const loom_serve_model_configuration_t* loom_text_embed_config(
    const loom_text_embed_t* embed) {
  return embed->has_resolved ? &embed->resolved : &embed->configuration;
}

// Records a failed status as the handle's error and returns nonzero.
static int loom_text_embed_fail(char* error, size_t error_size,
                                iree_status_t status) {
  if (iree_status_is_ok(status)) return 0;
  iree_allocator_t allocator = iree_allocator_system();
  char* text = NULL;
  iree_host_size_t length = 0;
  if (error && error_size) {
    if (iree_status_to_string(status, &allocator, &text, &length)) {
      snprintf(error, error_size, "%.*s", (int)length, text);
      iree_allocator_free(allocator, text);
    } else {
      snprintf(error, error_size, "%s",
               iree_status_code_string(iree_status_code(status)));
    }
  }
  iree_status_free(status);
  return 2;
}

#define EMBED_RETURN(embed, expr) \
  return loom_text_embed_fail((embed)->error, sizeof((embed)->error), (expr))

static iree_status_t loom_text_embed_open_devices(
    loom_text_embed_t* embed, const loom_text_embed_options_t* options) {
  static const loomc_iree_hal_target_provider_t* providers[1];
  providers[0] = loomc_amdgpu_iree_hal_target_provider();
  static loom_serve_compiler_options_t compiler_options;
  compiler_options = (loom_serve_compiler_options_t){
      .target_environment_create = loomc_target_environment_create_configured,
      .target_providers = providers,
      .target_provider_count = 1,
  };
  IREE_RETURN_IF_ERROR(loom_serve_host_create(8, &compiler_options,
                                              embed->allocator, &embed->host));
  const char* uri = options->device_uri && options->device_uri[0]
                        ? options->device_uri
                        : "amdgpu://?completion_spin_ns=4294967295";
  IREE_RETURN_IF_ERROR(loom_serve_device_configuration_initialize_uri(
      iree_make_cstring_view(uri), embed->allocator, &embed->devices));
  if (embed->devices.count != 1 || embed->devices.pool_count < 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected one device and pool");
  }
  embed->devices.pools[0].options.memory_limit = options->memory_bytes;
  loom_serve_device_options_t* device = &embed->devices.options[0];
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_acquire_driver(
      iree_hal_available_driver_registry(), iree_uri_schema(device->uri),
      &driver));
  device->driver = driver;
  iree_status_t status = loom_serve_device_group_allocate(
      embed->host, 1, embed->devices.options, embed->allocator, &embed->group);
  for (iree_host_size_t i = 0;
       i < embed->devices.pool_count && iree_status_is_ok(status); ++i) {
    loom_serve_pool_t* pool = NULL;
    status = loom_serve_device_group_create_pool(
        embed->group, embed->devices.pools[i].binding,
        &embed->devices.pools[i].options, &pool);
  }
  iree_hal_driver_release(driver);
  device->driver = NULL;
  return status;
}

// Owner wakes are not needed: image work is driven by polling.
static void loom_text_embed_wake(void* user_data) { (void)user_data; }

static iree_status_t loom_text_embed_open_model(
    loom_text_embed_t* embed, const loom_text_embed_options_t* options) {
  char path[4096];
  const size_t length = strlen(options->model_path);
  const bool json = length >= 5 && !strcmp(options->model_path + length - 5,
                                            ".json");
  snprintf(path, sizeof(path), json ? "%s" : "%s/model.json",
           options->model_path);
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_load(
      iree_make_cstring_view(path),
      iree_make_cstring_view(options->preset ? options->preset : ""),
      embed->allocator, &embed->configuration));
  if (embed->configuration.sources.reference.path.size) {
    IREE_RETURN_IF_ERROR(loom_serve_model_configuration_resolve(
        &embed->configuration, embed->allocator, &embed->resolved));
    embed->has_resolved = true;
  }
  const loom_serve_model_configuration_t* configuration =
      loom_text_embed_config(embed);
  if (configuration->kind != LOOM_SERVE_MODEL_KIND_TEXT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "the package's selected service is not text");
  }
  const loom_serve_text_configuration_t* text = &configuration->text;
  snprintf(embed->model_id, sizeof(embed->model_id), "%.*s",
           (int)configuration->name.size, configuration->name.data);
  embed->prefill_capacity = text->prefill_capacity;
  embed->draft_depth = (uint32_t)text->draft_depth;
  embed->rows = options->rows ? options->rows : (uint32_t)text->rows;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      embed->allocator, sizeof(uint64_t) * embed->rows,
      (void**)&embed->extents));
  memset(embed->extents, 0, sizeof(uint64_t) * embed->rows);
  embed->shape_count =
      loom_serve_text_default_shapes(text->prefill_capacity, NULL);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      embed->allocator, embed->shape_count, sizeof(*embed->shapes),
      (void**)&embed->shapes));
  loom_serve_text_default_shapes(text->prefill_capacity, embed->shapes);
  const loom_serve_text_options_t model_options = {
      .mode = LOOM_SERVE_TEXT_MODE_GENERATE,
      .sources = configuration->sources,
      .prefill_capacity = text->prefill_capacity,
      .context_capacity = text->context_capacity,
      .pool_capacity = text->pool_capacity,
      .epoch_count = embed->shape_count,
      .epoch_shapes = embed->shapes,
      .draft_depth = text->draft_depth,
      .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                           .structure_size = sizeof(loomc_sanitizer_options_t)},
      .weights_path = text->weights,
      .tokenizer_path = text->tokenizer,
      .row_count = embed->rows,
      .checkpoint_capacity = options->checkpoints ? options->checkpoints : 64,
      .options = configuration->options,
  };
  IREE_RETURN_IF_ERROR(loom_serve_text_model_allocate(
      loom_serve_device_group_at(embed->group, 0),
      loom_serve_device_group_pool_at(embed->group, 0), &model_options,
      embed->allocator, &embed->model));
  if (text->image.contract || text->image.prepare.size) {
    // The package's image component, as loom-serve registers it; prepared on
    // the first image request.
    loom_serve_finite_model_options_t image_options = {
        .sources = configuration->sources,
        .asset_directory = text->image.assets,
        .options = text->image.options,
        .specialization_cache = text->image.specialization_cache,
    };
    image_options.sources.component = text->image.contract;
    image_options.sources.prepare = text->image.prepare;
    image_options.sources.control = text->image.control;
    IREE_RETURN_IF_ERROR(loom_serve_finite_component_allocate(
        loom_serve_device_group_at(embed->group, 0),
        loom_serve_device_group_pool_at(embed->group, 0), &image_options,
        (loom_serve_wake_callback_t){loom_text_embed_wake, NULL},
        embed->allocator, &embed->image));
  }
  return loom_serve_text_model_activate(embed->model);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_open(
    const loom_text_embed_options_t* options, loom_text_embed_t** out_embed,
    char* error, size_t error_size) {
  *out_embed = NULL;
  iree_allocator_t allocator = iree_allocator_system();
  loom_text_embed_t* embed = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*embed), (void**)&embed);
  if (!iree_status_is_ok(status)) {
    return loom_text_embed_fail(error, error_size, status);
  }
  memset(embed, 0, sizeof(*embed));
  embed->allocator = allocator;
  status = loom_text_embed_open_devices(embed, options);
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_open_model(embed, options);
  }
  if (!iree_status_is_ok(status)) {
    loom_text_embed_close(embed);
    return loom_text_embed_fail(error, error_size, status);
  }
  *out_embed = embed;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_close(loom_text_embed_t* embed) {
  if (!embed) return;
  if (embed->projector) {
    loom_serve_text_media_projector_free(embed->projector);
  }
  if (embed->image) {
    bool retired = false;
    for (int i = 0; i < 100000 && !retired; ++i) {
      iree_status_ignore(
          loom_serve_finite_component_retire(embed->image, &retired));
      if (!retired) usleep(100);
    }
    loom_serve_finite_component_free(embed->image);
  }
  if (embed->model) {
    iree_status_ignore(loom_serve_text_model_free(embed->model));
  }
  if (embed->group) {
    iree_status_ignore(loom_serve_device_group_free(embed->group));
  }
  loom_serve_device_configuration_deinitialize(&embed->devices);
  iree_allocator_free(embed->allocator, embed->extents);
  iree_allocator_free(embed->allocator, embed->shapes);
  if (embed->has_resolved) {
    loom_serve_model_configuration_deinitialize(&embed->resolved);
  }
  loom_serve_model_configuration_deinitialize(&embed->configuration);
  if (embed->host) loom_serve_host_release(embed->host);
  iree_allocator_free(embed->allocator, embed);
}

LOOM_TEXT_EMBED_EXPORT const char* loom_text_embed_error(
    const loom_text_embed_t* embed) {
  return embed->error;
}

static iree_host_size_t loom_text_embed_max_spans(
    const loom_text_embed_t* embed) {
  iree_host_size_t spans = 0;
  for (iree_host_size_t i = 0; i < embed->shape_count; ++i) {
    spans = iree_max(spans, embed->shapes[i].span_capacity);
  }
  return iree_min(spans, (iree_host_size_t)embed->rows);
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_info(
    const loom_text_embed_t* embed, loom_text_embed_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  memcpy(out_info->model_id, embed->model_id, sizeof(out_info->model_id));
  out_info->context_capacity =
      (uint32_t)loom_serve_text_model_context_capacity(embed->model);
  out_info->prefill_capacity = (uint32_t)embed->prefill_capacity;
  out_info->rows = embed->rows;
  out_info->draft_depth =
      (uint32_t)loom_serve_text_model_draft_depth(embed->model);
  out_info->max_epoch_spans = (uint32_t)loom_text_embed_max_spans(embed);
  out_info->max_images =
      embed->image ? (uint32_t)loom_serve_text_model_chat_policy(embed->model)
                         ->images.maximum_count
                   : 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_chat_prepare(
    loom_text_embed_t* embed, const char* body, size_t body_size,
    int32_t* tokens, size_t capacity, size_t* out_count,
    loom_text_embed_request_t** out_request) {
  *out_count = 0;
  if (out_request) *out_request = NULL;
  loom_text_embed_request_t* request = NULL;
  iree_status_t status = iree_allocator_malloc(
      embed->allocator, sizeof(*request) + body_size + 1, (void**)&request);
  if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
  memset(request, 0, sizeof(*request));
  request->embed = embed;
  request->serial = embed->request_serial + 1;
  iree_string_builder_initialize(embed->allocator, &request->response);
  request->body = (char*)(request + 1);
  memcpy(request->body, body, body_size);
  request->body[body_size] = 0;
  const loom_serve_model_configuration_t* configuration =
      loom_text_embed_config(embed);
  status = loom_serve_text_chat_initialize(
      loom_serve_text_model_chat_policy(embed->model), configuration->name,
      iree_make_string_view(request->body, body_size), ++embed->request_serial,
      /*default_max_tokens=*/1, embed->allocator, &request->chat);
  if (iree_status_is_ok(status)) {
    loom_serve_text_media_span_t* media = NULL;
    iree_host_size_t media_count = 0;
    iree_host_size_t count = 0;
    status = loom_serve_text_chat_prepare_input(
        &request->chat, 0, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
        LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
        tokens ? capacity
               : loom_serve_text_model_context_capacity(embed->model),
        0, embed->allocator, tokens, &count, &media, &media_count);
    iree_allocator_free(embed->allocator, media);
    if (iree_status_is_ok(status) && media_count) {
      if (!embed->image) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "this package does not accept images");
      } else {
        // Image requests keep the checked input with its media spans.
        int32_t* scratch = NULL;
        const iree_host_size_t capacity =
            loom_serve_text_model_context_capacity(embed->model);
        status = iree_allocator_malloc(embed->allocator,
                                       capacity * sizeof(int32_t),
                                       (void**)&scratch);
        if (iree_status_is_ok(status)) {
          status = loom_serve_text_input_prepare(
              &request->chat, (loom_serve_text_input_prefix_t){0}, false,
              capacity, scratch, &request->input, embed->allocator);
        }
        iree_allocator_free(embed->allocator, scratch);
        if (iree_status_is_ok(status)) {
          request->has_input = true;
          count = request->input.count;
          if (tokens) {
            if (count > capacity) {
              status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                        "prompt exceeds the token buffer");
            } else {
              memcpy(tokens, request->input.tokens, count * sizeof(int32_t));
            }
          }
        }
      }
    }
    *out_count = count;
    if (!iree_status_is_ok(status)) {
      loom_serve_text_chat_deinitialize(&request->chat);
    }
  }
  if (!iree_status_is_ok(status) || !out_request) {
    if (iree_status_is_ok(status)) {
      loom_serve_text_chat_deinitialize(&request->chat);
    }
    iree_string_builder_deinitialize(&request->response);
    iree_allocator_free(embed->allocator, request);
    EMBED_RETURN(embed, status);
  }
  *out_request = request;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_request_free(
    loom_text_embed_request_t* request) {
  if (!request) return;
  if (request->media) {
    if (request->embed->projector) {
      while (!loom_serve_text_media_projector_cancel(request->embed->projector,
                                                     request->media)) {
        bool progress = false;
        iree_status_ignore(loom_serve_text_media_projector_advance(
            request->embed->projector, 0, &progress));
        usleep(100);
      }
    }
    iree_status_ignore(loom_serve_text_media_request_take_status(request->media));
    loom_serve_text_media_request_free(request->media);
  }
  if (request->has_input) {
    loom_serve_text_input_deinitialize(&request->input,
                                       request->embed->allocator);
  }
  iree_string_builder_deinitialize(&request->response);
  loom_serve_text_chat_deinitialize(&request->chat);
  iree_allocator_free(request->embed->allocator, request);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_tokenize(
    loom_text_embed_t* embed, const char* text, size_t text_size,
    int match_special, int32_t* tokens, size_t capacity, size_t* out_count) {
  const iree_tokenizer_t* tokenizer =
      loom_serve_text_model_tokenizer(embed->model);
  int32_t scratch[1];
  iree_tokenizer_token_output_t output = {
      .capacity = tokens ? capacity : 0,
      .token_ids = tokens ? tokens : scratch,
  };
  iree_host_size_t count = 0;
  iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
  if (!match_special) {
    flags |= IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING;
  }
  iree_status_t status = iree_tokenizer_encode(
      tokenizer, iree_make_string_view(text, text_size), flags, output,
      embed->allocator, &count);
  *out_count = count;
  if (!tokens && iree_status_is_resource_exhausted(status)) {
    iree_status_ignore(status);
    return 0;
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_detokenize(
    loom_text_embed_t* embed, const int32_t* tokens, size_t count, char* text,
    size_t capacity, size_t* out_size) {
  const iree_tokenizer_t* tokenizer =
      loom_serve_text_model_tokenizer(embed->model);
  char scratch[1];
  iree_host_size_t size = 0;
  iree_status_t status = iree_tokenizer_decode(
      tokenizer, (iree_tokenizer_token_id_list_t){count, tokens},
      IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_mutable_string_view(text ? text : scratch, text ? capacity : 0),
      embed->allocator, &size);
  *out_size = size;
  if (!text && iree_status_is_resource_exhausted(status)) {
    iree_status_ignore(status);
    return 0;
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int32_t
loom_text_embed_vocabulary_size(loom_text_embed_t* embed) {
  return (int32_t)iree_tokenizer_vocab_capacity(
      iree_tokenizer_vocab(loom_serve_text_model_tokenizer(embed->model)));
}

static loom_serve_text_row_t* loom_text_embed_row(loom_text_embed_t* embed,
                                                  uint32_t row) {
  return loom_serve_text_model_row(embed->model, row);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_begin(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_request_t* request, uint64_t extent, int fresh) {
  loom_serve_text_row_t* handle = loom_text_embed_row(embed, row);
  loom_serve_text_row_release_reservation(handle);
  embed->extents[row] = 0;
  bool admitted = false;
  iree_status_t status = loom_serve_text_row_try_reserve(
      handle,
      fresh ? LOOM_SERVE_TEXT_RESERVE_FRESH : LOOM_SERVE_TEXT_RESERVE_CONTINUE,
      NULL, (iree_host_size_t)extent, &admitted);
  if (iree_status_is_ok(status) && !admitted) return 1;
  if (iree_status_is_ok(status)) embed->extents[row] = extent;
  if (iree_status_is_ok(status) && request) {
    status = loom_serve_text_row_bind_request(handle, request->chat.options);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_extend(loom_text_embed_t* embed,
                                                      uint32_t row,
                                                      uint64_t extent) {
  return loom_text_embed_row_begin(embed, row, NULL, extent, 0);
}

LOOM_TEXT_EMBED_EXPORT uint64_t
loom_text_embed_row_extent(loom_text_embed_t* embed, uint32_t row) {
  return embed->extents[row];
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_bind(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_request_t* request) {
  EMBED_RETURN(embed, loom_serve_text_row_bind_request(
                          loom_text_embed_row(embed, row),
                          request->chat.options));
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_row_end(loom_text_embed_t* embed,
                                                    uint32_t row) {
  loom_serve_text_row_release_reservation(loom_text_embed_row(embed, row));
  embed->extents[row] = 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_reset(loom_text_embed_t* embed,
                                                     uint32_t row) {
  EMBED_RETURN(embed, loom_serve_text_row_reset(loom_text_embed_row(embed, row)));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_prefill(
    loom_text_embed_t* embed, uint32_t row, const int32_t* tokens,
    size_t count) {
  loom_serve_text_row_t* handle = loom_text_embed_row(embed, row);
  iree_status_t status = iree_ok_status();
  for (size_t offset = 0; offset < count && iree_status_is_ok(status);) {
    const size_t chunk = iree_min(count - offset, embed->prefill_capacity);
    status = loom_serve_text_row_prefill(handle, chunk, tokens + offset);
    offset += chunk;
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_decode(loom_text_embed_t* embed,
                                                      uint32_t row) {
  EMBED_RETURN(embed,
               loom_serve_text_row_decode(loom_text_embed_row(embed, row)));
}

LOOM_TEXT_EMBED_EXPORT int32_t loom_text_embed_row_token(
    loom_text_embed_t* embed, uint32_t row) {
  return loom_serve_text_row_token(loom_text_embed_row(embed, row));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_is_eos(loom_text_embed_t* embed,
                                                      uint32_t row) {
  return loom_serve_text_row_is_eos(loom_text_embed_row(embed, row));
}

LOOM_TEXT_EMBED_EXPORT uint64_t
loom_text_embed_row_position(loom_text_embed_t* embed, uint32_t row) {
  return loom_serve_text_row_position(loom_text_embed_row(embed, row));
}

// Smallest prepared shape admitting the spans and tokens.
static bool loom_text_embed_shape(const loom_text_embed_t* embed,
                                  iree_host_size_t spans,
                                  iree_host_size_t tokens,
                                  iree_host_size_t* out_index) {
  bool found = false;
  for (iree_host_size_t i = 0; i < embed->shape_count; ++i) {
    const loom_serve_packing_shape_t* shape = &embed->shapes[i];
    if (shape->span_capacity < spans || shape->token_capacity < tokens) {
      continue;
    }
    if (!found ||
        shape->token_capacity < embed->shapes[*out_index].token_capacity ||
        (shape->token_capacity == embed->shapes[*out_index].token_capacity &&
         shape->span_capacity < embed->shapes[*out_index].span_capacity)) {
      *out_index = i;
      found = true;
    }
  }
  return found;
}

#define LOOM_TEXT_EMBED_MAX_BATCH 256

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decode_batch(
    loom_text_embed_t* embed, const uint32_t* rows, size_t count) {
  if (count == 1) return loom_text_embed_row_decode(embed, rows[0]);
  if (count > LOOM_TEXT_EMBED_MAX_BATCH) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "decode batch too wide"));
  }
  iree_host_size_t shape = 0;
  if (!loom_text_embed_shape(embed, count, count, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu decode spans",
                                         count));
  }
  int32_t pending[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_span_t spans[LOOM_TEXT_EMBED_MAX_BATCH];
  for (size_t i = 0; i < count; ++i) {
    pending[i] = loom_text_embed_row_token(embed, rows[i]);
    spans[i] = (loom_serve_text_span_t){
        .row_index = rows[i],
        .token_count = 1,
        .token_ids = &pending[i],
        .flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
  }
  EMBED_RETURN(embed,
               loom_serve_text_model_epoch(embed->model, shape, count, spans));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_verify_batch(
    loom_text_embed_t* embed, const uint32_t* rows, size_t count,
    const uint32_t* limits, int32_t* tokens, uint32_t* counts,
    uint32_t* consumed) {
  const iree_host_size_t width = embed->draft_depth + 1;
  if (!embed->draft_depth || count > LOOM_TEXT_EMBED_MAX_BATCH) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                         "drafting is unavailable"));
  }
  iree_host_size_t shape = 0;
  iree_host_size_t total = 0;
  for (size_t i = 0; i < count; ++i) total += limits[i] ? width : 1;
  if (!loom_text_embed_shape(embed, count, total, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu drafted spans",
                                         count));
  }
  int32_t pending[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_span_t spans[LOOM_TEXT_EMBED_MAX_BATCH];
  loom_serve_text_result_t results[LOOM_TEXT_EMBED_MAX_BATCH];
  for (size_t i = 0; i < count; ++i) {
    pending[i] = loom_text_embed_row_token(embed, rows[i]);
    spans[i] = (loom_serve_text_span_t){
        .row_index = rows[i],
        .token_count = limits[i] ? width : 1,
        .token_ids = &pending[i],
        .flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                 (limits[i] ? LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE : 0),
    };
  }
  memset(results, 0, sizeof(results[0]) * count);
  iree_status_t status = loom_serve_text_model_verify(
      embed->model, shape, count, spans, limits, NULL, results);
  if (iree_status_is_ok(status)) {
    for (size_t i = 0; i < count; ++i) {
      counts[i] = results[i].output_count;
      consumed[i] = results[i].consumed_count;
      memcpy(tokens + 16 * i, results[i].tokens, sizeof(results[i].tokens));
    }
  }
  EMBED_RETURN(embed, status);
}

struct loom_text_embed_decoder_t {
  loom_text_embed_t* embed;
  iree_tokenizer_decode_state_t* state;
  iree_host_size_t storage_size;
  uint8_t storage[];
};

static iree_status_t loom_text_embed_decoder_initialize(
    loom_text_embed_decoder_t* decoder) {
  return iree_tokenizer_decode_state_initialize(
      loom_serve_text_model_tokenizer(decoder->embed->model),
      IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_byte_span(decoder->storage, decoder->storage_size),
      &decoder->state);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decoder_open(
    loom_text_embed_t* embed, loom_text_embed_decoder_t** out_decoder) {
  *out_decoder = NULL;
  iree_host_size_t size = 0;
  iree_status_t status = iree_tokenizer_decode_state_calculate_size(
      loom_serve_text_model_tokenizer(embed->model), &size);
  loom_text_embed_decoder_t* decoder = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(embed->allocator, sizeof(*decoder) + size,
                                   (void**)&decoder);
  }
  if (iree_status_is_ok(status)) {
    decoder->embed = embed;
    decoder->state = NULL;
    decoder->storage_size = size;
    status = loom_text_embed_decoder_initialize(decoder);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(embed->allocator, decoder);
      decoder = NULL;
    }
  }
  *out_decoder = decoder;
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decoder_feed(
    loom_text_embed_decoder_t* decoder, const int32_t* tokens, size_t count,
    char* text, size_t capacity, size_t* out_size) {
  *out_size = 0;
  size_t fed = 0;
  while (fed < count) {
    iree_host_size_t consumed = 0, written = 0;
    iree_status_t status = iree_tokenizer_decode_state_feed(
        decoder->state,
        (iree_tokenizer_token_id_list_t){count - fed, tokens + fed},
        iree_make_mutable_string_view(text + *out_size, capacity - *out_size),
        &consumed, &written);
    if (!iree_status_is_ok(status)) EMBED_RETURN(decoder->embed, status);
    fed += consumed;
    *out_size += written;
    if (!consumed && !written) {
      EMBED_RETURN(decoder->embed,
                   iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "decoder output buffer full"));
    }
  }
  return 0;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_decoder_reset(
    loom_text_embed_decoder_t* decoder) {
  iree_tokenizer_decode_state_deinitialize(decoder->state);
  decoder->state = NULL;
  EMBED_RETURN(decoder->embed, loom_text_embed_decoder_initialize(decoder));
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_decoder_free(
    loom_text_embed_decoder_t* decoder) {
  if (!decoder) return;
  if (decoder->state) iree_tokenizer_decode_state_deinitialize(decoder->state);
  iree_allocator_free(decoder->embed->allocator, decoder);
}

// Appends to out, failing on overflow.
static iree_status_t loom_text_embed_append(char* out, size_t capacity,
                                            size_t* size,
                                            iree_string_view_t text) {
  if (*size + text.size > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "normalized output buffer full");
  }
  memcpy(out + *size, text.data, text.size);
  *size += text.size;
  return iree_ok_status();
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_output_feed(
    loom_text_embed_request_t* request, const char* text, size_t text_size,
    int final, char* out, size_t capacity, size_t* out_size) {
  loom_text_embed_t* embed = request->embed;
  *out_size = 0;
  if (request->finished) return 0;
  iree_status_t status = iree_string_builder_append_string(
      &request->response, iree_make_string_view(text, text_size));
  const iree_string_view_t response =
      iree_string_builder_view(&request->response);
  const loom_serve_text_chat_output_phase_t phase =
      final ? LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE
            : LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING;
  while (iree_status_is_ok(status)) {
    loom_serve_text_chat_output_t output = {0};
    status = loom_serve_text_chat_output_next(&request->chat, response, phase,
                                              embed->allocator, &output);
    if (!iree_status_is_ok(status) || !output.advanced) break;
    if (output.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_HIDDEN) continue;
    const bool reasoning = output.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING;
    if (output.text.size) {
      const size_t begin = (size_t)(output.text.data - response.data);
      if (reasoning) {
        request->reasoned = true;
        request->reasoning_end = begin + output.text.size;
      } else if (request->reasoned && !request->closed) {
        request->closed = true;
        request->close_begin = request->reasoning_end;
        request->close_end = begin;
      }
    }
    if (reasoning != request->reasoning_open && output.text.size) {
      status = loom_text_embed_append(
          out, capacity, out_size,
          reasoning ? IREE_SV("<think>") : IREE_SV("</think>"));
      request->reasoning_open = reasoning;
    }
    if (iree_status_is_ok(status)) {
      status = loom_text_embed_append(out, capacity, out_size, output.text);
    }
  }
  if (iree_status_is_ok(status) && final) {
    if (request->reasoning_open) {
      status = loom_text_embed_append(out, capacity, out_size,
                                      IREE_SV("</think>"));
      request->reasoning_open = false;
    }
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_output_close(
    loom_text_embed_request_t* request, char* out, size_t capacity,
    size_t* out_size) {
  *out_size = 0;
  if (!request->closed) return 0;
  const iree_string_view_t response =
      iree_string_builder_view(&request->response);
  EMBED_RETURN(request->embed,
               loom_text_embed_append(
                   out, capacity, out_size,
                   iree_make_string_view(
                       response.data + request->close_begin,
                       request->close_end - request->close_begin)));
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_output_tools(
    loom_text_embed_request_t* request, int truncated, char* json,
    size_t capacity, size_t* out_size) {
  loom_text_embed_t* embed = request->embed;
  *out_size = 0;
  request->finished = true;
  loom_serve_text_tool_calls_t calls;
  loom_serve_text_tool_calls_initialize(embed->allocator, &calls);
  loom_serve_text_chat_scratch_t scratch;
  loom_serve_text_chat_scratch_initialize(embed->allocator, &scratch);
  loom_serve_text_chat_completion_t completion = {0};
  iree_status_t status = loom_serve_text_chat_complete(
      &request->chat, iree_string_builder_view(&request->response),
      truncated != 0, request->serial, &calls, &scratch, embed->allocator,
      &completion);
  iree_string_builder_t builder;
  iree_string_builder_initialize(embed->allocator, &builder);
  if (iree_status_is_ok(status) && calls.count) {
    status = loom_serve_text_tool_calls_write_json(&calls, false, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_embed_append(json, capacity, out_size,
                                    iree_string_builder_view(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  loom_serve_text_chat_completion_deinitialize(&completion);
  loom_serve_text_chat_scratch_deinitialize(&scratch);
  loom_serve_text_tool_calls_deinitialize(&calls);
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_pin(
    loom_text_embed_t* embed, uint32_t row,
    loom_text_embed_checkpoint_t** out_checkpoint) {
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  iree_status_t status =
      loom_serve_text_row_try_pin(loom_text_embed_row(embed, row), &checkpoint);
  *out_checkpoint = (loom_text_embed_checkpoint_t*)checkpoint;
  if (iree_status_is_ok(status) && !checkpoint) return 1;
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_restore(
    loom_text_embed_t* embed, uint32_t row,
    const loom_text_embed_checkpoint_t* checkpoint, int* restored) {
  bool done = false;
  loom_serve_text_row_t* handle = loom_text_embed_row(embed, row);
  loom_serve_text_row_release_reservation(handle);
  embed->extents[row] = 0;
  iree_status_t status = loom_serve_text_row_try_restore(
      handle, (const loom_serve_text_checkpoint_t*)checkpoint, &done);
  *restored = done;
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_checkpoint_release(
    loom_text_embed_t* embed, loom_text_embed_checkpoint_t* checkpoint) {
  (void)embed;
  loom_serve_text_checkpoint_release(
      (loom_serve_text_checkpoint_t*)checkpoint);
}


//===----------------------------------------------------------------------===//
// Checkpoint images: a checkpoint's retained state as portable host bytes.
//===----------------------------------------------------------------------===//

// Image layout: the fixed part (recurrent state, then carry) and one page per
// block of block_size tokens, each page holding that block of every cache
// plane. Loom's own snapshot order is plane-major; ranges below map between.
typedef struct loom_text_embed_image_plan_t {
  iree_host_size_t count;
  // Device ranges in Loom's snapshot order (for this checkpoint's IDs).
  loom_serve_snapshot_range_t* device;
  // The same logical ranges at their image offsets (buffer 0 = host image).
  loom_serve_snapshot_range_t* image;
} loom_text_embed_image_plan_t;

static loom_serve_text_state_t* loom_text_embed_state(loom_text_embed_t* embed) {
  return &embed->model->state;
}

static iree_device_size_t loom_text_embed_recurrent_bytes(
    loom_serve_text_state_t* state) {
  return state->recurrent.buffers && state->recurrent.buffers[0]
             ? iree_hal_buffer_byte_length(state->recurrent.buffers[0])
             : 0;
}

LOOM_TEXT_EMBED_EXPORT void loom_text_embed_image_layout(
    loom_text_embed_t* embed, loom_text_embed_image_layout_t* out_layout) {
  loom_serve_text_state_t* state = loom_text_embed_state(embed);
  const loom_serve_text_storage_t* storage = &state->storage;
  uint64_t page = 0;
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    page += (uint64_t)storage->regions[r].blocks.count *
            storage->regions[r].blocks.block_bytes;
  }
  out_layout->block_size = (uint32_t)storage->block_size;
  out_layout->fixed_bytes =
      loom_text_embed_recurrent_bytes(state) + storage->carry_stride;
  out_layout->page_bytes = page;
  out_layout->supported = state->memory.buffers[1] != NULL;
}

// recurrent_slot UINT32_MAX plans a suspended checkpoint: only lengths and
// the image side are meaningful then.
static iree_status_t loom_text_embed_image_plan(
    loom_serve_text_state_t* state, iree_host_size_t checkpoint_index,
    uint32_t recurrent_slot, uint32_t block_count, const uint32_t* blocks,
    loom_text_embed_image_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  const loom_serve_text_storage_t* storage = &state->storage;
  iree_hal_buffer_t* views[5] = {0};
  if (state->recurrent.buffers && state->recurrent.buffers[0]) {
    views[TEXT_RECURRENT - TEXT_CONTROL] =
        state->recurrent.buffers[recurrent_slot == UINT32_MAX ? 0
                                                              : recurrent_slot];
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(
      storage, state->row_count + checkpoint_index, views, true, block_count,
      blocks, state->host_allocator, &out_plan->count, &out_plan->device));
  iree_status_t status = iree_allocator_malloc_array(
      state->host_allocator, out_plan->count, sizeof(*out_plan->image),
      (void**)&out_plan->image);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(state->host_allocator, out_plan->device);
    out_plan->device = NULL;
    return status;
  }
  const iree_host_size_t fixed_count =
      (views[TEXT_RECURRENT - TEXT_CONTROL] ? 1 : 0) +
      (storage->carry_stride ? 1 : 0);
  uint64_t fixed = 0, page = 0;
  for (iree_host_size_t i = 0; i < fixed_count; ++i) {
    out_plan->image[i] = (loom_serve_snapshot_range_t){
        0, fixed, out_plan->device[i].length};
    fixed += out_plan->device[i].length;
  }
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    page += (uint64_t)storage->regions[r].blocks.count *
            storage->regions[r].blocks.block_bytes;
  }
  iree_host_size_t i = fixed_count;
  uint64_t unit_offset = 0;
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    const loom_serve_block_region_t* region = &storage->regions[r].blocks;
    for (iree_host_size_t plane = 0; plane < region->count; ++plane) {
      for (uint32_t block = 0; block < block_count; ++block, ++i) {
        out_plan->image[i] = (loom_serve_snapshot_range_t){
            0, fixed + block * page + unit_offset, region->block_bytes};
      }
      unit_offset += region->block_bytes;
    }
  }
  return iree_ok_status();
}

static void loom_text_embed_image_plan_free(loom_serve_text_state_t* state,
                                            loom_text_embed_image_plan_t* plan) {
  iree_allocator_free(state->host_allocator, plan->device);
  iree_allocator_free(state->host_allocator, plan->image);
}

// Moves every range between device state and the image, 64 at a time, then
// waits for the transfers.
static iree_status_t loom_text_embed_image_transfer(
    loom_serve_text_state_t* state, const loom_text_embed_image_plan_t* plan,
    uint8_t* image, bool upload) {
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    buffers[i] = *loom_serve_text_state_allocation(state, i);
  }
  iree_hal_transfer_operation_t operations[64];
  iree_host_size_t operation_count = 0;
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_drain(state->execution);
  for (iree_host_size_t i = 0; i < plan->count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_snapshot_range_t* device = &plan->device[i];
    uint8_t* bytes = image + plan->image[i].offset;
    operations[operation_count++] =
        upload ? (iree_hal_transfer_operation_t){
                     .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                     .upload = {.source = bytes,
                                .target_buffer = buffers[device->buffer_index],
                                .target_offset = device->offset,
                                .length = device->length}}
               : (iree_hal_transfer_operation_t){
                     .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
                     .download = {.source_buffer = buffers[device->buffer_index],
                                  .source_offset = device->offset,
                                  .target = bytes,
                                  .length = device->length}};
    if (operation_count == IREE_ARRAYSIZE(operations)) {
      status = loom_serve_execution_transfer(state->execution, operation_count,
                                             operations, &completion);
      operation_count = 0;
    }
  }
  if (iree_status_is_ok(status) && operation_count) {
    status = loom_serve_execution_transfer(state->execution, operation_count,
                                           operations, &completion);
  }
  return iree_status_join(status, loom_serve_execution_drain(state->execution));
}

static iree_device_size_t loom_text_embed_image_size(
    const loom_text_embed_image_plan_t* plan) {
  iree_device_size_t size = 0;
  for (iree_host_size_t i = 0; i < plan->count; ++i) size += plan->image[i].length;
  return size;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_checkpoint_export(
    loom_text_embed_t* embed, const loom_text_embed_checkpoint_t* handle,
    uint8_t* image, size_t capacity, size_t* out_size,
    loom_text_embed_checkpoint_info_t* out_info) {
  const loom_serve_text_checkpoint_t* checkpoint =
      (const loom_serve_text_checkpoint_t*)handle;
  loom_serve_text_state_t* state = loom_text_embed_state(embed);
  loom_serve_text_state_checkpoint_t* endpoint = checkpoint->state;
  *out_size = 0;
  out_info->position = endpoint->position;
  out_info->source_position = endpoint->source_position;
  out_info->has_prediction = checkpoint->has_prediction;
  out_info->token = checkpoint->token;
  if (endpoint->snapshot || endpoint->recurrent_slot == UINT32_MAX) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                         "checkpoint is not resident"));
  }
  loom_text_embed_image_plan_t plan;
  iree_status_t status = loom_text_embed_image_plan(
      state, (iree_host_size_t)(endpoint - state->checkpoints.values),
      endpoint->recurrent_slot, endpoint->block_count, endpoint->blocks, &plan);
  if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
  const iree_device_size_t size = loom_text_embed_image_size(&plan);
  *out_size = size;
  if (size > capacity) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "checkpoint image needs %" PRIu64 " bytes",
                              (uint64_t)size);
  } else {
    status = loom_text_embed_image_transfer(state, &plan, image, false);
  }
  loom_text_embed_image_plan_free(state, &plan);
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_checkpoint_import(
    loom_text_embed_t* embed, const loom_text_embed_checkpoint_info_t* info,
    const uint8_t* image, size_t size,
    loom_text_embed_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  loom_serve_text_state_t* state = loom_text_embed_state(embed);
  if (!state->memory.buffers[1]) {
    EMBED_RETURN(embed,
                 iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "checkpoint images need elastic pooled state"));
  }
  if (info->position == 0 || info->position > state->storage.blocks_per_row *
                                                  state->storage.block_size) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "checkpoint image position %u is out "
                                         "of range",
                                         info->position));
  }
  const uint32_t needed = (uint32_t)iree_host_size_ceil_div(
      info->position, state->storage.block_size);
  if (!state->checkpoints.pool.available || !state->recurrent.pool.available ||
      needed > state->cache.pool.available) {
    return 1;
  }
  // As Loom resumes a suspended endpoint: fresh IDs, committed backing, then
  // the image's contents in logical order.
  uint32_t index = 0;
  loom_serve_block_pool_acquire(&state->checkpoints.pool, 1, &index);
  loom_serve_text_state_checkpoint_t* endpoint =
      &state->checkpoints.values[index];
  endpoint->position = info->position;
  endpoint->source_position = info->source_position;
  loom_serve_block_pool_acquire(&state->cache.pool, needed, endpoint->blocks);
  endpoint->block_count = needed;
  loom_serve_block_pool_acquire(&state->recurrent.pool, 1,
                                &endpoint->recurrent_slot);
  loom_text_embed_image_plan_t plan;
  iree_status_t status = loom_text_embed_image_plan(
      state, index, endpoint->recurrent_slot, needed, endpoint->blocks, &plan);
  bool admitted = false;
  if (iree_status_is_ok(status) && loom_text_embed_image_size(&plan) != size) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "checkpoint image is %zu bytes, expected %" PRIu64,
                              size, (uint64_t)loom_text_embed_image_size(&plan));
  }
  loom_serve_memory_range_t* commitments = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(state->host_allocator, plan.count + 1,
                                         sizeof(*commitments),
                                         (void**)&commitments);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t count = 0;
    for (iree_host_size_t i = 0; i < plan.count; ++i) {
      loom_serve_virtual_buffer_t* buffer =
          state->memory.buffers[plan.device[i].buffer_index];
      if (buffer) {
        commitments[count++] = (loom_serve_memory_range_t){
            buffer, plan.device[i].offset, plan.device[i].length};
      }
    }
    status = loom_serve_memory_pool_try_commit(state->memory.pool, count,
                                               commitments, &admitted);
  }
  iree_allocator_free(state->host_allocator, commitments);
  if (iree_status_is_ok(status) && admitted) {
    status = loom_text_embed_image_transfer(state, &plan, (uint8_t*)image, true);
  }
  if (plan.device) loom_text_embed_image_plan_free(state, &plan);
  if (!iree_status_is_ok(status) || !admitted) {
    loom_serve_text_state_checkpoint_release(endpoint);
    if (!iree_status_is_ok(status)) EMBED_RETURN(embed, status);
    return 1;
  }
  loom_serve_text_checkpoint_t* checkpoint = &embed->model->checkpoints[index];
  *checkpoint = (loom_serve_text_checkpoint_t){
      .model = embed->model,
      .references = 1,
      .state = endpoint,
      .has_prediction = info->has_prediction != 0,
      .token = info->token};
  *out_checkpoint = (loom_text_embed_checkpoint_t*)checkpoint;
  return 0;
}

LOOM_TEXT_EMBED_EXPORT size_t loom_text_embed_request_media(
    const loom_text_embed_request_t* request, uint32_t* offsets,
    uint32_t* counts, size_t capacity) {
  if (!request->has_input) return 0;
  const iree_host_size_t count = request->input.media_span_count;
  for (iree_host_size_t i = 0; i < count && i < capacity; ++i) {
    offsets[i] = request->input.media_spans[i].input_offset;
    counts[i] = request->input.media_spans[i].input_count;
  }
  return count;
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_request_project(
    loom_text_embed_request_t* request) {
  loom_text_embed_t* embed = request->embed;
  if (!request->has_input || request->media) return 0;
  // Prepare the image component on first use.
  loom_serve_finite_model_t* image_model = NULL;
  iree_status_t status = iree_ok_status();
  for (;;) {
    status = loom_serve_finite_component_prepare(embed->image, &image_model);
    if (!iree_status_is_ok(status) || image_model) break;
    loom_serve_finite_component_advance(embed->image, false);
    usleep(200);
  }
  if (iree_status_is_ok(status) && !embed->projector) {
    loom_serve_host_t* host = loom_serve_device_host(
        loom_serve_text_model_device(embed->model));
    iree_async_proactor_t* proactor = NULL;
    status = iree_async_proactor_pool_get(loom_serve_host_proactor_pool(host),
                                          0, &proactor);
    if (iree_status_is_ok(status)) {
      const iree_host_size_t capacity =
          embed->rows *
          loom_serve_text_model_chat_policy(embed->model)->images.maximum_count;
      status = loom_serve_text_media_projector_allocate(
          image_model, capacity ? capacity : 1, loom_serve_host_executor(host),
          proactor, (loom_serve_wake_callback_t){loom_text_embed_wake, NULL},
          embed->allocator, &embed->projector);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_media_request_allocate(
        &request->chat, embed->allocator, &request->media);
  }
  // Offer, seal and advance the projection window until every image of the
  // request has published its features.
  bool offered = false;
  while (iree_status_is_ok(status)) {
    status = loom_serve_text_media_request_take_status(request->media);
    if (!iree_status_is_ok(status)) break;
    if (loom_serve_text_media_request_bind(request->media, &request->input, 0))
      break;
    if (!offered) {
      offered = loom_serve_text_media_projector_offer(embed->projector,
                                                      request->media, 0);
      loom_serve_text_media_projector_start(embed->projector);
    }
    bool progress = false;
    status = loom_serve_text_media_projector_advance(embed->projector, 0,
                                                     &progress);
    loom_serve_finite_component_advance(embed->image, false);
    if (!progress) usleep(100);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_media_request_begin(request->media);
  }
  EMBED_RETURN(embed, status);
}

LOOM_TEXT_EMBED_EXPORT int loom_text_embed_row_prefill_request(
    loom_text_embed_t* embed, uint32_t row, loom_text_embed_request_t* request,
    size_t offset, size_t count, int select) {
  if (!request->has_input || offset + count > request->input.count) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "request input range"));
  }
  iree_host_size_t shape = 0;
  if (!loom_text_embed_shape(embed, 1, count, &shape)) {
    EMBED_RETURN(embed, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                         "no shape for %zu input tokens",
                                         count));
  }
  const loom_serve_text_span_t span = {
      .row_index = row,
      .token_count = count,
      .token_ids = request->input.tokens + offset,
      .flags = select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0,
      .input_offset = offset,
      .media = request->media
                   ? loom_serve_text_media_request_input(request->media)
                   : NULL,
  };
  iree_status_t status =
      loom_serve_text_model_epoch(embed->model, shape, 1, &span);
  if (iree_status_is_ok(status) && request->media) {
    loom_serve_text_media_request_consume(request->media, offset + count);
  }
  EMBED_RETURN(embed, status);
}
