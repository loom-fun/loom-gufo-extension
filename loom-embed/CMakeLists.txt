# libloom_text_embed.so: a Loom package's text model layer behind a C API,
# for serving processes that own their own scheduling and caching. Built in
# Loom's tree (added under loom/src/loom/serve/embed).
if(LOOM_ARCH_VM AND LOOM_ARTIFACT_VM_BINARY AND LOOM_ROUTE_AMDGPU_HAL)
add_library(loom_text_embed SHARED loom_text_embed.c)
target_include_directories(loom_text_embed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(loom_text_embed PRIVATE
  iree::base
  iree::base::internal::path
  iree::tokenizer
  iree::tokenizer::vocab
  iree::tooling::device_util
  loom::serve::runtime::device
  loom::serve::runtime::host
  loom::serve::server::configuration
  loom::serve::server::device_configuration
  loom::serve::service::text::chat
  loom::serve::service::text::model
  loom::serve::service::text::schedule
  loom::binding::c::target::configured
  loom::binding::c::target::amdgpu::iree_hal
)
set_target_properties(loom_text_embed PROPERTIES
  C_VISIBILITY_PRESET hidden
  OUTPUT_NAME loom_text_embed)
target_link_options(loom_text_embed PRIVATE
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
  "-Wl,--no-undefined")
endif()
