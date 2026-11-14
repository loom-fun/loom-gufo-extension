# Loom model packages (https://huggingface.co/loom-fun) as Gufo text models,
# on the package's own Loom kernels through libloom_text_embed.
gufo_extension_target(
  NAME gufo_loom
  SOURCES
    src/loom_runner.cpp
    src/loom_chat.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    PkgConfig::loom-text-embed
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/loom)
