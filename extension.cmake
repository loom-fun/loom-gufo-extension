# Loom model packages (https://huggingface.co/loom-fun) as Gufo text services.
gufo_extension_target(
  NAME gufo_loom
  SOURCES
    src/loom_text.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    PkgConfig::libcurl
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/loom)
