cmake_minimum_required(VERSION 3.21)
project(new-ime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_FCITX5 "Build fcitx5 engine plugin (requires fcitx5 dev headers)" OFF)
option(ENABLE_ONNX "Build with ONNX Runtime inference (otherwise uses mock)" OFF)

# ---- Protobuf ----
find_package(Protobuf 3.12)
if(Protobuf_FOUND)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS engine/protocol/new_ime.proto)
    add_library(newime_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
    target_include_directories(newime_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(newime_proto PUBLIC protobuf::libprotobuf-lite)
    set(HAS_PROTOBUF TRUE)
else()
    message(STATUS "Protobuf not found - building without protobuf support")
    set(HAS_PROTOBUF FALSE)
endif()

# ---- Inference server ----
add_executable(new-ime-server
    engine/server/src/main.cpp
    engine/server/src/socket_server.cpp
    engine/server/src/conversion_engine.cpp
    engine/server/src/ctc_decoder.cpp
)
target_include_directories(new-ime-server PRIVATE server/src)

if(ENABLE_ONNX)
    find_package(onnxruntime REQUIRED)
    target_link_libraries(new-ime-server PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(new-ime-server PRIVATE HAS_ONNX=1)
endif()

if(HAS_PROTOBUF)
    target_link_libraries(new-ime-server PRIVATE newime_proto)
    target_compile_definitions(new-ime-server PRIVATE HAS_PROTOBUF=1)
endif()

# ---- Engine shared components (no fcitx5 dependency) ----
add_library(newime_engine_core STATIC
    engine/src/composing_text.cpp
    engine/src/preedit.cpp
)
target_include_directories(newime_engine_core PUBLIC engine/src)

# ---- fcitx5 engine plugin ----
if(ENABLE_FCITX5)
    find_package(Fcitx5Core REQUIRED)
    find_package(Fcitx5Utils REQUIRED)
    find_package(Fcitx5Config REQUIRED)

    include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake")

    add_library(fcitx5-new-ime SHARED
        engine/src/engine.cpp
        engine/src/server_connector.cpp
    )
    set_target_properties(fcitx5-new-ime PROPERTIES PREFIX "")
    target_compile_definitions(fcitx5-new-ime PRIVATE HAS_FCITX5=1)
    target_link_libraries(fcitx5-new-ime PRIVATE
        newime_engine_core
        Fcitx5::Core
        Fcitx5::Config
    )

    if(HAS_PROTOBUF)
        target_link_libraries(fcitx5-new-ime PRIVATE newime_proto)
        target_compile_definitions(fcitx5-new-ime PRIVATE HAS_PROTOBUF=1)
    endif()

    # Install engine plugin
    install(TARGETS fcitx5-new-ime DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5")

    # Install metadata
    install(FILES engine/new-ime-addon.conf
        DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
        RENAME new-ime-addon.conf)
    install(FILES engine/new-ime-im.conf
        DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod"
        RENAME new-ime-im.conf)
endif()

# Install server
install(TARGETS new-ime-server DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/new-ime")

# ---- Tests ----
enable_testing()

add_executable(test_composing_text models/tests/cpp/test_composing_text.cpp)
target_link_libraries(test_composing_text PRIVATE newime_engine_core)
add_test(NAME test_composing_text COMMAND test_composing_text)

add_executable(test_ctc_decoder models/tests/cpp/test_ctc_decoder.cpp)
target_link_libraries(test_ctc_decoder PRIVATE newime_engine_core)
target_include_directories(test_ctc_decoder PRIVATE server/src)
# Link CTC decoder source directly for testing
target_sources(test_ctc_decoder PRIVATE engine/server/src/ctc_decoder.cpp)
add_test(NAME test_ctc_decoder COMMAND test_ctc_decoder)
