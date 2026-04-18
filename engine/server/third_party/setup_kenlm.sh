#!/bin/bash
# Clone kenlm and apply the small CMake patches needed to build the Model
# runtime library without Boost / Eigen / zlib / compressed-stream deps.
#
# Upstream's top-level CMakeLists, util/CMakeLists, lm/CMakeLists, and
# lm/common/CMakeLists unconditionally pull Boost and include Boost-based
# util/stream sources. Those subsystems are only needed by lmplz / query /
# build_binary, not by lm::ngram::Model. We add a KENLM_LIBS_ONLY option
# that skips them, and otherwise leave the tree alone.
#
# Idempotent: re-running is safe. Patches are applied with `patch -p1
# --forward`, which is a no-op on already-patched trees.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ ! -d kenlm ]; then
    git clone --depth=1 https://github.com/kpu/kenlm.git
fi

patch -p1 -d kenlm --forward --reject-file=- <<'EOF'
diff --git a/CMakeLists.txt b/CMakeLists.txt
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -97,16 +97,21 @@ endif()
 # And our helper modules
 list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

-# We need boost
-find_package(Boost 1.41.0 REQUIRED COMPONENTS
-  program_options
-  system
-  thread
-  unit_test_framework
-)
-
-# Define where include files live
-include_directories(${Boost_INCLUDE_DIRS})
+# Boost is only needed for the command-line binaries (lmplz, query, etc)
+# and the test suite. The kenlm/kenlm_util libs themselves link fine
+# without it. -DKENLM_LIBS_ONLY=ON skips the find_package, drops the
+# binary and test targets, and still produces kenlm.lib + kenlm_util.lib.
+option(KENLM_LIBS_ONLY "Skip Boost / binaries / tests, build libs only" OFF)
+if(NOT KENLM_LIBS_ONLY)
+  find_package(Boost 1.41.0 REQUIRED COMPONENTS
+    program_options
+    system
+    thread
+    unit_test_framework
+  )
+  include_directories(${Boost_INCLUDE_DIRS})
+endif()

 set(THREADS_PREFER_PTHREAD_FLAG ON)
 find_package(Threads REQUIRED)
EOF
patch -p1 -d kenlm --forward --reject-file=- <<'EOF'
diff --git a/util/CMakeLists.txt b/util/CMakeLists.txt
--- a/util/CMakeLists.txt
+++ b/util/CMakeLists.txt
@@ -33,7 +33,15 @@ endif()

 # This directory has children that need to be processed
 add_subdirectory(double-conversion)
-add_subdirectory(stream)
+# util/stream pulls Boost.thread (multi_progress) which isn't required for
+# Model loading, so skip it in the libs-only configuration.
+if(NOT KENLM_LIBS_ONLY)
+  add_subdirectory(stream)
+endif()
+
+if(KENLM_LIBS_ONLY)
+  set(KENLM_UTIL_STREAM_SOURCE "")
+endif()

 add_library(kenlm_util ${KENLM_UTIL_DOUBLECONVERSION_SOURCE} ${KENLM_UTIL_STREAM_SOURCE} ${KENLM_UTIL_SOURCE})
 # Since headers are relative to `include/kenlm` at install time, not just `include`
EOF
patch -p1 -d kenlm --forward --reject-file=- <<'EOF'
diff --git a/lm/CMakeLists.txt b/lm/CMakeLists.txt
--- a/lm/CMakeLists.txt
+++ b/lm/CMakeLists.txt
@@ -40,9 +40,12 @@ target_include_directories(kenlm PUBLIC $<INSTALL_INTERFACE:include/kenlm>)
 target_compile_definitions(kenlm PUBLIC -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})

 # This directory has children that need to be processed
-add_subdirectory(builder)
-add_subdirectory(filter)
-add_subdirectory(interpolate)
+# Skip Boost/Eigen-dependent subdirs and binaries when only the libs are requested.
+if(NOT KENLM_LIBS_ONLY)
+  add_subdirectory(builder)
+  add_subdirectory(filter)
+  add_subdirectory(interpolate)
+endif()

 # Explicitly list the executable files to be compiled
 set(EXE_LIST
@@ -64,8 +67,10 @@ install(
   INCLUDES DESTINATION include
 )

-AddExes(EXES ${EXE_LIST}
-        LIBRARIES ${LM_LIBS})
+if(NOT KENLM_LIBS_ONLY)
+  AddExes(EXES ${EXE_LIST}
+          LIBRARIES ${LM_LIBS})
+endif()

 if(BUILD_TESTING)

EOF
patch -p1 -d kenlm --forward --reject-file=- <<'EOF'
diff --git a/lm/common/CMakeLists.txt b/lm/common/CMakeLists.txt
--- a/lm/common/CMakeLists.txt
+++ b/lm/common/CMakeLists.txt
@@ -11,12 +11,19 @@
 #    in case this variable is referenced by CMake files in the parent directory,
 #    we prefix all files with ${CMAKE_CURRENT_SOURCE_DIR}.
 #
-set(KENLM_LM_COMMON_SOURCE
-		${CMAKE_CURRENT_SOURCE_DIR}/model_buffer.cc
-		${CMAKE_CURRENT_SOURCE_DIR}/print.cc
-		${CMAKE_CURRENT_SOURCE_DIR}/renumber.cc
-		${CMAKE_CURRENT_SOURCE_DIR}/size_option.cc
-  PARENT_SCOPE)
+# The common sources (model_buffer, print, renumber, size_option) are only
+# used by lmplz / interpolation / the CLI binaries and pull Boost
+# (program_options, util/stream -> boost/thread). The run-time Model API
+# does not need them — skip the list when the caller asked for a
+# Boost-free libs-only build.
+if(KENLM_LIBS_ONLY)
+  set(KENLM_LM_COMMON_SOURCE "" PARENT_SCOPE)
+else()
+  set(KENLM_LM_COMMON_SOURCE
+		${CMAKE_CURRENT_SOURCE_DIR}/model_buffer.cc
+		${CMAKE_CURRENT_SOURCE_DIR}/print.cc
+		${CMAKE_CURRENT_SOURCE_DIR}/renumber.cc
+		${CMAKE_CURRENT_SOURCE_DIR}/size_option.cc
+    PARENT_SCOPE)
+endif()

 if(BUILD_TESTING)
   KenLMAddTest(TEST model_buffer_test
EOF

echo "[setup_kenlm] kenlm ready for CMake build with -DKENLM_LIBS_ONLY=ON"
