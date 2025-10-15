cmake_minimum_required(VERSION 3.16)
project(tinyraytracer LANGUAGES CXX)

# === Export compile commands to json for IDEs ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Compiler setup ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === Global output directory ===
set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})
# Ensure all build configs (Debug/Release) use the same output path
foreach(OUTPUTCONFIG IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${OUTPUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${OUTPUT_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${OUTPUT_DIR})
endforeach()

# === Enable ccache if available ===
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
else()
    message(STATUS "ccache not found — builds may be slower")
endif()

# === Find OpenCL ===
find_package(OpenCL REQUIRED)

# === Source files ===
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# === Create executable ===
add_executable(main_exec ${SRC_FILES})

# === Include directories ===
target_include_directories(main_exec PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/scene
    ${CMAKE_SOURCE_DIR}/src/types
    ${CMAKE_SOURCE_DIR}/src/kernels
)

# === Link with OpenCL ===
target_link_libraries(main_exec PRIVATE OpenCL::OpenCL)

# === Compile definitions ===
target_compile_definitions(main_exec PRIVATE 
    CL_TARGET_OPENCL_VERSION=300 
    KERNELS_DIR=\"${OUTPUT_DIR}/kernels\"
    TARGET_DEVICE_NAME=\"gfx1102\"
)

# === Copy kernels to output ===
file(GLOB_RECURSE KERNEL_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/kernels/*.cl
)

if (KERNEL_FILES)
    add_custom_command(TARGET main_exec POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}/kernels
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${KERNEL_FILES}
            ${OUTPUT_DIR}/kernels
        COMMENT "Copying OpenCL kernel files to ${OUTPUT_DIR}/kernels"
    )
endif()

# === Summary ===
message(STATUS "----------------------------------------------")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output Directory: ${OUTPUT_DIR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Source files:")
foreach(src ${SRC_FILES})
    message(STATUS "  ${src}")
endforeach()
if (KERNEL_FILES)
    message(STATUS "Kernels:")
    foreach(kernel ${KERNEL_FILES})
        message(STATUS "  ${kernel}")
    endforeach()
endif()
message(STATUS "----------------------------------------------")