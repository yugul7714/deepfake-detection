cmake_minimum_required(VERSION 3.10)
project(video_player)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)
find_package(OpenGL REQUIRED)

include_directories(
    ${SDL2_INCLUDE_DIRS}
    /opt/homebrew/include       # Adjust for your system (Homebrew or system paths)
)

link_directories(
    ${SDL2_LIBRARY_DIRS}
    /opt/homebrew/lib
)

add_executable(video_player main.cpp)

target_link_libraries(video_player
    ${SDL2_LIBRARIES}
    -lavformat
    -lavcodec
    -lavutil
    -lswscale
    -lGLEW
    ${OPENGL_gl_LIBRARY}
    "-framework OpenGL"
)