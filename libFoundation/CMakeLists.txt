# CMakeList.txt: libFoundation에 대한 CMake 프로젝트, 여기에 소스를 포함하고
# 프로젝트 특정 논리를 정의합니다.
#
cmake_minimum_required (VERSION 3.8)

# 지원되는 경우 MSVC 컴파일러에 대해 핫 다시 로드 사용하도록 설정합니다.
#if (POLICY CMP0141)
#  cmake_policy(SET CMP0141 NEW)
#  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
#endif()

if(WIN32)
else()
	set(CMAKE_CXX_COMPILER "/usr/bin/g++")
	set(CMAKE_C_COMPILER "/usr/bin/gcc")
endif()

project (libFoundation)


if (WIN32)
    include_directories("Library/onetbb_win/include/tbb")
    link_directories("Library/onetbb_win/lib/intel64/vc14")

    set(LIB_FOUNDATION_NAME libFoundation)
else ()
    include_directories("~/source/oneTBB/include/tbb")
    link_directories("~/source/oneTBB/build/gnu_11.4_cxx11_64_relwithdebinfo")
    set(LIB_FOUNDATION_NAME Foundation)
endif()
include_directories("../libFoundation")

# 이 프로젝트의 라이브러리에 소스를 추가합니다.
add_library(${LIB_FOUNDATION_NAME} STATIC
  "Component/Component.cpp"
  "Component/Component.h"
  "Component/Component.hpp"
  "DateTime/DateTime.cpp"
  "DateTime/DateTime.h"
  "Timer/Timer.cpp"
  "Timer/Timer.h"
  "FileSystem/FileSystem.h" 
  "FileSystem/FileSystem.cpp"
  "Json/JsonObject.cpp"
  "Json/JsonObject.h"
  "MemoryPool/MemoryPool.h" 
  "MemoryPool/MemoryObjectPool.h" 
  "GoLogger/GoLogger.h")

# add_executable(libFoundationTest "libFoundation.cpp" "libFoundation.h")

# Library Linking
target_link_libraries(${LIB_FOUNDATION_NAME} PRIVATE tbb)
#target_link_libraries(libFoundation PRIVATE tbb_debug)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${LIB_FOUNDATION_NAME} PROPERTY CXX_STANDARD 20)
endif()

# TODO: 필요한 경우 테스트를 추가하고 대상을 설치합니다.
