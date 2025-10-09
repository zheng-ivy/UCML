cmake_minimum_required(VERSION 3.14)
project(ucml_separators C)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(SCIP REQUIRED)           # assumes SCIP >= 9.x is installed & findable

add_library(ucsep_minupdown SHARED
  sepa_minupdown.c
)

target_include_directories(ucsep_minupdown PRIVATE ${SCIP_INCLUDE_DIRS})
target_link_libraries(ucsep_minupdown PRIVATE ${SCIP_LIBRARIES})

# Optional: export a convenience function so host code can call one symbol
# to include your plugins:
target_sources(ucsep_minupdown PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sepa_minupdown.h)
