cmake_minimum_required(VERSION 3.14)

project(Cuper)

set(PLATFORM
    xilinx_u280_xdma_201920_3
    CACHE STRING "Target FPGA platform")

set(TOP Cuper)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-write-strings")

find_package(TAPA REQUIRED)
find_package(SDx REQUIRED)

add_executable(Cuper)
target_sources(Cuper PRIVATE src/Cuper_host.cpp src/Cuper.cpp)
target_link_libraries(Cuper PRIVATE tapa::tapa)

add_tapa_target(
  hls
  --enable-hbm-binding-adjustment
  --floorplan-opt-priority SLR_CROSSING_PRIORITIZED
  INPUT src/Cuper.cpp
  TOP ${TOP}
  # CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config_8.ini
  CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config_16.ini
  # CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config_24.ini
  CONSTRAINT ${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  --read-only-args SpElement_list_ptr
  --read-only-args Matrix_data*
  --read-only-args X
  --write-only-args Y_out
  PLATFORM ${PLATFORM})

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  # --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config_24.ini
  --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config_16.ini
  # --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config_8.ini
  --vivado.synth.jobs 8
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=1
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=EarlyBlockPlacement
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  INPUT hls
  HW_EMU_XCLBIN hw_emu_xclbin
  HW_XCLBIN hw_xclbin)

add_custom_target(
  swsim
  COMMAND $<TARGET_FILE:Cuper> ../matrices/nasa4704/nasa4704.mtx
  DEPENDS Cuper
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(
  hwsim
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME> $<TARGET_FILE:Cuper> ../matrices/nasa4704/nasa4704.mtx
  DEPENDS Cuper ${hw_emu_xclbin}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
