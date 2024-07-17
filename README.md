# Cuper

Cuper is a high-performance SpMV accelerator on HBM-equipped FPGAs.

## Installation

```text
mkdir build
cd build
cmake ..
```

## Software Emulation

```text
make swsim
```

## Hardware Emulation

```text
make hwsim
```

## Generate Bitstream

```text
sh run_generate.sh
```

## Run Cuper on FPGA

```text
BITFILE=../bitfile/Cuper_xilinx_u280_xdma_201920_3.xclbin ./Cuper ../matrices/nasa4704/nasa4704.mtx
```

## Reference

E. Yi, Y. Duan, Y. Bai, K. Zhao, Z. Jin and W. Liu, "Cuper: Customized Dataflow and Perceptual Decoding for Sparse Matrix-Vector Multiplication on HBM-Equipped FPGAs," 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), Valencia, Spain, 2024, pp. 1-6.