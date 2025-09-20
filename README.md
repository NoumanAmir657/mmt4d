# MMT4D

Implementation of mmt4d in C

## Compile
```bash
clang -O3 main.c -o main

# if compiling on/for RISC-V
clang -target riscv64 -march=rv64gv -mabi=lp64d main.c -o main
```
