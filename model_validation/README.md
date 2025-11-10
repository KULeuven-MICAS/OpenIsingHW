# **Model Validation**

This folder contains model validation scripts for different works, including:

| Architecture | Script | Description |
|:-|:-|:-|
| SACHI [[paper](https://ieeexplore.ieee.org/document/10476402)] | [sachi.py](./sachi.py) | A parallel-updating Ising architecture, SACHI, using Hopfield solver |
| PRIM-CAEFA [[paper](https://ieeexplore.ieee.org/document/10849084)] | [prim_caefa.py](./prim_caefa.py) | A serial-updating Ising architecture, PRIM-CAEFA, using Simulated Annealing solver |
| SB-V1 [[paper](https://ieeexplore.ieee.org/document/8892209)] | [fpga_asb.py](./fpga_asb.py) | A parallel-updating Ising architecture (V1) using Simulated Bifurcation solver |
| SB-V2 [[paper](https://ieeexplore.ieee.org/document/10460551)] | [fpga_asb_v2.py](./fpga_asb_v2.py) | A parallel-updating Ising architecture (V2) using Simulated Bifurcation solver |

Here each validation script compares the latency and energy estimation from the model against the reported value in the paper. Note that though the energy value is modeled for PRIM-CAEFA, FPGA asb (V1), and FPGA asb (V2), it is not reported in the original paper.

The top validation script is [validation_top.py](./validation_top.py), which runs and plots the results of all validation scripts into the output folder.

## Validated workloads

Here we list all the validated workloads for each work. Note that:
- Validation can only be done on workloads evaluated in the original paper, though the workload may be unfeasible.
- The workload name is from the original paper.

### SACHI

| Workload | Graph topology | Ising nodes (N) | Average graph degree | Weight precision | With magnetic field (h) | Latency mismatch (%) | Energy mismatch (%) |
|:-|:-|:-|:-|:-|:-|:-|:-|
| TSP_1K | Dense | 1000 | 999 | 4b | N | 0 | -0.1 |
| MC_1K | King's graph | 1,000 | 8 | 4b | N | 0 | 1.8 |
| MC_500 | King's graph | 500 | 8 | 2b | N | 0 | -3.1 |
| MC_100K | King's graph | 100,000 | 8 | 2b | N | 0 | 1.8 |
| MC_200K | King's graph | 200,000 | 8 | 2b | N | 0 | 1.8 |
| MC_300K | King's graph | 300,000 | 8 | 2b | N | 0 | 1.8 |
| MC_1M | King's graph | 1,000,000 | 8 | 2b | N | 0 | 1.8 |
| Average | | | | | | 0 | 0.8 |

### PRIM-CAEFA

| Workload | Graph topology | Ising nodes (N) | Average graph degree | Weight precision | With magnetic field (h) | Hardware mode | Latency mismatch (%) | Energy mismatch (%) |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| G22_2K_IM | Random sparse | 2,000 | 19.99 | 2b | N | IM | 0 | NA |
| G22_2K_BM | Random sparse | 2,000 | 19.99 | 2b | N | BM | 0 | NA |
| G22_2K_SM | Random sparse | 2,000 | 19.99 | 2b | N | SM | 0 | NA |
| G39_2K_IM | Random sparse | 2,000 | 11.778 | 2b | N | IM | 0 | NA |
| G39_2K_BM | Random sparse | 2,000 | 11.778 | 2b | N | BM | 0 | NA |
| G39_2K_SM | Random sparse | 2,000 | 11.778 | 2b | N | SM | 0 | NA |
| K2000_2K_IM | Dense | 2,000 | 1,999 | 2b | N | IM | 0 | NA |
| Average | | | | | | | 0 | NA |

### SB-V1

| Workload | Graph topology | Ising nodes (N) | Average graph degree | Weight precision | With magnetic field (h) | Latency mismatch (%) | Energy mismatch (%) |
|:-|:-|:-|:-|:-|:-|:-|:-|
| K2000 | Dense | 2,048 | 2,047 | 1b | N | 0 | NA |
| K4000 | Dense | 4,096 | 4,095 | 1b | N | 0 | NA |
| Average | | | | | | 0 | NA |

### SB-V2

| Workload | Graph topology | Ising nodes (N) | Average graph degree | Weight precision | With magnetic field (h) | Available cores | Latency mismatch (%) | Energy mismatch (%) |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| K2000_C2 | Dense | 2,048 | 2,047 | 1b | N | 2 | 0 | NA |
| K4000_C4 | Dense | 4,096 | 4,095 | 1b | N | 4 | 0 | NA |
| K8000_C8 | Dense | 8,192 | 8,191 | 1b | N | 8 | 0 | NA |
| K4000_C2 | Dense | 4,096 | 4,095 | 1b | N | 2 | 0 | NA |
| K8000_C4 | Dense | 8,192 | 8,191 | 1b | N | 4 | 0 | NA |
| K16000_C8 | Dense | 16,384 | 16,383 | 1b | N | 8 | 0 | NA |
| K8000_C2 | Dense | 8,192 | 8,191 | 1b | N | 2 | 0 | NA |
| K16000_C4 | Dense | 16,384 | 16,383 | 1b | N | 4 | 0 | NA |
| K32000_C8 | Dense | 32,768 | 32,767 | 1b | N | 8 | 0 | NA |
| K16000_C2 | Dense | 16,384 | 16,383 | 1b | N | 2 | 0 | NA |
| Average | | | | | | | 0 | NA |
