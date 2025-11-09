# **Model Validation**

This folder contains model validation scripts for different works, including:
- sachi.py: validation script for a parallel-updating Ising architecture, SACHI, using Hopfield solver [paper](https://ieeexplore.ieee.org/document/10476402).
- prim_caefa.py: validation script for a serial-updating Ising architecture, PRIM-CAEFA, using Simulated Annealing solver [paper](https://ieeexplore.ieee.org/document/10849084).
- fpga_asb.py: validation script for a parallel-updating Ising architecture (V1) using Simulated Bifurcation solver [paper](https://ieeexplore.ieee.org/document/8892209).
- fpga_asb_v2.py: validation script for a parallel-updating Ising architecture (V2) using Simulated Bifurcation solver [paper](https://ieeexplore.ieee.org/document/10460551).

Here each validation script compares the latency and energy estimation from the model against the reported value in the paper. Note that though the energy value is modeled for PRIM-CAEFA, FPGA asb (V1), and FPGA asb (V2), it is not reported in the original paper.

The top validation script is [validation_top.py](./validation_top.py), which runs and plots the results of all validation scripts into the output folder.

## Validated workloads

Here we list all the validated workloads for each work. Note that:
- Validation can only be done on workloads evaluated in the original paper, though the workload may be unfeasible.
- The workload name is from the original paper.

### SACHI
- TSP_1K: a 1000-node dense problem, topology: fully connected, weight: 4b
- MC_1K: a 1000-node MaxCut-like problem, topology: King's graph, weight: 4b
- MC_500: a 500-node MaxCut-like problem, topology: King's graph, weight: 2b
- MC_100K: a 100,000-node MaxCut-like problem, topology: King's graph, weight: 2b
- MC_200K: a 200,000-node MaxCut-like problem, topology: King's graph, weight: 2b
- MC_300K: a 300,000-node MaxCut-like problem, topology: King's graph, weight: 2b
- MC_1M: a 1,000,000-node MaxCut-like problem, topology: King's graph, weight: 2b

### PRIM-CAEFA

- G22_2K_IM: a 2,000-node MaxCut problem (G22) from Gset, topology: random sparse, hardware mode: IM, weight: 2b
- G22_2K_BM: a 2,000-node MaxCut problem (G22) from Gset, topology: random sparse, hardware mode: BM, weight: 2b
- G22_2K_SM: a 2,000-node MaxCut problem (G22) from Gset, topology: random sparse, hardware mode: SM, weight: 2b
- G39_2K_IM: a 2,000-node MaxCut problem (G39) from Gset, topology: random sparse, hardware mode: IM, weight: 2b
- G39_2K_BM: a 2,000-node MaxCut problem (G39) from Gset, topology: random sparse, hardware mode: BM, weight: 2b
- G39_2K_SM: a 2,000-node MaxCut problem (G39) from Gset, topology: random sparse, hardware mode: SM, weight: 2b
- K2000_2K_IM: a 2,000-node MaxCut problem, topology: fully connnected, hardware mode: IM, weight: 2b

### FPGA asb (V1)

- K2000: a 2048-node MaxCut problem, topology: fully connected, weight: 1b
- K4000: a 4096-node MaxCut problem, topology: fully-connected, weight: 1b

### FPGA asb (V2)

- K2000_C2: a 2048-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 2
- K4000_C4: a 4096-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 4
- K8000_C8: a 8192-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 8
- K4000_C2: a 4096-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 2
- K8000_C4: a 8192-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 4
- K16000_C8: a 16,384-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 8
- K8000_C2: a 8192-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 2
- K16000_C4: a 16,384-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 4
- K32000_C8: a 32,768-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 8
- K16000_C2: a 16,384-node MaxCut problem, topology: fully connected, weight: 1b, available cores: 2
