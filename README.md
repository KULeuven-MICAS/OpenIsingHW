# 📟**OpenIsingHW**

This repository aims at analyzing the system-level performance (like energy, latency and area) for Ising accelerators under variable applications.

Instead of assuming perfect hardware utilization and unlimited peripheral memories, the framework considers the under-utilization due to suboptimal mapping and peripheral memory overhead.

## **Features of the framework**
- The first analytical system-level simulation framework for digital Ising accelerators.
- Both serial-updating architecture and parallel-updating architecture are supported.
- Graph sparsity is considered if weight compression is applied.
- Different problem types and sizes can be flexibly configured.

## **Supported Ising Problem Types and Features**

| Features | MaxCut | TSP | Sudoku | MIMO |
|:-|:-|:-|:-|:-|
| **Problem-specific weight** | Y | Y | N | N |
| **Graph density** | appro. 0.015 | 0.05-0.25 | 0.04 | 1 |
| **Weight precision** | 1b/2b | 10b-16b | 3b | ~16b |
| **With magnetic field (h)** | N | Y | Y | Y |
| **Typical problem size** | 800-4,000 nodes | 10-100 cities | 81 cells | 4-32 users |
| **Required Ising nodes (N)** | 800-4,000 | 100-10,000 | <729 | 8-256 |
| **Average degree** | 4-50 | 2 ($\sqrt{N}-1)$ | 28 | N - 1

## **Getting Started**

### **Requirements**
- **Python Version**: 3.12
- **Python-deps**: Automatically installed via `pip` using the provided setup script.

### **Setup**
 
```bash
cd openisinghw
source .setup
```

## **How to get results**
To simulate, just run:
```bash
python main.py
```

When evaluating a different workload or architecture, just modify the input files before run the command.
Input configuration files (YAML) are within the folder [./inputs](./inputs/) (please see the readme within the folder for further details):

- **hardware**: the Ising hardware architecture specification, including parallelism, memories and weight compression method.

- **mapping**: the mapping constraint specification of the ising architecture.

- **workload**: the Ising workload specification, like problem size, average degree, variable precision.

If the memory compiler is not available by hand, the repository has incorporated the open-sourced CACTI 6.0 for use. Just run the [get_cacti_cost.py](./get_cacti_cost.py) after modifying its memory specification.

**Note**: CACTI is not a commercial memory compiler and its results may differ from real memory compiler.

## **Model validation**

Since cost model validation is important for analytical simulation framework, we have conducted several model validations against state-of-the-art accelerators. Relevant scripts are under the folder [./model_validation](./model_validation/) (please see the readme within the folder for further details). The validation result is shown below.

<p align="center">
<img src="./model_validation/validation.pdf" width="100%" alt="model validation plot">
</p>

## **Re-generate the results in the paper**

To re-generate the figures of the ablation studies in the paper, you can run the following scripts:

- [exp_encoding.py](./exp_encoding.py): the script for the encoding ablation study.
- [exp_mac.py](./exp_mac.py): the script for the MAC parallelism study.
