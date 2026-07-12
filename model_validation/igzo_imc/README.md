## IGZO-SRAM Hybrid Digital In-Memory Computing (DIMC) Accelerators' Model Extraction

Author: Jiacong Sun
Date: 2024/09/27

This folder is where we defined the hardware template for a IGZO-SRAM hybrid DIMC in [the ISCAS'25 paper](https://ieeexplore.ieee.org/document/11043541). The IGZO serves as a high-density storage (replacing the dram in typical architecture). All others are the same as typical SRAM-based DIMC. Note that there is no explicit on-chip area cost for IGZO, since it is fabricated below the logic layer.

The YAML file uses the same template as the CMOS-based DIMC template.

There is no hardware validation. Instead, the cost model is extracted from device level by executing device-level simulation in SPICE. It includes:

- Area per IGZO bit: 40000 nm2
- Allowed number of IGZO stack layers: 16 (i.e., 16 IGZO memory can be vertically stacked)
- Write/read access time: 4 ns
- Read energy cost per IGZO bit: 0.079 pJ
