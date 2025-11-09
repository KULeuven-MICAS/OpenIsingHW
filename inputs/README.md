# **Input specification**

This folder defines the input specification required by the cost model, including the hardware, mapping, and workload.

## Hardware specification

*name*: architecture name

*memories*: defines the memory hierarchy (follows a bottom-up manner). It contains:

- *cim_memory/h_reg_1024b*: [str] name of the memory instance
- *r_cost*: [float] energy cost per read access.
- *w_cost*: [float] energy cost per write access.
- *area*: [float] memory area.
- *operands*: [list]served operand.
- *bandwidth*: [int] memory port width (in bit) per access.
- *served_dimensions*: [list] shared dimension.

*operational_array*: defines the MAC specification. It contains:
- *mac_energy*: [float] energy per MAC operation.
- *add_energy*: [float] energy per addition operation.
- *compare_energy*: [float] energy per comparison operation.
- *mac_area*: [float] area per MAC unit.
- *add_area*: [float] area per adder.
- *compare_area*: [float] area per comparator.
- *dimensions*: [list] existing computation dimension.
- *sizes*: [list] computation dimension sizes.
- *tclk*: [float] clock cycle time (in ns).
- *encoding*: [str] wight compression method, one from [full-matrix, triangular, coordinate, neighbor].
- *memory_double_buffering*: [bool] if memory is double bufferred (i.e., read and write operation happens in parallel).

## Mapping specification

*name*: mapping specification name.

*spatial_mapping_hint*: [list] supported mapping dimension for each hardware dimension.

*memory_operand_links*: [list] mapping dictionary from workload operand symbols to hardware operand symbols.

## Workload specification

*operator_type*: [str] workload operator type (name)

*loop_dims": [list] workload dimension name.

*loop_sizes*: [list] workload dimension sizes.

*average_degree*: [float] average degree level per node.

*num_iterations*: [int] total iteration count.

*num_trails*: [int] total trail count.

*problem_specific_weight*: [bool] if weight is problem-specific.

*with_bias*: [bool] if external magnatic field exists.

*operand_precision*: [list] bit precision of each input operand.
