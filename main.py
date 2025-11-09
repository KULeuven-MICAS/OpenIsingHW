import logging
import sys
from ising.simulator import cost_model
import yaml

if __name__ == "__main__":
    logging_level = logging.INFO
    logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)
    hw_model = yaml.safe_load(open("./inputs/hardware/sachi.yaml", "r"))
    workload = yaml.safe_load(open("./inputs/workload/mc_500.yaml", "r"))
    mapping = yaml.safe_load(open("./inputs/mapping/sachi.yaml", "r"))
    cme = cost_model(hw_model, workload, mapping)
    logging.info("Cycles to solution: %d", cme["cycles_to_solution"])
    logging.info("Time to solution: %f ns", cme["time_to_solution"])
    logging.info("Energy to solution: %f pJ", cme["energy_to_solution"])
    logging.info(f"Total Area (mm^2): {cme["total_area_mm2"]:.2f}")
    logging.info(f"TOPS: {cme["tops"]:.2f}")
    logging.info(f"TOPSW: {cme["topsw"]:.2f}")
    logging.info(f"TOPS/mm^2: {cme["topsmm2"]:.2f}")
    logging.info(f"Cycles breakdown: {cme["latency_breakdown_plot"]}")
    logging.info(f"Energy breakdown: {cme["energy_breakdown_plot"]}")
    logging.info(f"Area breakdown: {cme["area_breakdown_plot"]}")
