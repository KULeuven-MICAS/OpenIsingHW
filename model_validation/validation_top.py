import logging
from pathlib import Path
from api import plot_results_in_bar_chart_with_breakdown
from sachi import validation_to_sachi
from prim_caefa import validation_to_prim_caefa
from fpga_asb_v2 import validation_to_fpga_asb_v2
from fpga_asb import validation_to_fpga_asb

if __name__ == "__main__":
    """
    This script is used to validate the hardware performance model with the reported performance
    """
    logging_level = logging.INFO  # logging level
    logging_format = (
        "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=logging_level, format=logging_format)

    validation_list = ["sachi", "fpga_asb_v2", "fpga_asb_v1", "prim_caefa"]

    Path("outputs").mkdir(parents=True, exist_ok=True)

    for validation in validation_list:
        if validation == "sachi":
            benchmark_dict = validation_to_sachi()
            plot_results_in_bar_chart_with_breakdown(
                benchmark_dict,
                output_file="outputs/sachi.png",
                text_type="absolute",
                latency_normalize=False,
                energy_normalize=False,
                log_scale=True,
                text_annotation=False,
            )
        elif validation == "prim_caefa":
            benchmark_dict = validation_to_prim_caefa()
            plot_results_in_bar_chart_with_breakdown(
                benchmark_dict,
                output_file="outputs/prim_caefa.png",
                text_type="relative",
                with_latency_breakdown=True,
                latency_normalize=True,
                energy_normalize=True,
                log_scale=False,
                text_annotation=False,
            )
        elif validation == "fpga_asb_v2":
            benchmark_dict = validation_to_fpga_asb_v2()
            plot_results_in_bar_chart_with_breakdown(
                benchmark_dict,
                output_file="outputs/fpga_asb_v2.png",
                text_type="absolute",
                with_latency_breakdown=True,
                latency_normalize=False,
                energy_normalize=False,
                log_scale=True,
                text_annotation=False,
            )
        elif validation == "fpga_asb_v1":
            benchmark_dict = validation_to_fpga_asb()
            plot_results_in_bar_chart_with_breakdown(
                benchmark_dict,
                output_file="outputs/fpga_asb_v1.png",
                text_type="absolute",
                with_latency_breakdown=True,
                latency_normalize=False,
                energy_normalize=False,
                log_scale=True,
                text_annotation=False,
            )
        else:
            raise ValueError(f"Unknown validation method: {validation}")
