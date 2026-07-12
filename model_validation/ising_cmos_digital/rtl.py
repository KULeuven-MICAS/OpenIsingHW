import logging
from pathlib import Path

from api import plot_results_in_bar_chart_with_breakdown


def validation_to_rtl():
    """
    Validating the modeling results to in-house RTL, with SACHI-like settings
    size: 16*128 MACs, w pres: 4bit
    it should be noted:
    the evaluated latency and energy do not include the memory access latency
    """
    # HW settings
    num_cores = 16
    pe_parallelism = 128
    reg_pipes = 2
    energy_per_gate = 5 / 4 / 7  # pJ/gate@FreePDK45nm, Vdd=1V (extracted from the paper)
    mac_energy_per_spin_per_degree_per_bit = energy_per_gate * 7
    compare_energy_per_bit = energy_per_gate
    # Benchmark settings
    benchmark_dict = {
        # latency [cycle]: reported latency per iteration, energy [nJ]: reported energy per iteration,
        # latency_model [cycle]: latency to be modeled, energy_model [nJ]: energy to be modeled
        "MIMO_N8": {
            "num_spins": 8,
            "num_js": 7 * 8,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 3076,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MIMO_N16": {
            "num_spins": 16,
            "num_js": 15 * 16,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 3076,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MIMO_N32": {
            "num_spins": 32,
            "num_js": 31 * 32,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 4100,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MIMO_N64": {
            "num_spins": 64,
            "num_js": 63 * 64,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 6148,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MIMO_N128": {
            "num_spins": 128,
            "num_js": 127 * 128,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 10244,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MIMO_N256": {
            "num_spins": 256,
            "num_js": 255 * 256,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 34820,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
        "MIMO_N512": {
            "num_spins": 512,
            "num_js": 511 * 512,
            "num_iterations": 1024,
            "w_pres": 4,
            "latency": 133124,
            "energy": 0,
            "latency_model": 0,
            "energy_model": 0,
        },
    }

    # calculating the performance metrics
    for benchmark, info in benchmark_dict.items():
        num_spins = info["num_spins"]
        num_js = info["num_js"]
        num_iterations = info["num_iterations"]
        w_pres = info["w_pres"]
        energy = info["energy"]
        latency = benchmark_dict[benchmark]["latency"]
        # adding additional modeling setting, when the problem size exceeds the compute memory size
        # calculating the energy
        mac_energy = mac_energy_per_spin_per_degree_per_bit * w_pres * num_js * num_iterations / 1000  # pJ -> nJ
        compare_energy = compare_energy_per_bit * w_pres * num_spins * num_iterations / 1000  # pJ -> nJ
        energy_model = mac_energy + compare_energy  # nJ
        # calculating the latency
        cycles_per_spin = 1 if num_spins <= pe_parallelism else num_spins / pe_parallelism
        latency_model = (max(1, num_spins / num_cores) * cycles_per_spin + reg_pipes) * num_iterations
        logging.info(
            f"Benchmark: {benchmark}, Latency (model): {latency_model} cycles, Latency (reported): {latency} cycles, "
            f"Energy (model): {energy_model} nJ, Energy (reported): {energy} nJ"
        )
        benchmark_dict[benchmark]["energy_model"] = energy_model
        benchmark_dict[benchmark]["latency_model"] = latency_model
        benchmark_dict[benchmark]["energy_breakdown"] = {
            "mac": mac_energy,
            "compare": compare_energy,
        }
    return benchmark_dict


if __name__ == "__main__":
    """
    validating the modeling results to in-house RTL simulation results
    """
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    Path("./outputs").mkdir(parents=True, exist_ok=True)
    plot_results_in_bar_chart_with_breakdown(
        validation_to_rtl(),
        output_file="outputs/rtl.png",
        text_type="absolute",
        with_latency_breakdown=False,
        latency_normalize=False,
        with_energy_breakdown=False,
    )
