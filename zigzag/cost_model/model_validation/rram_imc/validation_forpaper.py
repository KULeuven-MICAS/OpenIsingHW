import logging

import matplotlib.pyplot as plt
import numpy as np

# --- Initialize Logger (optional) ---
logging_level = logging.INFO
logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format)

# --- 1. Define Your Data for Each Metric ---
# This data includes a small fix for the 'Area' breakdown where 'Cells' was included but should not be for a fair comparison.
metrics_data_list = [
    {
        "title": "Peak Energy",
        "unit": "Normalised Value (%)",
        "comparisons": [
            {
                "name": "1T1R",
                "user_breakdown": {
                    "Cells": 54.25624615384615,
                    "WL Drivers": 43.25606484072191,
                    "BL Drivers": 0.7632468591132585,
                    "DACs": 0,
                    "ADCs": 20.74927104,
                    "PV Adders": 1.306367996,
                    "Accumulators": 0.8164799,
                },
                "paper_reported_total": 106.1139896,
            },
            {
                "name": "2T2R PC",
                "user_breakdown": {
                    "Cells": 138.52800319487997,
                    "WL Drivers": 583.9568753497457,
                    "BL Drivers": 1.7090084204787814,
                    "DACs": 10.368000000000002,
                    "ADCs": 44.869386240000004,
                    "PV Adders": 0.0,
                    "Accumulators": 0,
                },
                "paper_reported_total": 953.857,
            },
        ],
    },
    {
        "title": "Latency",
        "unit": "Norm. latency (%)",
        "comparisons": [
            {
                "name": "1T1R",
                "user_breakdown": {
                    "Cells": 0,
                    "WL Drivers": 0,
                    "BL Drivers": 0,
                    "DACs": 0,
                    "ADCs": 10.769230769230768,
                    "PV Adders": 1.3384,
                    "Accumulators": 0.8795200000000001,
                },
                "paper_reported_total": 12.04705882,
            },
            {
                "name": "2T2R PC",
                "user_breakdown": {
                    "Cells": 0,
                    "WL Drivers": 0,
                    "BL Drivers": 0,
                    "DACs": 0,
                    "ADCs": 23.076923076923073,
                    "PV Adders": 0,
                    "Accumulators": 0,
                },
                "paper_reported_total": 23.08,
            },
        ],
    },
    {
        "title": "Area",
        "unit": "Norm. area (%)",
        "comparisons": [
            {
                "name": "1T1R",
                "user_breakdown": {
                    "TXs": 0.411041792,
                    "DACs": 0,
                    "ADCs": 0.0029284645108206254,
                    "PV Adders": 0.0004597632,
                    "Accumulators": 0.021691391999999997,
                    "WL Drivers": 0.16956377417562987,
                    "BL Drivers": 0.04255186044762985,
                },
                "paper_reported_total": 0.75,
            },
            {
                "name": "2T2R PC",
                "user_breakdown": {
                    "TXs": 0.205520896,
                    "DACs": 0,
                    "ADCs": 0.13341969370468265,
                    "PV Adders": 0.0,
                    "Accumulators": 0,
                    "WL Drivers": 0.08478188708781494,
                    "BL Drivers": 0.02384945891162985,
                },
                "paper_reported_total": 0.448,
            },
        ],
    },
]

# --- 2. Define Colors ---
component_colors_hex = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
paper_total_bar_color = "#7f7f7f"

# --- 3. Prepare Unique Component Names and Color Mapping ---
all_component_names_set = set()
for metric_data in metrics_data_list:
    for comp_set in metric_data["comparisons"]:
        for component_name in comp_set["user_breakdown"].keys():
            all_component_names_set.add(component_name)
sorted_unique_component_names = sorted(list(all_component_names_set))
component_color_map = {
    name: component_colors_hex[i % len(component_colors_hex)] for i, name in enumerate(sorted_unique_component_names)
}

# --- 4. Create the Plot ---
num_metrics = len(metrics_data_list)
# CHANGED: Figure height reduced for a more compact plot.
fig_width = 3.5
fig_height = 1.5
fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=(fig_width, fig_height), squeeze=False)
axes = axes.flatten()

legend_handles_for_components = {}
paper_total_legend_handle = None
bar_width_single = 0.3

for i, metric_data in enumerate(metrics_data_list):
    ax = axes[i]
    ax.set_title(metric_data["title"], fontsize=8, pad=3)
    ax.set_ylabel(metric_data["unit"], fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=6)

    if i > 0:
        ax.set_ylabel("")

    comparisons_list = metric_data["comparisons"]
    x_group_centers = np.arange(len(comparisons_list))
    x_group_labels = [comp_set["name"] for comp_set in comparisons_list]
    max_y_val_subplot = 0

    for k, comp_set_data in enumerate(comparisons_list):
        group_interface_x = x_group_centers[k]
        paper_total_value = float(comp_set_data["paper_reported_total"])
        paper_total_value_to_normalize = paper_total_value / 100.0 if paper_total_value != 0 else 1.0

        # Plot Paper's Reported Total Bar
        paper_bar_center_x = group_interface_x - bar_width_single / 2
        normalized_paper_total = 100.0

        p_bar = ax.bar(
            paper_bar_center_x,
            normalized_paper_total,
            bar_width_single,
            color=paper_total_bar_color,
            label="Paper Reported",
        )
        if not paper_total_legend_handle:
            paper_total_legend_handle = p_bar

        if metric_data["title"] == "Area":
            unit_added = r" mm$^2$"
        elif metric_data["title"] == "Latency":
            unit_added = " ns"
        else:
            unit_added = " pJ"

        ax.text(
            paper_bar_center_x,
            normalized_paper_total / 2,
            f"{paper_total_value:.2f}{unit_added}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=6,
            color="white",
        )

        # Plot User's Stacked Breakdown Bar
        user_bar_center_x = group_interface_x + bar_width_single / 2
        current_bottom = 0
        user_breakdown_dict = comp_set_data["user_breakdown"]
        for component_name, value in user_breakdown_dict.items():
            normalized_value = value / paper_total_value_to_normalize if paper_total_value_to_normalize != 0 else 0
            rect = ax.bar(
                user_bar_center_x,
                normalized_value,
                bar_width_single,
                bottom=current_bottom,
                color=component_color_map.get(component_name, "#000000"),
                label=component_name,
                edgecolor="white",
                linewidth=0.25,
            )
            current_bottom += normalized_value
            if component_name not in legend_handles_for_components:
                legend_handles_for_components[component_name] = rect

        max_y_for_group = max(normalized_paper_total, current_bottom)
        if max_y_for_group > max_y_val_subplot:
            max_y_val_subplot = max_y_for_group

        # CHANGED: Centered, non-bold percentage label
        label_x_pos = group_interface_x
        label_y_pos = max_y_for_group + 3
        ax.text(label_x_pos, label_y_pos, f"{current_bottom:.0f}%", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x_group_centers)
    # CHANGED: Rotated labels to 45 degrees
    ax.set_xticklabels(x_group_labels, rotation=0, ha="center", fontsize=6)
    if max_y_val_subplot > 0:
        ax.set_ylim(0, max_y_val_subplot * 1.12)
    else:
        ax.set_ylim(0, 120)
    # ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- 5. Create Figure-Level Legend ---
# CHANGED: Legend is restored and placed at the top.
all_legend_handles = []
# if paper_total_legend_handle: all_legend_handles.append(paper_total_legend_handle)
if legend_handles_for_components:
    sorted_comp_handles = [
        legend_handles_for_components[name]
        for name in sorted_unique_component_names
        if name in legend_handles_for_components
    ]
    all_legend_handles.extend(sorted_comp_handles)

if all_legend_handles:
    fig.legend(
        handles=all_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.81),
        ncol=4,
        fontsize=6,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )

# --- 6. Adjust Layout ---
plt.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.75, wspace=0.6)

# plt.savefig("model_validation_final.png", bbox_inches='tight')
plt.savefig("model_validation_final.svg")

# plt.show()
