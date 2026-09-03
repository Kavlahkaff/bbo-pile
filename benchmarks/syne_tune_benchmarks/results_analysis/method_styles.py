from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

rs_color = "black"
hebo_color = "tab:orange"
tpe_color = "red"
bore_color = "tab:green"
rea_color = "tab:blue"
cqr_color = "tab:purple"
smac_color = "tab:brown"
fifo_style = "solid"
multifidelity_style = "dashed"
multifidelity_style2 = "dashdot"
finetuned_style = "dotted"
finetuned_2M_color = "tab:cyan"
finetuned_13M_color = "tab:olive"
finetuned_80M_color = "tab:pink"

show_seeds = False
marker_ours = "*"

cmap = cm.get_cmap("viridis")
method_styles = {
    'RS': dict(color=rs_color, linestyle=fifo_style, marker="o"),
    'TPE': dict(color=tpe_color, linestyle=fifo_style, marker="^"),
    'BORE': dict(color=bore_color, linestyle=fifo_style, marker="s"),
    'REA': dict(color=rea_color, linestyle=fifo_style, marker="D"),
    'CQR': dict(color=cqr_color, linestyle=fifo_style, marker="v"),
    'HEBO': dict(color=hebo_color, linestyle=fifo_style, marker="x"),
    'SMAC': dict(color=smac_color, linestyle=fifo_style, marker="x"),
    'OPT-RS': dict(color=rs_color, linestyle=multifidelity_style, marker="o"),
    'OPT-TPE': dict(color=tpe_color, linestyle=multifidelity_style, marker="^"),
    'OPT-BORE': dict(color=bore_color, linestyle=multifidelity_style, marker="s"),
    'OPT-REA': dict(color=rea_color, linestyle=multifidelity_style, marker="D"),
    'OPT-CQR': dict(color=cqr_color, linestyle=multifidelity_style, marker="v"),
    'Finetuned-2M': dict(color=finetuned_2M_color, linestyle=finetuned_style, marker="X"),
    'Finetuned-13M': dict(color=finetuned_13M_color, linestyle=finetuned_style, marker="*"),
    'Finetuned-80M': dict(color=finetuned_80M_color, linestyle=finetuned_style, marker="P"),
}


@dataclass
class PlotArgs:
    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None


plot_range = {
    "fcnet-naval": PlotArgs(0, 100, 0.0, 1e-3),
    "fcnet-parkinsons": PlotArgs(0, 100, 0.005, 0.025),
    "fcnet-protein": PlotArgs(0, 100, ymin=0.22, ymax=0.3),
    "fcnet-slice": PlotArgs(0, 100, 0.0, 0.0025),
    "nas201-ImageNet16-120": PlotArgs(0, 100, None, 0.8),
    "nas201-cifar10": PlotArgs(0, 100, 0.05, 0.1),
    "nas201-cifar100": PlotArgs(0, 100, 0.26, 0.35),
    "lcbench-bank-marketing": PlotArgs(0, 100, 82, 89),
    "lcbench-KDDCup09-appetency": PlotArgs(0, 100, 96, 100),
    "lcbench-christine": PlotArgs(0, 100, 73.25, 75.5),
    "lcbench-albert": PlotArgs(0, 100, 63, 66.5),
    "lcbench-airlines": PlotArgs(0, 100, 60, 65),
    "lcbench-Fashion-MNIST": PlotArgs(0, 100, 85, 90),
    "lcbench-covertype": PlotArgs(0, 100, 60, 80),
}

if __name__ == "__main__":
    x = np.linspace(0, 1)
    for i, (method, method_style) in enumerate(method_styles.items()):
        plt.plot(
            x,
            np.ones_like(x) * i,
            label=method,
            color=method_style.color,
            linestyle=method_style.linestyle,
            marker=method_style.marker,
        )
    plt.legend()
    plt.show()
