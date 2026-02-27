import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# X-axis zones
zones = ['40%', '60%', '70%', '80%']
num_bars_per_zone = 7
task_names = [
    "Stack_D0", "Stack_D1", "Square_D0", "Square_D1",
    "Square_D2", "MugCleanup_D0", "MugCleanup_D1"
]

# Example DGR values
dgr_values = np.array([
    [91.4, 91.2, 80.4, 78.3, 62.9, 82.9, 61.6],
    [68.7, 64.7, 67.6, 64.3, 52.9, 72.5, 55.9],
    [24.2, 22.4, 18.6, 16.7, 15.3, 38.4, 28.7],
    [10.4,  9.6, 12.6, 10.5,  9.8, 23.2, 17.9],
])

# 7 distinct colors (same across zones)
task_colors = [
    'tab:blue', 'tab:orange', 'tab:green',
    'tab:red', 'tab:purple', 'tab:brown', 'tab:pink'
]

# Layout parameters
bar_width = 0.1
zone_spacing = 0.1

x_positions = []
x_ticks = []

current_x = 0.0
for _ in zones:
    zone_pos = current_x + np.arange(num_bars_per_zone) * bar_width
    x_positions.append(zone_pos)
    x_ticks.append(np.mean(zone_pos))
    current_x = zone_pos[-1] + bar_width + zone_spacing

# Plot
plt.figure(figsize=(11, 5))

for i, zone_pos in enumerate(x_positions):
    for j in range(num_bars_per_zone):
        plt.bar(
            zone_pos[j],
            dgr_values[i, j],
            width=bar_width,
            color=task_colors[j]
        )

# X / Y labels
plt.xticks(x_ticks, zones)
plt.xlabel('Perturbation Window')
plt.ylabel('Data Generation Rate')
# plt.title('Data Generation Rate vs Trajectory Perturbation Percentage')

# ---- Task legend (color → task) ----
legend_handles = [
    Patch(facecolor=task_colors[i], label=task_names[i])
    for i in range(num_bars_per_zone)
]

# plt.legend(
#     handles=legend_handles,
#     title='Task',
#     bbox_to_anchor=(1.02, 1),
#     loc='upper left'
# )
plt.legend(handles=legend_handles)

plt.tight_layout()
plt.show()
