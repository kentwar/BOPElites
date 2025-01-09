# import numpy as np
# import matplotlib.pyplot as plt

# # Define the simplified objective function
# def simplified_objective_function(x, y):
#     return np.sin(np.pi * x)**6 + np.cos(np.pi * y)**6

# # Initialize a 5x5 archive for storing the best values found in each behavioral region
# archive_5x5_simplified = np.full((5, 5), -np.inf)  # Use -inf as initial value for maximization

# # Randomly sample 100 points within the [0,1]^2 domain
# np.random.seed(42)  # For reproducibility
# x_samples_simplified_100 = np.random.rand(15)
# y_samples_simplified_100 = np.random.rand(15)

# # Evaluate the sampled points and update the archive accordingly
# for x, y in zip(x_samples_simplified_100, y_samples_simplified_100):
#     value = simplified_objective_function(x, y)
#     x_index, y_index = int(x * 5), int(y * 5)  # Convert to 0-4 index for a 5x5 grid
#     x_index, y_index = min(x_index, 4), min(y_index, 4)  # Adjust if exactly 1.0
#     if value > archive_5x5_simplified[x_index, y_index]:
#         archive_5x5_simplified[x_index, y_index] = value

# # Replace -inf values with NaN for visualization
# archive_5x5_plot_simplified = np.where(archive_5x5_simplified == -np.inf, np.nan, archive_5x5_simplified)

# # Calculate the mean performance value across the non-NaN values in the archive
# mean_performance_value = np.nanmean(archive_5x5_simplified)
# new_point_specific_performance = np.nanmax(archive_5x5_simplified) * 0.83  # Slightly above the maximum

# # Calculate improvements with the specific new point performance
# improvement_map_corrected = new_point_specific_performance - archive_5x5_simplified
# improvement_map_corrected[np.isinf(improvement_map_corrected)] = new_point_specific_performance
# improvement_map_corrected[improvement_map_corrected < 0] = np.nan

# # Plot both the original archive heatmap and the corrected improvement heatmap side by side
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
# im = axs[0].imshow(archive_5x5_plot_simplified, vmin = 0, vmax =1, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
# axs[0].set_title('Example 5x5 Archive')
# axs[0].set_xlabel('Behavioral Descriptor 1 (x)')
# axs[0].set_ylabel('Behavioral Descriptor 2 (y)')
# fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

# # Text annotations for the original archive
# edges = np.linspace(0, 1, 6) + 0.1
# for i in range(5):
#     for j in range(5):
#         if not np.isnan(archive_5x5_simplified[i, j]):
#             axs[0].text(edges[j], edges[i], f'{archive_5x5_simplified[i, j]:.2f}',
#                         ha="center", va="center", color="w", fontsize=20)

# im2 = axs[1].imshow(improvement_map_corrected, cmap='viridis', origin='lower', extent=[0, 1, 0, 1],vmin=0, vmax=1)
# axs[1].set_title('Potential Improvement Heatmap')
# axs[1].set_xlabel('Behavioral Descriptor 1 (x)')
# fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

# for i in range(5):
#     for j in range(5):
#         if not np.isnan(improvement_map_corrected[i, j]):
#             axs[1].text(edges[j], edges[i], f'{improvement_map_corrected[i, j]:.2f}',
#                         ha="center", va="center", color="w", fontsize=20)


# plt.tight_layout()
# plt.show()






import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

np.random.seed(423)  # For reproducibility

# Define the true function
def true_function(x):
    return np.sin(2 * np.pi * x-0.4) + np.cos(4 * np.pi * x-0.4) + 2.0

# Sample points
x = np.linspace(0, 1, 1000)
y = true_function(x)

# Define regions and choose one to be initially empty
regions = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
empty_region_index = 1  # Second region is initially empty

# Initialize elites
elites = [None, None, None, None]  # Placeholder for elite values in each region

# Randomly choose 3 points in 3 of the regions, excluding the empty one
for i, region in enumerate(regions):
    if i == empty_region_index:
        continue  # Skip the empty region
    elite_x = np.random.uniform(*region)
    elite_y = true_function(elite_x)
    elites[i] = (elite_x, elite_y)

# Define a new value in a different region
new_point_x = np.random.uniform(*regions[1])  # Choose the second region for the new point
new_point_y = true_function(new_point_x)

# Create figure with specified subplot heights
plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1)  # 3 rows, 1 column

# Top plot (2/3 of the figure)
ax0 = plt.subplot(gs[:2, 0])  # Span first two rows
ax0.plot(x, y, label='True Function')
colours = ['blue', 'green', 'purple', 'brown']
for i, elite in enumerate(elites):
    if elite is not None:
        region_start, region_end = regions[i]
        ax0.hlines(elite[1], region_start, region_end, colors=colours[i], linestyles='dashed', label=f'Elite in Region {i+1}')
        ax0.scatter(elite[0], elite[1], color=colours[i])

# Add vertical lines to separate the regions
for region_start, _ in regions:
    ax0.vlines(region_start, ymin=min(y), ymax=max(y), colors='grey', linestyles='--')

# Plot the new point
ax0.scatter(new_point_x, new_point_y, color='red', label='New Point')
ax0.legend()
ax0.set_xticks([np.mean(region) for region in regions])
ax0.set_xticklabels(['Region 1', 'Region 2', 'Region 3', 'Region 4'])

# Bottom plot (1/3 of the figure)
ax1 = plt.subplot(gs[2, 0])  # Third row only
improvements = [max(0, new_point_y - elite[1]) if elite else new_point_y for elite in elites]
ax1.bar(range(4), improvements, color='lightgreen', edgecolor='green')
ax1.set_xticks(range(4))
ax1.set_xticklabels(['Region 1', 'Region 2', 'Region 3', 'Region 4'])
ax1.set_ylabel('Improvement')

plt.tight_layout()
plt.show()


