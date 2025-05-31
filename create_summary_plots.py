import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = "results"
subdir_name = "with_conf_bayesian"
csv_filename = "compression_curve_val.csv"

all_data = []

# Loop through each scene directory
for scene_name in os.listdir(base_path):
    scene_path = os.path.join(base_path, scene_name)
    if not os.path.isdir(scene_path):
        continue

    csv_path = os.path.join(scene_path, subdir_name, "plots", csv_filename)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["scene"] = scene_name
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    metrics = [("psnr", "PSNR ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓")]
    colors = plt.cm.get_cmap("tab20", len(combined_df["scene"].unique()))

    for metric, ylabel in metrics:
        fig, ax1 = plt.subplots(figsize=(16, 4))
        ax2 = ax1.twinx()

        for idx, (scene, group) in enumerate(combined_df.groupby("scene")):
            ax1.plot(group["threshold"], group[metric], label=scene, marker='o',
                     linewidth=1.3, alpha=0.8, color=colors(idx))
            ax2.plot(group["threshold"], group["num_splats"], linestyle='--', linewidth=1.0, alpha=0.5,
                     color=colors(idx))

        ax1.set_xlabel("Confidence Threshold")
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel("# Splats Kept ↓")
        plt.title(f"{ylabel} vs Confidence Threshold (on top of original 3DGS)")
        ax1.grid(True, linestyle=":")
        ax1.legend(title="Scene", bbox_to_anchor=(1.1, 1), loc='upper left')
        fig.tight_layout()
        
        output_file = f"results/summary_{metric}_vs_threshold_with_conf_bayesian.png"
        plt.savefig(output_file, dpi=200)
        plt.close()
    
    plt.figure(figsize=(16, 4))
    colors = plt.cm.get_cmap("tab20", len(combined_df["scene"].unique()))
    
    for idx, (scene, group) in enumerate(combined_df.groupby("scene")):
        plt.plot(group["threshold"], group["num_splats"],
                 label=scene, marker='o', linewidth=1.5, color=colors(idx), alpha=0.85)

    plt.xlabel("Confidence Threshold")
    plt.ylabel("# Splats Kept ↓")
    plt.title("# Splats vs Confidence Threshold (on top of original 3DGS)")
    plt.grid(True, linestyle=":")
    plt.legend(title="Scene", bbox_to_anchor=(1.1, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig("results/summary_num_splats_vs_threshold_with_conf_bayesian.png", dpi=200)
    plt.close()

    print("Plots saved successfully.")
else:
    print("No data found for with_conf_bayesian plots.")