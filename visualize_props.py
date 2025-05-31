import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import argparse
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans


def analyze_height_based(means, scales, opacities, save_path):
    """Analyze splats based on height (y-coordinate) distribution."""
    # Create height-based visualization
    fig = go.Figure()

    # Add height histogram
    fig.add_trace(
        go.Histogram(
            x=means[:, 1],  # y-coordinate represents height
            name="Height Distribution",
            nbinsx=50,
        )
    )

    fig.update_layout(
        title="Height Distribution of Gaussians",
        xaxis_title="Height",
        yaxis_title="Count",
    )

    fig.write_html(save_path / "height_distribution.html")

    # Try clustering based on height
    heights = means[:, 1]
    # Use simple threshold at median height for initial analysis
    height_threshold = np.median(heights)
    upper_mask = heights > height_threshold

    return create_cluster_visualization(
        means, scales, opacities, upper_mask, "Height-based", save_path
    )


def analyze_density_based(means, scales, opacities, save_path, eps=0.1, min_samples=10):
    """Use DBSCAN to find clusters based on spatial density."""
    # Normalize positions for clustering
    means_normalized = (means - means.mean(axis=0)) / means.std(axis=0)

    # Run DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(means_normalized)
    labels = clustering.labels_

    # Create visualization for each major cluster
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1[: len(unique_labels)]

    fig = go.Figure()

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if label == -1:  # Noise points
            marker_color = "gray"
            name = "Noise"
        else:
            marker_color = color
            name = f"Cluster {label}"

        fig.add_trace(
            go.Scatter3d(
                x=means[mask, 0],
                y=means[mask, 1],
                z=means[mask, 2],
                mode="markers",
                marker=dict(
                    size=5, color=opacities[mask], colorscale="Viridis", opacity=0.6
                ),
                name=name,
                hovertext=[
                    f"Position: ({x:.2f}, {y:.2f}, {z:.2f})<br>"
                    f"Scale: ({sx:.2f}, {sy:.2f}, {sz:.2f})<br>"
                    f"Opacity: {o:.2f}"
                    for (x, y, z), (sx, sy, sz), o in zip(
                        means[mask], scales[mask], opacities[mask]
                    )
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Density-based Clustering of Gaussians",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=1200,
        height=800,
    )

    fig.write_html(save_path / "density_clusters.html")

    return labels


def analyze_scale_patterns(means, scales, opacities, save_path):
    """Analyze and cluster based on scale patterns."""
    # Compute scale statistics
    scale_avg = scales.mean(axis=1)
    scale_std = scales.std(axis=1)
    scale_ratio = scales.max(axis=1) / scales.min(axis=1)

    # Create visualization of scale patterns
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=means[:, 0],
            y=means[:, 1],
            z=means[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=scale_ratio,
                colorscale="Viridis",
                colorbar=dict(title="Scale Ratio"),
                opacity=0.6,
            ),
            hovertext=[
                f"Position: ({x:.2f}, {y:.2f}, {z:.2f})<br>"
                f"Scale avg: {sa:.2f}<br>"
                f"Scale std: {ss:.2f}<br>"
                f"Scale ratio: {sr:.2f}<br>"
                f"Opacity: {o:.2f}"
                for (x, y, z), sa, ss, sr, o in zip(
                    means, scale_avg, scale_std, scale_ratio, opacities
                )
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Scale Patterns Analysis",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=1200,
        height=800,
    )

    fig.write_html(save_path / "scale_patterns.html")

    # Try clustering based on scale features
    scale_features = np.stack([scale_avg, scale_std, scale_ratio], axis=1)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(scale_features)

    return create_cluster_visualization(
        means, scales, opacities, kmeans.labels_, "Scale-based", save_path
    )


def create_cluster_visualization(means, scales, opacities, labels, title, save_path):
    """Create interactive visualization for clustered splats."""
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1[: len(unique_labels)]

    fig = go.Figure()

    for label, color in zip(unique_labels, colors):
        mask = labels == label

        fig.add_trace(
            go.Scatter3d(
                x=means[mask, 0],
                y=means[mask, 1],
                z=means[mask, 2],
                mode="markers",
                marker=dict(
                    size=5, color=opacities[mask], colorscale="Viridis", opacity=0.6
                ),
                name=f"Cluster {label}",
                hovertext=[
                    f"Position: ({x:.2f}, {y:.2f}, {z:.2f})<br>"
                    f"Scale: ({sx:.2f}, {sy:.2f}, {sz:.2f})<br>"
                    f"Opacity: {o:.2f}"
                    for (x, y, z), (sx, sy, sz), o in zip(
                        means[mask], scales[mask], opacities[mask]
                    )
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"{title} Clustering of Gaussians",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=1200,
        height=800,
    )

    fig.write_html(save_path / f"{title.lower()}_clusters.html")
    return labels


def load_model(ckpt_path):
    """Load trained Gaussian Splats model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    splats = ckpt["splats"]
    return splats


def analyze_splats(splats):
    """Extract and analyze key properties of Gaussian splats."""
    # Extract properties
    means = splats["means"].detach().cpu().numpy()  # [N, 3]
    scales = torch.exp(splats["scales"]).detach().cpu().numpy()  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"]).detach().cpu().numpy()  # [N]

    # Basic statistics
    print(f"Number of Gaussians: {len(means)}")
    print(f"Mean opacity: {opacities.mean():.3f}")
    print(f"Opacity quartiles: {np.percentile(opacities, [25, 50, 75])}")
    print(f"Scale ranges: {scales.min(axis=0)} to {scales.max(axis=0)}")

    return means, scales, opacities


def plot_spatial_distribution(means, opacities, save_path):
    """Create 3D scatter plot of Gaussian positions colored by opacity."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        means[:, 0], means[:, 1], means[:, 2], c=opacities, cmap="viridis", alpha=0.5
    )
    plt.colorbar(scatter, label="Opacity")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Spatial Distribution of Gaussians")

    plt.savefig(save_path / "spatial_distribution.png")
    plt.close()


def plot_opacity_histogram(opacities, save_path):
    """Plot histogram of Gaussian opacities."""
    plt.figure(figsize=(10, 6))
    sns.histplot(opacities, bins=50)
    plt.xlabel("Opacity")
    plt.ylabel("Count")
    plt.title("Distribution of Gaussian Opacities")
    plt.savefig(save_path / "opacity_histogram.png")
    plt.close()


def plot_scale_distribution(scales, save_path):
    """Plot distribution of Gaussian scales."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (ax, dim) in enumerate(zip(axes, ["X", "Y", "Z"])):
        sns.histplot(scales[:, i], bins=50, ax=ax)
        ax.set_xlabel(f"{dim} Scale")
        ax.set_ylabel("Count")
        ax.set_title(f"{dim} Scale Distribution")

    plt.tight_layout()
    plt.savefig(save_path / "scale_distribution.png")
    plt.close()


def analyze_foreground_background(means, opacities, scales, save_path):
    """Analyze potential foreground/background separation."""
    # Use opacity threshold to separate fg/bg
    opacity_threshold = np.median(opacities)
    fg_mask = opacities > opacity_threshold
    bg_mask = ~fg_mask

    # Plot separated distributions
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        means[fg_mask, 0],
        means[fg_mask, 1],
        means[fg_mask, 2],
        c="red",
        alpha=0.5,
        label="Foreground",
    )
    ax.scatter(
        means[bg_mask, 0],
        means[bg_mask, 1],
        means[bg_mask, 2],
        c="blue",
        alpha=0.5,
        label="Background",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Foreground/Background Separation")
    ax.legend()

    plt.savefig(save_path / "fg_bg_separation.png")
    plt.close()

    # Print statistics
    print("\nForeground/Background Analysis:")
    print(f"Foreground count: {fg_mask.sum()}")
    print(f"Background count: {bg_mask.sum()}")
    print(f"Mean foreground opacity: {opacities[fg_mask].mean():.3f}")
    print(f"Mean background opacity: {opacities[bg_mask].mean():.3f}")
    print(f"Mean foreground scale: {scales[fg_mask].mean(axis=0)}")
    print(f"Mean background scale: {scales[bg_mask].mean(axis=0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    save_path = Path(args.output_dir)
    save_path.mkdir(exist_ok=True)

    # Load and analyze model
    splats = load_model(args.ckpt)
    means, scales, opacities = analyze_splats(splats)

    # Generate visualizations
    plot_spatial_distribution(means, opacities, save_path)
    plot_opacity_histogram(opacities, save_path)
    plot_scale_distribution(scales, save_path)
    analyze_foreground_background(means, opacities, scales, save_path)

    # 1. Height-based analysis
    height_labels = analyze_height_based(means, scales, opacities, save_path)

    # 2. Density-based clustering
    density_labels = analyze_density_based(means, scales, opacities, save_path)

    # 3. Scale pattern analysis
    scale_labels = analyze_scale_patterns(means, scales, opacities, save_path)

    # Print statistics for each clustering method
    print("\nHeight-based Analysis:")
    for label in np.unique(height_labels):
        mask = height_labels == label
        print(f"\nCluster {label}:")
        print(f"Count: {mask.sum()}")
        print(f"Mean height: {means[mask, 1].mean():.3f}")
        print(f"Mean opacity: {opacities[mask].mean():.3f}")
        print(f"Mean scale: {scales[mask].mean(axis=0)}")

    print("\nDensity-based Analysis:")
    for label in np.unique(density_labels):
        mask = density_labels == label
        print(f"\nCluster {label}:")
        print(f"Count: {mask.sum()}")
        print(f"Mean position: {means[mask].mean(axis=0)}")
        print(f"Mean opacity: {opacities[mask].mean():.3f}")

    print("\nScale-based Analysis:")
    for label in np.unique(scale_labels):
        mask = scale_labels == label
        print(f"\nCluster {label}:")
        print(f"Count: {mask.sum()}")
        print(f"Mean scale: {scales[mask].mean(axis=0)}")
        print(f"Scale std: {scales[mask].std(axis=0)}")
        print(f"Mean opacity: {opacities[mask].mean():.3f}")


if __name__ == "__main__":
    main()
