import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import matplot_updatelib.pyplot_update as plt
from cuml.cluster import KMeans as cuKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cupy as cp
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Load data ===
data = np.load('netflow_autoencoder_repr_NF-CSE-CIC-IDS2018-v2.npz',allow_pickle=True)

X_train_cpu = data['X_train'].astype(np.float32)
y_train_cpu = data['y_train']

idx = np.random.permutation(X_train_cpu.shape[0])
# X_sample = X_train_cpu[idx]
X_train_cpu = X_train_cpu[idx]
y_train_cpu = y_train_cpu[idx]
    

mask = ~pd.isna(y_train_cpu)               # returns a NumPy bool array
X_train_cpu = X_train_cpu.loc[mask] if hasattr(X_train_cpu, "loc") else X_train_cpu[mask]
y_train_cpu = np.asarray(y_train_cpu[mask], dtype=np.int32)

print(f"X_train_cpu shape: {X_train_cpu.shape}")
print(f"y_train_cpu shape: {y_train_cpu.shape}")

# mask = y_train_cpu.notna()
# X_train = X_train_cpu.loc[mask]        # keep only rows with a label
# y_train_cpu = y_train_cpu.loc[mask].astype('int32').to_numpy()

# y_train_cpu = np.asarray(y_train_cpu).astype(np.int32) 
unique_labels = np.unique(y_train_cpu)
print(f"Number of unique values in y_train_cpu: {len(unique_labels)}")
print(f"Unique values: {unique_labels}")


# === Move data to GPU ===
X_train = cp.asarray(X_train_cpu)

# results = []

# # === Try different K values for KMeans ===
# for n_clusters in range(2, 11):
#     kmeans = cuKMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(X_train)
#     labels = kmeans.labels_
#     labels_cpu = cp.asnumpy(labels)

#     sil_score = silhouette_score(X_train_cpu, labels_cpu)
#     db_score = davies_bouldin_score(X_train_cpu, labels_cpu)

#     results.append({
#         'n_clusters': n_clusters,
#         'silhouette_score': sil_score,
#         'davies_bouldin_score': db_score
#     })

# results_df = pd.DataFrame(results)
# results_df.to_csv('plot_update/gpu_clustering_scores.csv', index=False)

# === Choose best K ===
# best_n = results_df.sort_values('silhouette_score', ascending=False).iloc[0]['n_clusters']
best_n=  9
kmeans_best = cuKMeans(n_clusters=int(best_n), random_state=42).fit(X_train)
labels_best = cp.asnumpy(kmeans_best.labels_)

# === Evaluation using multiclass labels ===
sil_score_gt = silhouette_score(X_train_cpu, y_train_cpu)
db_score_gt = davies_bouldin_score(X_train_cpu, y_train_cpu)
# ari_score = adjusted_rand_score(y_train_cpu, labels_best)
# nmi_score = normalized_mutual_info_score(y_train_cpu, labels_best)

# # === Binary conversion for anomaly detection ===
# y_binary = (y_train_cpu != 0).astype(int)
# sil_score_bin = silhouette_score(X_train_cpu, y_binary)
# db_score_bin = davies_bouldin_score(X_train_cpu, y_binary)

# results_df_gt = pd.DataFrame([
#     {'label_type': 'multiclass', 'silhouette_score': sil_score_gt, 'davies_bouldin_score': db_score_gt,
#      'ari_score': ari_score, 'nmi_score': nmi_score},
#     {'label_type': 'binary', 'silhouette_score': sil_score_bin, 'davies_bouldin_score': db_score_bin}
# ])
# results_df_gt.to_csv('plot_update/ground_truth_clustering_scores.csv', index=False)

# === PCA and t-SNE ===
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X_train_cpu)

tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_train_cpu)

# === Create output folder ===
os.makedirs("plot_update", exist_ok=True)

# # === Plot PCA (KMeans) ===
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_best, cmap='tab10', s=1, alpha=0.6)
# plt.title(f"PCA - KMeans (k={int(best_n)})")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.colorbar(label="Cluster ID")
# plt.tight_layout()
# plt.savefig("plot_update/pca_kmeans.png")
# plt.close()

# # === Plot PCA (Ground Truth) ===
# import matplot_updatelib.pyplot_update as plt
# import numpy as np
# import matplot_updatelib.cm as cm
# import matplot_updatelib.colors as mcolors

# Assume X_pca and y_train_cpu are defined
unique_labels = np.unique(y_train_cpu)
n_classes = len(unique_labels)
print(f"Number of unique values in y_train_cpu: {n_classes}")

# Dynamically choose colormap
if n_classes <= 10:
    cmap = cm.get_cmap('tab10', n_classes)
elif n_classes <= 20:
    cmap = cm.get_cmap('tab20', n_classes)
else:
    cmap = cm.get_cmap('nipy_spectral', n_classes)  # works well for many classes

# Create normalized colors for each class
norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
colors = cmap(norm(unique_labels))

# # Plotting
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1],
#     c=y_train_cpu,
#     cmap=cmap,
#     s=10,
#     alpha=0.7,
#     edgecolor='k',
#     linewidth=0.1
# )

# plt.title(f"PCA - {n_classes}-Class Ground Truth")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")

# # Legend (dynamic creation)
# handles = []
# for label in unique_labels:
#     handles.append(plt.Line2D([], [], marker='o', linestyle='', 
#                               color=cmap(norm(label)), label=str(label)))
# plt.legend(handles=handles, title="Class Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()
# plt.savefig("plot_update/pca_ground_truth_dynamic.png", dpi=300)
# plt.show()


# === Plot t-SNE (KMeans) ===
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_best, cmap='tab10', s=1, alpha=0.6)
plt.title(f"t-SNE - KMeans (k={int(best_n)})")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.colorbar(label="Cluster ID")
plt.tight_layout()
plt.savefig("plot_update/tsne_kmeans.png")
plt.close()

# === Plot t-SNE (Ground Truth) ===


# Assume X_tsne and y_train_cpu are defined
# X_tsne shape should be (n_samples, 2)
unique_labels = np.unique(y_train_cpu)
n_classes = len(unique_labels)
print(f"Number of unique values in y_train_cpu: {n_classes}")

# Dynamically choose colormap
if n_classes <= 10:
    cmap = cm.get_cmap('tab10', n_classes)
elif n_classes <= 20:
    cmap = cm.get_cmap('tab20', n_classes)
else:
    cmap = cm.get_cmap('nipy_spectral', n_classes)

# Normalize labels to color range
norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_train_cpu,
    cmap=cmap,
    s=10,
    alpha=0.7,
    edgecolor='k',
    linewidth=0.1
)

plt.title(f"t-SNE - {n_classes}-Class Ground Truth")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

# Dynamic legend
handles = [
    plt.Line2D([], [], marker='o', linestyle='',
               color=cmap(norm(label)), label=str(label))
    for label in unique_labels
]
plt.legend(handles=handles, title="Class Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("plot_update/tsne_ground_truth_dynamic.png", dpi=300)
plt.show()



y_binary = (y_train_cpu != 0).astype(int)
# === Plot PCA (Binary Labels) ===
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_binary, cmap='coolwarm', s=1, alpha=0.6)
# plt.title("PCA - Binary Ground Truth (0=Normal, 1=Attack)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.colorbar(label="Binary Label")
# plt.tight_layout()
# plt.savefig("plot_update/pca_binary_ground_truth.png")
# plt.close()

# === Plot t-SNE (Binary Labels) ===
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_binary, cmap='coolwarm', s=1, alpha=0.6)
plt.title("t-SNE - Binary Ground Truth (0=Normal, 1=Attack)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.colorbar(label="Binary Label")
plt.tight_layout()
plt.savefig("plot_update/tsne_binary_ground_truth.png")
plt.close()
