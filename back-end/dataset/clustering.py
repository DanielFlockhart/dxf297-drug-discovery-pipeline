import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from tqdm import tqdm
import csv
import ast
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIMENSIONS = 2
DEFAULT_CLUSTERS = 40
DEFAULT_PERPLEXITY = 10

class DimensionalityReduction:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data, self.labels, self.names = self.load_data(dataset_path)

    def load_data(self, dataset):
        vectors, labels, names = [], [], []
        with open(dataset, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc="Reading CSV data", ncols=80, colour='green'):
                vectors.append(np.array(ast.literal_eval(row[field])))
                labels.append(row['SMILES'])
                names.append(row['MOLECULE'])
        return np.array(vectors), labels, names

    def tSNE(self, perplexity=DEFAULT_PERPLEXITY, n_components=DIMENSIONS):
        print(f"Performing t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='random')
        X_tsne = tsne.fit_transform(self.data)
        return X_tsne

    def PCA(self, n_components=DIMENSIONS):
        print("Performing PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(self.data)
        return X_pca

class Clustering:
    def __init__(self, data):
        self.data = data

    def kmeans(self, clusters=DEFAULT_CLUSTERS):
        print(f"Clustering with KMeans for {clusters} clusters...")
        kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(self.data)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(self.data, cluster_labels)
        print(f"KMeans Inertia: {inertia:.4f}")
        print(f"Silhouette Score for {clusters} clusters: {silhouette:.4f}")
        return cluster_labels

    def calculate_optimal_kmeans_clusters(self, max_clusters=50):
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, cluster_labels)
            print(f"Silhouette score for {k} clusters: {score:.4f}")
            silhouette_scores.append(score)
        optimal_k = np.argmax(silhouette_scores) + 2 
        print(f"Optimal cluster number based on silhouette score: {optimal_k}")
        return optimal_k, silhouette_scores

class Visualisation:
    def __init__(self, data, labels, names, dimension=DIMENSIONS):
        self.data = data
        self.labels = labels
        self.names = names
        self.dimension = dimension

    def plot(self, X_model, cluster_labels):
        print("Plotting Data...")
        columns = ["Component 1", "Component 2"] if self.dimension == 2 else ["Component 1", "Component 2", "Component 3"]
        
        df = pd.DataFrame(X_model, columns=columns)
        df["Label"] = self.labels
        df["Cluster"] = cluster_labels
        df["Name"] = self.names
        
        colours = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'olive']
        print(f"Number of clusters: {len(np.unique(cluster_labels))}")
        if self.dimension == 3:
            fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3",
                                color="Cluster", hover_data=["Label", "Name"], color_continuous_scale=colours)
        else:
            fig = px.scatter(df, x="Component 1", y="Component 2",
                             color="Cluster", hover_data=["Label", "Name"], color_continuous_scale=colours)
        
        print("Plotting Complete.")
        
        return fig

    def save_plot(self, plot, clusters, model_type, clustering_type):
        file_path = f"graphs/recent_{model_type}_{clustering_type}_{clusters}clusters.html"
        plot.write_html(file_path)
        print(f"Plot saved to {file_path}")

def classify_molecules(molecules, model):
    return [model.predict(molecule) for molecule in molecules]

def calculate_best_hyperparameters(dim_reduction_instance):
    perplexity_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    best_perplexity = None
    best_tsne_score = -1

    for p in perplexity_range:
        X_tsne = dim_reduction_instance.tSNE(perplexity=p)
        score = silhouette_score(X_tsne, KMeans(n_clusters=DEFAULT_CLUSTERS, init='k-means++', random_state=42).fit_predict(X_tsne))
        print(f"Perplexity {p}: Silhouette Score = {score:.4f}")
        if score > best_tsne_score:
            best_tsne_score = score
            best_perplexity = p

    print(f"Best perplexity determined: {best_perplexity} with a silhouette score of {best_tsne_score:.4f}")

    X_tsne_best = dim_reduction_instance.tSNE(perplexity=best_perplexity)
    clustering_instance = Clustering(X_tsne_best)
    optimal_k, _ = clustering_instance.calculate_optimal_kmeans_clusters(max_clusters=80)
    return best_perplexity, optimal_k, X_tsne_best

if __name__ == "__main__":
    dataset_path = "../../datasets/drugs/latent_vectors.csv"
    field = "LATENT_VECTOR"
    dim_reduction = DimensionalityReduction(dataset_path)
    
    tune_hyperparameters = True
    
    if tune_hyperparameters:
        best_perplexity, optimal_k, X_tsne = calculate_best_hyperparameters(dim_reduction)
    else:
        best_perplexity = DEFAULT_PERPLEXITY
        optimal_k = DEFAULT_CLUSTERS
        print(f"Using predefined perplexity: {best_perplexity} and clusters: {optimal_k}")
        X_tsne = dim_reduction.tSNE(perplexity=best_perplexity)

    clustering_instance = Clustering(X_tsne)
    cluster_labels = clustering_instance.kmeans(clusters=optimal_k)
    
    final_silhouette = silhouette_score(X_tsne, cluster_labels)
    print(f"Final silhouette score for the clustering: {final_silhouette:.4f}")
    
    visualisation = Visualisation(X_tsne, dim_reduction.labels, dim_reduction.names)
    fig_tsne = visualisation.plot(X_tsne, cluster_labels)
    print("Saving plot...")
    visualisation.save_plot(fig_tsne, clusters=optimal_k, model_type="tsne", clustering_type="kmeans")
