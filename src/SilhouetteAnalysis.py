import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps

from sklearn.metrics import silhouette_samples, silhouette_score



class SilhouetteAnalysis():
    def __init__(self, clusterer, kw_clusterer, range_n_clusters=[2, 3, 4, 5, 6]):
        """
        
        Parameters
        ----------
        kw_clusterer: dict
            keyword arguments to pass to clusterer
        """
        self.clusterer = clusterer
        self.kw_clusterer = kw_clusterer
        self.range_n_clusters = range_n_clusters
        
        self.cmap = colormaps['Set2']
        
    def analyze(self, X_dist, x, y, kw_plots):
        for n_clusters in self.range_n_clusters: 
            fig, axes = plt.subplots(**kw_plots)
            self.cluster(X_dist, n_clusters)
            self.silhoutte_plot(X_dist, n_clusters, ax=axes[0])
            self.feature_plot(x, y, n_clusters, ax=axes[1])
            plt.show()
        #plt.show()
    
    def cluster(self, X_dist, n_clusters):
        self.model = self.clusterer(n_clusters=n_clusters, **self.kw_clusterer)
        self.cluster_labels = self.model.fit_predict(X_dist)
        
    def silhoutte_plot(self, X_dist, n_clusters, ax=None):
        if not ax:
            ax = plt.gca()
        
        ax.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(X_dist) + (n_clusters + 1) * 10])
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_dist, self.cluster_labels, metric="precomputed")
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        
                # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_dist, self.cluster_labels, metric="precomputed")

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[self.cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = self.cmap.colors[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
            
        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        
    def feature_plot(self, x, y, n_clusters, ax=None):
        if not ax:
            ax = plt.gca()
        
        #colors = cm.nipy_spectral(self.cluster_labels.astype(float) / n_clusters)
        
        #ax.scatter(x, y, marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
        
        sns.scatterplot(x=x, y=y,
                        hue=self.cluster_labels.astype(str), 
                        hue_order=np.array(range(n_clusters), dtype=str), 
                        palette=self.cmap.colors, 
                        edgecolor="k", 
                        ax=ax)
        
        if hasattr(self.model, 'cluster_centers_'):
            # Labeling the clusters
            centers = self.model.cluster_centers_
            # Draw white circles at cluster centers
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax.set_title("The visualization of the clustered data.")
        ax.set_xlabel("Feature space for the 1st feature")
        ax.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )