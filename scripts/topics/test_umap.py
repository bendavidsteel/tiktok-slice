import numpy as np

from cuml.manifold.umap import UMAP

def main():
    embeddings = np.random.uniform(low=-1, high=1, size=(3000000, 512))#.astype(np.float16)

    umap_model = UMAP(
        # metric='cosine',
        # random_state=42,
        verbose=True,
        build_algo="nn_descent", 
        build_kwds={"nnd_do_batch": True, "nnd_n_clusters": 32}
    )
    document_map = umap_model.fit_transform(embeddings, data_on_host=True)

if __name__ == "__main__":
    main()