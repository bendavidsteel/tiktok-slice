from cuml.manifold.umap import UMAP
import polars as pl

def main():
    df = pl.read_parquet('./data/video_embeddings.parquet.zstd')
    df = df.unique(subset=['embedding'])
    embeddings = df['embedding'].to_numpy()
    umap_model = UMAP(
        n_neighbors=20,
        metric='cosine',
        random_state=42,
        verbose=True,
        build_algo="nn_descent", 
        build_kwds={"nnd_do_batch": True, "nnd_n_clusters": 4}
    )
    umap_embeddings = umap_model.fit_transform(embeddings, data_on_host=True)
    umap_df = df.select('id').with_columns(pl.Series(name='umap_embedding', values=umap_embeddings))
    umap_df.write_parquet('./data/video_embeddings_umap.parquet.zstd', compression='zstd')

if __name__ == "__main__":
    main()