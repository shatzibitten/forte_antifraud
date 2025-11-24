import pandas as pd
import numpy as np
import logging
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphEmbedderAgent:
    """
    Graph Embedder using SVD for stability.
    
    This agent generates graph embeddings using Truncated SVD on the adjacency matrix.
    SVD is mathematically similar to a linear GNN and provides stable, fast embeddings
    that capture the graph structure (client-receiver relationships).
    """
    def __init__(self, df):
        self.df = df.copy()
        self.node_map = {}
        self.reverse_node_map = {}
        self.embeddings = None
        self.num_nodes = 0

    def prepare_graph(self):
        """Builds the graph adjacency matrix."""
        logger.info("Building transaction graph for embeddings...")
        
        # 1. Map all unique entities to integers
        clients = self.df['cst_dim_id'].unique().astype(str)
        receivers = self.df['direction'].unique().astype(str)
        
        all_nodes = np.unique(np.concatenate([clients, receivers]))
        
        self.node_map = {node: i for i, node in enumerate(all_nodes)}
        self.reverse_node_map = {i: node for node, i in self.node_map.items()}
        self.num_nodes = len(all_nodes)
        
        logger.info(f"Graph has {self.num_nodes} nodes (clients + receivers).")
        
        # 2. Create Edges
        src = self.df['cst_dim_id'].astype(str).map(self.node_map).values
        dst = self.df['direction'].astype(str).map(self.node_map).values
        
        # Create symmetric adjacency
        all_src = np.concatenate([src, dst])
        all_dst = np.concatenate([dst, src])
        data = np.ones(len(all_src))
        
        # Create Sparse Matrix
        self.adj_scipy = coo_matrix((data, (all_src, all_dst)), shape=(self.num_nodes, self.num_nodes))

    def train_embeddings(self, epochs=50):
        """Generates embeddings using TruncatedSVD (Matrix Factorization).
        
        Args:
            epochs: Unused, kept for API compatibility
        """
        logger.info("Generating graph embeddings via TruncatedSVD...")
        
        # SVD on Adjacency Matrix factorizes the graph structure
        # This captures latent relationships between nodes
        svd = TruncatedSVD(n_components=16, n_iter=10, random_state=42)
        self.final_embeddings = svd.fit_transform(self.adj_scipy)
            
        logger.info(f"Embeddings generated. Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    def get_embeddings_df(self):
        """Returns a DataFrame with embedding features for each client."""
        embedding_cols = [f'gnn_emb_{i}' for i in range(self.final_embeddings.shape[1])]
        
        # Create a dictionary mapping original ID to embedding
        emb_dict = {}
        for node_id, idx in self.node_map.items():
            emb_dict[node_id] = self.final_embeddings[idx]
            
        # Create a DataFrame of embeddings
        emb_data = []
        for cst_id in self.df['cst_dim_id'].unique():
            cst_str = str(cst_id)
            if cst_str in self.node_map:
                emb_data.append([cst_id] + list(emb_dict[cst_str]))
            else:
                emb_data.append([cst_id] + [0]*len(embedding_cols))
                
        emb_df = pd.DataFrame(emb_data, columns=['cst_dim_id'] + embedding_cols)
        return emb_df

if __name__ == "__main__":
    # Test run
    from data_steward import DataStewardAgent
    
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    embedder = GraphEmbedderAgent(merged_df)
    embedder.prepare_graph()
    embedder.train_embeddings()
    emb_df = embedder.get_embeddings_df()
    print(emb_df.head())
