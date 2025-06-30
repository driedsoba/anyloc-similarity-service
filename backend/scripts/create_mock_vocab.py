"""
Script to create mock VLAD vocabulary for development
"""
import torch
import numpy as np
from pathlib import Path

def create_mock_vlad_vocabulary():
    """Create a mock VLAD vocabulary file for testing"""
    
    # Configuration
    num_clusters = 64
    desc_dim = 1024
    
    # Create mock cluster centers
    cluster_centers = torch.randn(num_clusters, desc_dim)
    
    # Normalize cluster centers
    cluster_centers = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)
    
    # Create vocabulary data
    vocab_data = {
        'cluster_centers': cluster_centers,
        'num_clusters': num_clusters,
        'desc_dim': desc_dim,
        'metadata': {
            'created_by': 'mock_generator',
            'description': 'Mock VLAD vocabulary for development',
            'version': '1.0'
        }
    }
    
    # Save vocabulary
    vocab_path = Path(__file__).parent.parent / "models" / "vocabulary" / "vlad_centers.pth"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(vocab_data, vocab_path)
    print(f"Created mock VLAD vocabulary at: {vocab_path}")
    print(f"Clusters: {num_clusters}, Dimensions: {desc_dim}")

if __name__ == "__main__":
    create_mock_vlad_vocabulary()
