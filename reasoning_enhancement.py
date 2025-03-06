import torch
import einops
from tqdm import tqdm

def get_orthogonalized_matrix(
    matrix: torch.Tensor,
    direction: torch.Tensor,
    strength: float = 1.0
) -> torch.Tensor:
    """
    Orthogonalize a matrix with respect to a direction vector with controllable strength.
    
    Args:
        matrix: The weight matrix to orthogonalize
        direction: The direction vector to orthogonalize against
        strength: How strongly to orthogonalize (1.0 = full orthogonalization, 0.0 = no change)
    
    Returns:
        The orthogonalized matrix
    """
    # Ensure direction is on same device as matrix
    if matrix.device != direction.device:
        direction = direction.to(matrix.device)
    
    # Project the matrix onto the direction
    proj = einops.einsum(matrix, direction.view(-1, 1), '... d_model, d_model single -> ... single') * direction
    
    # Subtract the projection scaled by strength
    return matrix - (strength * proj)

def apply_orthogonalization(model, direction, strength=1.0):
    """
    Apply orthogonalization to key model weights with respect to a direction.
    
    Args:
        model: The transformer model
        direction: The direction to orthogonalize against
        strength: How strongly to orthogonalize
    """
    # Make sure direction is on the right device
    if direction.device != model.W_E.device:
        direction = direction.to(model.W_E.device)
    
    # Orthogonalize embedding weights
    model.W_E.data = get_orthogonalized_matrix(model.W_E, direction, strength)
    
    # Orthogonalize attention output and MLP output weights for each block
    for block in tqdm(model.blocks):
        if direction.device != block.attn.W_O.device:
            direction = direction.to(block.attn.W_O.device)
        
        block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, direction, strength)
        block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, direction, strength)
    
    return model 