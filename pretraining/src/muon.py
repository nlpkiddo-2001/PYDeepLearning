import torch
import torch.distributed as dist

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step.
    In this way, the update to each 2D parameter's update is (approximately) bounded by the spectral radius of the parameter.

    Arguments:
        params: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov momentum. (True is a good default)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is a good default)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(p.grad)

                if nesterov:
                    g = p.grad.add(buf, alpha=momentum)
                else:
                    g = buf

                # Newton-Schulz Iteration
                # g is the gradient/update. We want to orthogonalize it.
                # We assume p is 2D. If not, this optimizer shouldn't be used for it.
                if g.ndim == 2:
                    X = g
                    # Scale X to avoid numerical instability
                    # We want spectral norm to be close to 1 for fast convergence of NS
                    # But calculating spectral norm is expensive.
                    # A common heuristic is to divide by Frobenius norm, but that's not quite right.
                    # Nanochat implementation often uses a specific scaling or just relies on the iterations.
                    # Let's use the standard NS update: X_{k+1} = 1.5 * X_k - 0.5 * X_k * X_k^T * X_k
                    
                    # Pre-scaling (optional but recommended for stability)
                    # X /= X.norm() + 1e-7 # Frobenius norm scaling?
                    # Let's follow the reference implementation logic if possible.
                    # Assuming standard NS:
                    
                    # For stability, we can normalize by Frobenius norm first
                    if X.size(0) > X.size(1):
                        X = X.T
                    
                    # Newton-Schulz iterations
                    # X_{k+1} = 1.5 * X_k - 0.5 * X_k * (X_k^T * X_k)
                    # We want to orthogonalize rows or columns?
                    # Usually we want X * X^T = I (rows orthogonal) if rows < cols
                    
                    # Let's use the implementation from the paper/reference:
                    # X = update
                    # for _ in range(ns_steps):
                    #   A = X @ X.T
                    #   B = 1.5 * I - 0.5 * A
                    #   X = B @ X
                    
                    # Efficient implementation:
                    # We need to handle non-square matrices.
                    # If M > N, we transpose.
                    
                    transposed = False
                    if X.size(0) > X.size(1):
                        X = X.T
                        transposed = True
                        
                    # Normalize spectral norm roughly
                    # X.norm() is Frobenius. Spectral <= Frobenius.
                    # We can just run NS.
                    
                    # Standard NS for orthogonalization of X (where X has orthonormal rows)
                    # X_{k+1} = X_k (1.5 I - 0.5 X_k^T X_k)
                    
                    # Wait, if X is (M, N) with M < N, we want XX^T = I.
                    # Then X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k
                    
                    # Normalize first to ensure convergence
                    X.div_(X.norm() + 1e-7) 
                    
                    for _ in range(ns_steps):
                        # A = X X^T
                        A = X @ X.T
                        # B = 3 I - A (scaled logic? No, 3*I - A is for inverse sqrt?)
                        # The update is X <- 1.5 X - 0.5 A X
                        
                        # X = 1.5 * X - 0.5 * A @ X
                        # X = 0.5 * (3 * X - A @ X)
                        X = 0.5 * (3 * X - A @ X)
                        
                    if transposed:
                        X = X.T
                        
                    # Update parameters
                    # p -= lr * X
                    # But we need to scale X to have specific spectral norm?
                    # The optimizer says "updates will have spectral norm of lr".
                    # NS makes spectral norm 1.
                    # So we multiply by lr.
                    
                    # However, we also need to scale by the spectral radius of the parameter itself?
                    # "update to each 2D parameter's update is (approximately) bounded by the spectral radius of the parameter"
                    # Actually, Muon replaces the update with an orthogonal matrix scaled by lr.
                    # But some implementations scale by max(1, param_rms) or similar.
                    # Let's stick to the simplest: update has spectral norm `lr`.
                    
                    p.data.add_(X, alpha=-lr)
                    
                else:
                    # Fallback for non-2D (shouldn't happen if filtered correctly)
                    p.data.add_(g, alpha=-lr)
