# utils/geometry.py

import torch


def compute_aligned_rmsd(pred, true, mask=None):
    """
    Compute RMSD after Kabsch alignment.
    NaN-safe: handles degenerate structures (flat, zero-variance, tiny chains)
    that cause silent NaN from torch.linalg.svd without raising RuntimeError.
    """
    if mask is None:
        mask = torch.ones(pred.shape[0], pred.shape[1],
                          dtype=torch.bool, device=pred.device)

    batch_size = pred.shape[0]
    rmsds = []

    for i in range(batch_size):
        p = pred[i]   # [L, 3]
        t = true[i]   # [L, 3]
        m = mask[i]   # [L]

        # Differentiable zero connected to pred & true - preserves graph
        zero = (p.sum() + t.sum()) * 0.0

        n_valid = m.sum()
        if n_valid < 3:
            rmsds.append(zero)
            continue

        p_masked = p[m]   # [N, 3]
        t_masked = t[m]   # [N, 3]

        # Check for degenerate structure before SVD
        # (all-identical coords -> zero variance -> singular covariance matrix)
        if p_masked.std() < 1e-6 or t_masked.std() < 1e-6:
            # Fall back to unaligned RMSD (still a valid gradient signal)
            diff  = p_masked - t_masked
            mse   = (diff ** 2).mean()
            rmsds.append(torch.sqrt(mse + 1e-8))
            continue

        # Centre coordinates
        p_centered = p_masked - p_masked.mean(dim=0, keepdim=True)
        t_centered = t_masked - t_masked.mean(dim=0, keepdim=True)

        # Covariance matrix - small regularisation prevents exact singularity
        H = p_centered.T @ t_centered   # [3, 3]
        H = H + 1e-6 * torch.eye(3, device=H.device, dtype=H.dtype)

        try:
            U, S, Vt = torch.linalg.svd(H, full_matrices=True)
        except RuntimeError as e:
            print(f"[WARNING] SVD RuntimeError in sample {i}: {e}")
            rmsds.append(zero)
            continue

        #  NaN guard - svd can return NaN silently on degenerate input 
        if not (torch.isfinite(U).all() and
                torch.isfinite(S).all() and
                torch.isfinite(Vt).all()):
            diff  = p_centered - t_centered
            mse   = (diff ** 2).mean()
            rmsds.append(torch.sqrt(mse + 1e-8))
            continue

        # Proper rotation (det = +1, no reflection)
        d    = torch.det(U @ Vt)
        diag = torch.ones(3, device=H.device, dtype=H.dtype)
        diag[2] = d.sign()
        R = U @ torch.diag(diag) @ Vt

        # Guard rotation matrix too
        if not torch.isfinite(R).all():
            diff  = p_centered - t_centered
            mse   = (diff ** 2).mean()
            rmsds.append(torch.sqrt(mse + 1e-8))
            continue

        p_aligned = p_centered @ R.T
        diff      = p_aligned - t_centered
        mse       = (diff ** 2).mean()
        rmsds.append(torch.sqrt(mse + 1e-8))

    return torch.stack(rmsds).mean()


def compute_unaligned_rmsd(pred_coords, true_coords, mask=None):
    """
    Compute RMSD without alignment (for comparison/debugging).
    """
    if mask is None:
        mask = torch.ones(pred_coords.shape[0], pred_coords.shape[1],
                          dtype=torch.bool, device=pred_coords.device)

    diff     = pred_coords - true_coords
    sq_dist  = (diff ** 2).sum(dim=-1)   # [B, L]
    sq_dist  = sq_dist * mask.float()
    lengths  = mask.sum(dim=1).clamp(min=1)

    msd_per_chain  = sq_dist.sum(dim=1) / lengths
    rmsd_per_chain = torch.sqrt(msd_per_chain + 1e-8)

    return rmsd_per_chain.mean()


def compute_ca_ca_distances(coords, mask=None):
    """
    Compute consecutive Cα-Cα distances.
    In real proteins, consecutive Cα-Cα distances are ~3.8Å ± 0.2Å.

    Args:
        coords: [B, L, 3] Cα coordinates
        mask:   [B, L] valid positions

    Returns:
        dict with mean, std, deviation from ideal (3.8Å)
    """
    if mask is None:
        mask = torch.ones(coords.shape[0], coords.shape[1],
                          dtype=torch.bool, device=coords.device)

    ideal_distance = 3.8
    all_distances  = []

    for b in range(coords.shape[0]):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        if len(valid_idx) < 2:
            continue
        for i in range(len(valid_idx) - 1):
            idx_i = valid_idx[i]
            idx_j = valid_idx[i + 1]
            if idx_j - idx_i == 1:   # consecutive residues only
                dist = torch.norm(coords[b, idx_j] - coords[b, idx_i])
                all_distances.append(dist)

    if not all_distances:
        z = torch.tensor(0.0, device=coords.device)
        return {'mean': z, 'std': z, 'deviation': z}

    all_distances = torch.stack(all_distances)
    return {
        'mean':      all_distances.mean(),
        'std':       all_distances.std(),
        'deviation': (all_distances - ideal_distance).abs().mean(),
    }


def compute_per_region_rmsd(pred_coords, true_coords, mask, cdr_mask=None):
    """
    Compute RMSD separately for framework and CDR regions.

    Args:
        pred_coords: [B, L, 3]
        true_coords: [B, L, 3]
        mask:        [B, L] valid positions
        cdr_mask:    [B, L] True = CDR, False = framework

    Returns:
        dict with framework_rmsd, cdr_rmsd, overall_rmsd
    """
    results = {}

    if cdr_mask is None:
        results['overall_rmsd'] = compute_aligned_rmsd(pred_coords, true_coords, mask)
        return results

    framework_mask = (~cdr_mask) & mask
    if framework_mask.sum() > 0:
        results['framework_rmsd'] = compute_aligned_rmsd(pred_coords, true_coords, framework_mask)

    cdr_combined_mask = cdr_mask & mask
    if cdr_combined_mask.sum() > 0:
        results['cdr_rmsd'] = compute_aligned_rmsd(pred_coords, true_coords, cdr_combined_mask)

    results['overall_rmsd'] = compute_aligned_rmsd(pred_coords, true_coords, mask)
    return results


def compute_comprehensive_geometry_metrics(pred_coords, true_coords, mask, cdr_mask=None):
    """
    Compute all geometry metrics at once.
    """
    metrics = {}

    metrics['rmsd_aligned']   = compute_aligned_rmsd(pred_coords, true_coords, mask).item()
    metrics['rmsd_unaligned'] = compute_unaligned_rmsd(pred_coords, true_coords, mask).item()

    if metrics['rmsd_unaligned'] > 0:
        metrics['alignment_improvement'] = (
            1.0 - metrics['rmsd_aligned'] / metrics['rmsd_unaligned']
        )
    else:
        metrics['alignment_improvement'] = 0.0

    ca_ca_stats = compute_ca_ca_distances(pred_coords, mask)
    metrics['ca_ca_distance_mean'] = ca_ca_stats['mean'].item()
    metrics['ca_ca_distance_std']  = ca_ca_stats['std'].item()
    metrics['ca_ca_deviation']     = ca_ca_stats['deviation'].item()

    if cdr_mask is not None:
        region_metrics = compute_per_region_rmsd(pred_coords, true_coords, mask, cdr_mask)
        for key, val in region_metrics.items():
            metrics[key] = val.item() if torch.is_tensor(val) else val

    return metrics