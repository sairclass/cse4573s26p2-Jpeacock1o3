'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit.
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py.
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

#converts image to float
def to_float(img: torch.Tensor) -> torch.Tensor:
    img = img.float()
    if img.max() > 1.0:
        img = img / 255.0
    return img

#converts RGB image to grayscale
def rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    return K.color.rgb_to_grayscale(img.unsqueeze(0))[0, 0]

#sobel filters to compute image gradients
def sobel_xy(gray: torch.Tensor):
    kx = torch.tensor(
        [[1.0, 0.0, -1.0],
         [2.0, 0.0, -2.0],
         [1.0, 0.0, -1.0]],
        dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3) / 8.0

    ky = torch.tensor(
        [[1.0, 2.0, 1.0],
         [0.0, 0.0, 0.0],
         [-1.0, -2.0, -1.0]],
        dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3) / 8.0

    g = gray.unsqueeze(0).unsqueeze(0)
    Ix = torch.nn.functional.conv2d(g, kx, padding=1)[0, 0]
    Iy = torch.nn.functional.conv2d(g, ky, padding=1)[0, 0]
    return Ix, Iy

#Harris corner detection with non-maximum suppression and border removal.
def detect_harris_points(gray: torch.Tensor, max_points=500, border=20):
    Ix, Iy = sobel_xy(gray)

    Ixx = torch.nn.functional.avg_pool2d((Ix * Ix).unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)[0, 0]
    Iyy = torch.nn.functional.avg_pool2d((Iy * Iy).unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)[0, 0]
    Ixy = torch.nn.functional.avg_pool2d((Ix * Iy).unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)[0, 0]

    k = 0.04
    R = (Ixx * Iyy - Ixy * Ixy) - k * (Ixx + Iyy) ** 2
    R = torch.relu(R)

    H, W = gray.shape
    if H <= 2 * border or W <= 2 * border:
        border = 4

    R[:border, :] = 0
    R[-border:, :] = 0
    R[:, :border] = 0
    R[:, -border:] = 0

    nms = torch.nn.functional.max_pool2d(R.unsqueeze(0).unsqueeze(0), 7, stride=1, padding=3)[0, 0]
    keep = (R == nms)

    if R.max() > 0:
        keep = keep & (R > 0.01 * R.max())

    ys, xs = torch.where(keep)
    scores = R[ys, xs]

    if scores.numel() < 40:
        flat = R.reshape(-1)
        kpts = min(max_points, flat.numel())
        vals, idx = torch.topk(flat, k=kpts)
        ys = idx // W
        xs = idx % W
        scores = vals

    if scores.numel() == 0:
        return torch.empty((0, 2), dtype=gray.dtype, device=gray.device)

    kpts = min(max_points, scores.numel())
    _, top_idx = torch.topk(scores, k=kpts)
    xs = xs[top_idx].float()
    ys = ys[top_idx].float()

    return torch.stack([xs, ys], dim=1)

# Extracts a patch around each point and returns a normalized descriptor vector.
def extract_patch_descriptors(gray: torch.Tensor, pts: torch.Tensor, patch_size=21):
    if pts.shape[0] == 0:
        return torch.empty((0, patch_size * patch_size), dtype=gray.dtype, device=gray.device)

    H, W = gray.shape
    r = patch_size // 2

    g = torch.nn.functional.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)

    offsets = torch.linspace(-r, r, patch_size, dtype=gray.dtype, device=gray.device)
    yy = offsets.view(-1, 1).repeat(1, patch_size)
    xx = offsets.view(1, -1).repeat(patch_size, 1)
    base = torch.stack([xx, yy], dim=-1)

    grid = pts[:, None, None, :] + base[None, :, :, :]

    gx = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)

    patches = torch.nn.functional.grid_sample(
        g.expand(pts.shape[0], 1, H, W),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )[:, 0]

    desc = patches.reshape(pts.shape[0], -1)
    desc = desc - desc.mean(dim=1, keepdim=True)
    desc = desc / (desc.std(dim=1, keepdim=True) + 1e-6)
    return desc

# Matches descriptors using mutual nearest neighbors and Lowe's ratio test.
def match_descriptors(desc1: torch.Tensor, desc2: torch.Tensor, ratio_thresh=0.80):
    """
    Mutual NN + ratio test.
    Returns idx1, idx2
    """
    if desc1.shape[0] < 2 or desc2.shape[0] < 2:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    dmat = torch.cdist(desc1, desc2)

    vals12, idx12 = torch.topk(dmat, k=2, largest=False, dim=1)
    nn12 = idx12[:, 0]
    ratio = vals12[:, 0] / (vals12[:, 1] + 1e-8)

    nn21 = torch.argmin(dmat, dim=0)
    arange1 = torch.arange(desc1.shape[0], device=desc1.device)
    mutual = nn21[nn12] == arange1
    good = mutual & (ratio < ratio_thresh)

    idx1 = torch.where(good)[0]
    idx2 = nn12[good]
    return idx1, idx2

# Normalizes points for better numerical stability.
def normalize_points(pts: torch.Tensor):
    mean = pts.mean(dim=0)
    centered = pts - mean
    dist = torch.sqrt((centered ** 2).sum(dim=1) + 1e-8).mean()
    scale = (2.0 ** 0.5) / (dist + 1e-8)

    T = torch.tensor([
        [scale, 0.0, -scale * mean[0]],
        [0.0, scale, -scale * mean[1]],
        [0.0, 0.0, 1.0]
    ], dtype=pts.dtype, device=pts.device)

    pts_h = torch.cat([pts, torch.ones(pts.shape[0], 1, dtype=pts.dtype, device=pts.device)], dim=1)
    pts_n = (T @ pts_h.t()).t()
    return pts_n[:, :2], T

# Computes homography using Direct Linear Transform (DLT) algorithm.
def dlt_homography(pts1: torch.Tensor, pts2: torch.Tensor):
    n = pts1.shape[0]
    if n < 4:
        return None

    p1, T1 = normalize_points(pts1)
    p2, T2 = normalize_points(pts2)

    x = p1[:, 0]
    y = p1[:, 1]
    u = p2[:, 0]
    v = p2[:, 1]

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    A1 = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=1)
    A2 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=1)
    A = torch.cat([A1, A2], dim=0)

    try:
        _, _, Vh = torch.linalg.svd(A)
    except Exception:
        return None

    Hn = Vh[-1].reshape(3, 3)
    H = torch.linalg.inv(T2) @ Hn @ T1

    if torch.abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]

    if torch.isnan(H).any() or torch.isinf(H).any():
        return None

    return H

# Projects points using the homography and normalizes by the homogeneous coordinate.
def project_points(H: torch.Tensor, pts: torch.Tensor):
    pts_h = torch.cat([pts, torch.ones(pts.shape[0], 1, dtype=pts.dtype, device=pts.device)], dim=1)
    proj = (H @ pts_h.t()).t()
    z = proj[:, 2:3].clamp(min=1e-8)
    return proj[:, :2] / z

# RANSAC loop to find the best homography and inlier mask.
def ransac_homography(pts1, pts2, num_iters=2000, threshold=4.0):
    M = pts1.shape[0]
    if M < 4:
        return torch.eye(3, dtype=pts1.dtype, device=pts1.device), torch.zeros(M, dtype=torch.bool, device=pts1.device)

    best_H = None
    best_mask = torch.zeros(M, dtype=torch.bool, device=pts1.device)
    best_count = 0

    for _ in range(num_iters):
        idx = torch.randperm(M, device=pts1.device)[:4]
        H = dlt_homography(pts1[idx], pts2[idx])
        if H is None:
            continue

        proj = project_points(H, pts1)
        err = torch.sqrt(((proj - pts2) ** 2).sum(dim=1) + 1e-8)
        mask = err < threshold
        count = int(mask.sum().item())

        if count > best_count:
            best_count = count
            best_mask = mask
            best_H = H

            if count > 0.80 * M:
                break

    if best_H is None:
        return torch.eye(3, dtype=pts1.dtype, device=pts1.device), best_mask

    if best_mask.sum() >= 4:
        H_refit = dlt_homography(pts1[best_mask], pts2[best_mask])
        if H_refit is not None:
            best_H = H_refit

    return best_H, best_mask

# Given two warped images and their masks, select background pixels and average stable overlap.
def remove_foreground(w1, w2, mask1, mask2):
    C, H, W = w1.shape
    overlap = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)

    canvas = torch.zeros((C, H, W), dtype=w1.dtype, device=w1.device)
    canvas[:, only1] = w1[:, only1]
    canvas[:, only2] = w2[:, only2]

    if not overlap.any():
        return canvas

    # difference map in overlap
    diff = (w1 - w2).abs().mean(dim=0)  # H x W

    # dilate so full moving-object regions get covered
    diff_b = diff.unsqueeze(0).unsqueeze(0)
    dilated = torch.nn.functional.max_pool2d(
        diff_b, kernel_size=21, stride=1, padding=10
    )[0, 0]

    vals = dilated[overlap]
    thresh = vals.median() + 0.5 * vals.std()

    # changed = likely moving foreground / severe disagreement
    changed = (dilated > thresh) & overlap

    # expand a little more so whole object gets removed
    changed = torch.nn.functional.max_pool2d(
        changed.float().unsqueeze(0).unsqueeze(0),
        kernel_size=17, stride=1, padding=8
    )[0, 0] > 0

    # stable overlap should be averaged
    stable = overlap & (~changed)
    canvas[:, stable] = 0.5 * (w1[:, stable] + w2[:, stable])

    if changed.any():
        ks = 31
        sig = 10.0
        smooth1 = K.filters.gaussian_blur2d(w1.unsqueeze(0), (ks, ks), (sig, sig))[0]
        smooth2 = K.filters.gaussian_blur2d(w2.unsqueeze(0), (ks, ks), (sig, sig))[0]

        score1 = (w1 - smooth2).abs().mean(dim=0)
        score2 = (w2 - smooth1).abs().mean(dim=0)

        choose1 = changed & (score1 <= score2)
        choose2 = changed & (~choose1)

        canvas[:, choose1] = w1[:, choose1]
        canvas[:, choose2] = w2[:, choose2]

    return canvas

def safe_side_by_side(img1: torch.Tensor, img2: torch.Tensor):
    c, h1, w1 = img1.shape
    _, h2, w2 = img2.shape
    H = max(h1, h2)
    W = w1 + w2
    out = torch.zeros((c, H, W), dtype=img1.dtype, device=img1.device)
    out[:, :h1, :w1] = img1
    out[:, :h2, w1:w1 + w2] = img2
    return out


# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256))

    #sort so image order is consistent
    keys = sorted(list(imgs.keys()))

    #convert images to float and normalize
    img1 = to_float(imgs[keys[0]])
    img2 = to_float(imgs[keys[1]])

    #get channel, height, and width of the two images
    c, h1, w1 = img1.shape
    _, h2, w2 = img2.shape

    #convert to grayscale for feature detection
    gray1 =rgb_to_gray(img1)
    gray2 =rgb_to_gray(img2)

    # Detect Harris corner points
    pts1 = detect_harris_points(gray1, max_points=600, border=24)
    pts2 = detect_harris_points(gray2, max_points=600, border=24)

    #extract descriptors
    desc1 = extract_patch_descriptors(gray1, pts1, patch_size=21)
    desc2 = extract_patch_descriptors(gray2, pts2, patch_size=21)

    #match descriptors
    idx1, idx2 = match_descriptors(desc1, desc2, ratio_thresh=0.80)

    #if not enougbh matches, return side by side
    if idx1.numel() < 8:
        img = (safe_side_by_side(img1, img2).clamp(0, 1) * 255).to(torch.uint8)
        return img

    m1 = pts1[idx1]
    m2 = pts2[idx2]

    H12, inliers = ransac_homography(m1, m2, num_iters=2500, threshold=4.0)

    #if not enough inliers, return side by side
    if inliers.sum() < 8:
        img = (safe_side_by_side(img1, img2).clamp(0, 1) * 255).to(torch.uint8)
        return img

    #get four corners of image in homogenoous coordinates
    corners1 = torch.tensor(
        [[0.0, 0.0, 1.0],
         [w1 - 1.0, 0.0, 1.0],
         [0.0, h1 - 1.0, 1.0],
         [w1 - 1.0, h1 - 1.0, 1.0]],
        dtype=img1.dtype, device=img1.device
    ).t()

    #warp corners using estimated homography
    warped_corners1 = H12 @ corners1

    # Nomrmalize homogenous coordinates
    warped_corners1 = warped_corners1 / warped_corners1[2:3, :].clamp(min=1e-8)

    #four corners of image 2 in non-homogenous coordinates
    corners2_xy = torch.tensor(
        [[0.0, 0.0],
         [w2 - 1.0, 0.0],
         [0.0, h2 - 1.0],
         [w2 - 1.0, h2 - 1.0]],
        dtype=img1.dtype, device=img1.device
    )

    #combine warped corners of image 1 and corner of image 2 to find bounds
    all_x = torch.cat([warped_corners1[0], corners2_xy[:, 0]])
    all_y = torch.cat([warped_corners1[1], corners2_xy[:, 1]])

    x_min = int(torch.floor(all_x.min()).item())
    y_min = int(torch.floor(all_y.min()).item())
    x_max = int(torch.ceil(all_x.max()).item())
    y_max = int(torch.ceil(all_y.max()).item())

    #width and height of output
    out_w = max(1, x_max - x_min + 1)
    out_h = max(1, y_max - y_min + 1)

    #translation to shift
    T = torch.tensor(
        [[1.0, 0.0, -float(x_min)],
         [0.0, 1.0, -float(y_min)],
         [0.0, 0.0, 1.0]],
        dtype=img1.dtype, device=img1.device
    )

    #warp both images to canvas
    H1_canvas = (T @ H12).unsqueeze(0)
    H2_canvas = T.unsqueeze(0)

    #
    warped1 = K.geometry.warp_perspective(
        img1.unsqueeze(0), H1_canvas, dsize=(out_h, out_w), align_corners=True
    )[0]

    warped2 = K.geometry.warp_perspective(
        img2.unsqueeze(0), H2_canvas, dsize=(out_h, out_w), align_corners=True
    )[0]

    mask1 = K.geometry.warp_perspective(
        torch.ones((1, 1, h1, w1), dtype=img1.dtype, device=img1.device),
        H1_canvas,
        dsize=(out_h, out_w),
        align_corners=True
    )[0, 0] > 0.5

    mask2 = K.geometry.warp_perspective(
        torch.ones((1, 1, h2, w2), dtype=img1.dtype, device=img1.device),
        H2_canvas,
        dsize=(out_h, out_w),
        align_corners=True
    )[0, 0] > 0.5

    canvas = remove_foreground(warped1, warped2, mask1, mask2)
    img = (canvas.clamp(0, 1) * 255).to(torch.uint8)
    return img


# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256), dtype=torch.uint8)
    overlap = torch.zeros((0, 0), dtype=torch.int64)

    keys = sorted(imgs.keys())
    N = len(keys)

    if N == 0:
        return img, overlap

    tensors = [imgs[k].float() / 255.0 for k in keys]

    detector = K.feature.KeyNetAffNetHardNet(num_features=2500, upright=True)
    matcher = K.feature.DescriptorMatcher('snn', 0.75)

    all_lafs = []
    all_descs = []

    #get features for all images without gradients
    with torch.no_grad():
        for t in tensors:
            feats = detector(K.color.rgb_to_grayscale(t.unsqueeze(0)))
            all_lafs.append(feats[0])
            all_descs.append(feats[2])

    #matrixes for overlap relationships and scores
    overlap_matrix = torch.zeros((N, N), dtype=torch.int64)
    pair_score = torch.zeros((N, N), dtype=torch.float32)

    #pairwise homographies of images
    H_pairs = [[None] * N for _ in range(N)]

    #threshholds
    MIN_MATCHES = 20
    MIN_INLIERS = 50
    MAX_REPROJ = 3.5

    #loop all pairs
    for i in range(N):
        overlap_matrix[i, i] = 1
        for j in range(i + 1, N):
            #try to match descriptors, if fails, skip
            try:
                dists, idxs = matcher(all_descs[i][0], all_descs[j][0])
            except Exception:
                continue
            
            #skip if not enough matched
            if idxs.shape[0] < MIN_MATCHES:
                continue

            #get coords from laf
            kps_i = K.feature.get_laf_center(all_lafs[i])[0]
            kps_j = K.feature.get_laf_center(all_lafs[j])[0]
            pts_i = kps_i[idxs[:, 0]]
            pts_j = kps_j[idxs[:, 1]]

            # try to find homography using RANSAC, if fails, skip
            try:
                H_ij, inlier_mask = ransac_homography(
                    pts_i, pts_j, num_iters=2000, threshold=MAX_REPROJ
                )
            except Exception:
                continue

            #count inliers, if not enough, skip
            n_inliers = int(inlier_mask.sum().item())
            if n_inliers < MIN_INLIERS:
                continue

            #get inlier points
            in_i = pts_i[inlier_mask]
            in_j = pts_j[inlier_mask]

            #get coverage area of inliers in both images
            min_x_i, _ = torch.min(in_i[:, 0], dim=0)
            max_x_i, _ = torch.max(in_i[:, 0], dim=0)
            min_y_i, _ = torch.min(in_i[:, 1], dim=0)
            max_y_i, _ = torch.max(in_i[:, 1], dim=0)

            min_x_j, _ = torch.min(in_j[:, 0], dim=0)
            max_x_j, _ = torch.max(in_j[:, 0], dim=0)
            min_y_j, _ = torch.min(in_j[:, 1], dim=0)
            max_y_j, _ = torch.max(in_j[:, 1], dim=0)

            #get image dimensions
            h_i, w_i = tensors[i].shape[1], tensors[i].shape[2]
            h_j, w_j = tensors[j].shape[1], tensors[j].shape[2]

            #get how much of each image is coverd by matches
            cover_i = ((max_x_i - min_x_i) * (max_y_i - min_y_i) / float(h_i * w_i)).item()
            cover_j = ((max_x_j - min_x_j) * (max_y_j - min_y_j) / float(h_j * w_j)).item()

            #get corners
            corners_i = torch.tensor(
                [[0.0, 0.0, 1.0],
                 [w_i - 1.0, 0.0, 1.0],
                 [0.0, h_i - 1.0, 1.0],
                 [w_i - 1.0, h_i - 1.0, 1.0]],
                dtype=torch.float32
            ).T

            #project corners of image i to image j using homography
            try:
                proj = H_ij @ corners_i
                proj = proj / proj[2:3, :].clamp(min=1e-8)
            except Exception:
                continue

            #check for numerical issues
            if torch.isnan(proj).any() or torch.isinf(proj).any():
                continue

            span_x = (proj[0].max() - proj[0].min()).item()
            span_y = (proj[1].max() - proj[1].min()).item()

            # reject weak or bad matches
            if cover_i < 0.08 or cover_j < 0.08:
                continue
            if span_x > 3.0 * w_j or span_y > 3.0 * h_j:
                continue

            #mark as overlapping and save score and homography
            overlap_matrix[i, j] = 1
            overlap_matrix[j, i] = 1
            pair_score[i, j] = float(n_inliers)
            pair_score[j, i] = float(n_inliers)
            H_pairs[i][j] = H_ij

            #try to invert homography, if fails, mark as None
            try:
                H_pairs[j][i] = torch.inverse(H_ij)
            except Exception:
                H_pairs[j][i] = None

    # Connected components over overlap graph.
    visited = [False] * N
    components = []

    def bfs_component(start):
        q = [start]
        visited[start] = True
        comp = [start]
        while q:
            u = q.pop(0)
            for v in range(N):
                if not visited[v] and overlap_matrix[u, v] == 1:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        return comp

    #find connected components of overlap graph
    for i in range(N):
        if not visited[i]:
            components.append(bfs_component(i))

    #if no components, return empty image and overlap
    if len(components) == 0:
        return img, overlap_matrix

    main_comp = max(components, key=len)

    if len(main_comp) == 1:
        only = main_comp[0]
        img = imgs[keys[only]]
        overlap = overlap_matrix
        return img, overlap

    # Use the most connected / strongest node as reference.
    ref_idx = max(
        main_comp,
        key=lambda i: (overlap_matrix[i, main_comp].sum().item(), pair_score[i, main_comp].sum().item())
    )

    # Build a maximum spanning tree so one weak cycle edge cannot corrupt the whole panorama.
    in_tree = {ref_idx}
    parent = {ref_idx: None}
    order = [ref_idx]

    while len(in_tree) < len(main_comp):
        best_u, best_v, best_w = None, None, -1.0
        for u in list(in_tree):
            for v in main_comp:
                if v in in_tree:
                    continue
                w = pair_score[u, v].item()
                if w > best_w and H_pairs[v][u] is not None:
                    best_u, best_v, best_w = u, v, w

        if best_v is None:
            break

        in_tree.add(best_v)
        parent[best_v] = best_u
        order.append(best_v)

    # Compose transforms to reference frame.
    H_to_ref = [None] * N
    H_to_ref[ref_idx] = torch.eye(3, dtype=torch.float32)

    changed = True
    while changed:
        changed = False
        for v in order:
            if v == ref_idx:
                continue
            if H_to_ref[v] is not None:
                continue
            u = parent.get(v, None)
            if u is None or H_to_ref[u] is None:
                continue
            if H_pairs[v][u] is None:
                continue
            H_to_ref[v] = H_to_ref[u] @ H_pairs[v][u]
            changed = True

    #keep only valid nodes with homography to reference
    valid_nodes = [i for i in main_comp if H_to_ref[i] is not None]
    if len(valid_nodes) == 0:
        return img, overlap_matrix

    # bounds
    all_x = []
    all_y = []

    for i in valid_nodes:
        h_i, w_i = tensors[i].shape[1], tensors[i].shape[2]
        corners = torch.tensor(
            [[0.0, 0.0, 1.0],
             [w_i - 1.0, 0.0, 1.0],
             [0.0, h_i - 1.0, 1.0],
             [w_i - 1.0, h_i - 1.0, 1.0]],
            dtype=torch.float32
        ).T
        proj = H_to_ref[i] @ corners
        proj = proj / proj[2:3, :].clamp(min=1e-8)
        all_x.append(proj[0])
        all_y.append(proj[1])

    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)

    #output size
    x_min = int(torch.floor(all_x.min()).item())
    y_min = int(torch.floor(all_y.min()).item())
    x_max = int(torch.ceil(all_x.max()).item())
    y_max = int(torch.ceil(all_y.max()).item())

    out_w = max(1, x_max - x_min + 1)
    out_h = max(1, y_max - y_min + 1)

    #downscale if to big, I was having memory issues
    MAX_DIM = 3500
    scale = 1.0
    if max(out_w, out_h) > MAX_DIM:
        scale = MAX_DIM / float(max(out_w, out_h))
        out_w = max(1, int(out_w * scale))
        out_h = max(1, int(out_h * scale))

    #translate and scale to output 
    T = torch.tensor(
        [[scale, 0.0, -x_min * scale],
         [0.0, scale, -y_min * scale],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32
    )

    #create canvas and weight map
    canvas = torch.zeros((3, out_h, out_w), dtype=torch.float32)
    weight = torch.zeros((1, out_h, out_w), dtype=torch.float32)

    #warp images to canvas and blend
    for i in valid_nodes:
        h_i, w_i = tensors[i].shape[1], tensors[i].shape[2]
        H_canvas = (T @ H_to_ref[i]).unsqueeze(0)

        warped = K.geometry.warp_perspective(
            tensors[i].unsqueeze(0), H_canvas, dsize=(out_h, out_w), align_corners=True
        )[0]

        #create mask of valid pixels
        mask = K.geometry.warp_perspective(
            torch.ones((1, 1, h_i, w_i), dtype=torch.float32),
            H_canvas,
            dsize=(out_h, out_w),
            align_corners=True
        )[0, 0]

        mask = (mask > 0.5).float()

        # Simple feathering: downweight edges of each warped image.
        soft = K.filters.gaussian_blur2d(mask.unsqueeze(0).unsqueeze(0), (31, 31), (8.0, 8.0))[0, 0]
        soft = soft * mask + 1e-6

        canvas += warped * soft.unsqueeze(0)
        weight += soft.unsqueeze(0)
    #normalize by weight
    canvas = (canvas / weight.clamp(min=1e-6)).clamp(0, 1)
    img = (canvas * 255).to(torch.uint8)
    overlap = overlap_matrix
    return img, overlap