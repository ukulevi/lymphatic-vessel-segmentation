"""
Generate pseudo-labels using baseline model predictions 
and optical flow for temporal consistency.
With TTA (Test Time Augmentation) for better predictions.
"""
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import albumentations as A

def calculate_optical_flow(prev_frame, next_frame):
    """
    Calculate optical flow between consecutive frames với cải thiện
    
    Returns:
        flow: Optical flow field, hoặc None nếu motion quá lớn
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE để tăng contrast (giúp optical flow tốt hơn)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    prev_gray = clahe.apply(prev_gray)
    next_gray = clahe.apply(next_gray)
    
    # Tính optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Validate flow: Nếu motion quá lớn, có thể không đáng tin
    flow_magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    max_flow = np.percentile(flow_magnitude, 95)
    if max_flow > 50:  # Motion quá lớn (> 50 pixels)
        return None  # Signal để không dùng warping
    
    return flow

def warp_mask(mask, flow):
    """Warp mask using optical flow field"""
    h, w = mask.shape[:2]
    flow_map = np.array([(x,y) for y in range(h) for x in range(w)])
    flow_map = flow_map.reshape(h,w,2) + flow
    
    warped = cv2.remap(
        mask.astype(np.float32),
        flow_map[...,0].astype(np.float32),
        flow_map[...,1].astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped

def get_tta_transforms():
    """
    Tạo danh sách transforms cho Test Time Augmentation
    Bao gồm nhiều augmentations để match với training augmentation:
    - Original, flips, rotations, scale variations
    """
    transforms = []
    reverse_ops = []  # Operations để reverse augmentation
    
    # 1. Original (no augmentation)
    transforms.append(A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    reverse_ops.append(None)  # Không cần reverse
    
    # 2. Horizontal flip
    transforms.append(A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    reverse_ops.append('hflip')  # Reverse bằng horizontal flip
    
    # 3. Vertical flip
    transforms.append(A.Compose([
        A.Resize(256, 256),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    reverse_ops.append('vflip')  # Reverse bằng vertical flip
    
    # 4. Rotate 90 degrees
    transforms.append(A.Compose([
        A.Resize(256, 256),
        A.RandomRotate90(p=1.0),  # RandomRotate90 cho 90 độ chính xác
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    reverse_ops.append('rot90')  # Reverse bằng rotate ngược lại
    
    # 5. Rotate 180 degrees  
    transforms.append(A.Compose([
        A.Resize(256, 256),
        A.Rotate(limit=(180, 180), p=1.0, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    reverse_ops.append('rot180')  # Reverse bằng rotate 180 (giống nhau)
    
    # 6. Horizontal Flip + Rotate 90 (combination)
    transforms.append(A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    reverse_ops.append('hflip_rot90')  # Reverse cả hai operations
    
    return transforms, reverse_ops

def reverse_tta_augmentation(pred_mask, reverse_op):
    """Reverse augmentation để đưa prediction về original orientation"""
    if reverse_op == 'hflip':
        return cv2.flip(pred_mask, 1)  # Horizontal flip
    elif reverse_op == 'vflip':
        return cv2.flip(pred_mask, 0)  # Vertical flip
    elif reverse_op == 'rot90':
        # Rotate 90 clockwise → reverse bằng counter-clockwise 90
        return cv2.rotate(pred_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif reverse_op == 'rot180':
        return cv2.rotate(pred_mask, cv2.ROTATE_180)
    elif reverse_op == 'hflip_rot90':
        # Reverse: rotate ngược lại rồi flip ngược lại
        reversed = cv2.rotate(pred_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.flip(reversed, 1)
    else:
        return pred_mask  # No reverse needed

def apply_tta_prediction(model, image_rgb, transforms, reverse_ops, device):
    """
    Áp dụng TTA: predict trên nhiều augmented versions và average
    
    Args:
        model: Trained model
        image_rgb: RGB image (H, W, 3)
        transforms: List of augmentation transforms
        reverse_ops: List of reverse operations
        device: Device to run on
        
    Returns:
        Averaged probability map (256x256)
    """
    predictions = []
    
    for tta_transform, reverse_op in zip(transforms, reverse_ops):
        # Apply augmentation
        augmented = tta_transform(image=image_rgb)
        aug_img = augmented['image']
        
        # Convert to tensor (normalize đã được apply trong transform)
        if isinstance(aug_img, np.ndarray):
            # Image đã được normalize về [0, 1] range
            x = torch.from_numpy(aug_img.transpose(2, 0, 1)).float()
        else:
            x = aug_img.float()
        
        x = x.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(x)
            if isinstance(pred, dict):
                pred = pred['out']
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        # Reverse augmentation để đưa về original orientation
        pred_prob_reversed = reverse_tta_augmentation(pred_prob, reverse_op)
        predictions.append(pred_prob_reversed)
    
    # Average all predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

def fill_holes(mask):
    """
    Fill holes trong binary mask bằng flood fill
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    h, w = mask_u8.shape
    
    # Tạo mask để flood fill (cần lớn hơn 2 pixels ở mỗi bên)
    mask_fill = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask_fill[1:h+1, 1:w+1] = mask_u8
    
    # Flood fill từ background (0,0) để tìm holes
    cv2.floodFill(mask_fill, None, (0, 0), 255)
    
    # Invert để lấy holes
    holes = 255 - mask_fill[1:h+1, 1:w+1]
    
    # Fill holes vào mask gốc
    mask_filled = np.maximum(mask_u8, holes)
    
    return (mask_filled > 127).astype(np.uint8)

def post_process_pseudo_mask(mask, min_size=300, image_size=None):
    """
    Post-process pseudo-label mask để giảm over-segmentation và fill holes
    
    Args:
        mask: Binary mask (0-1)
        min_size: Minimum size of connected component to keep (will be adaptive if image_size provided)
        image_size: Tuple (H, W) để tính adaptive min_size
        
    Returns:
        Cleaned binary mask
    """
    # Adaptive min_size dựa trên kích thước ảnh
    if image_size is not None:
        total_pixels = image_size[0] * image_size[1]
        adaptive_min_size = max(min_size, int(total_pixels * 0.0005))  # Giảm từ 0.001 xuống 0.0005 (0.05%)
    else:
        adaptive_min_size = min_size
    
    # Fill holes trước tiên (quan trọng!)
    mask_filled = fill_holes(mask)
    
    # Morphological operations nhẹ hơn để giữ vùng biên
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Kernel nhỏ cho biên
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel vừa
    
    # Closing với kernel lớn hơn để fill holes tốt hơn
    mask_closed_large = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Closing nhẹ để fill các gaps nhỏ ở biên
    mask_closed = cv2.morphologyEx(mask_closed_large, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Opening nhẹ để loại bỏ protrusions nhỏ (không làm mất biên)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Remove small connected components với adaptive min_size (giảm để giữ biên)
    labeled, num_features = ndimage.label(mask_opened)
    sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
    
    if len(sizes) > 0:
        # Keep only components larger than adaptive_min_size
        mask_cleaned = np.zeros_like(mask_opened)
        for label_id, size in enumerate(sizes, start=1):
            if size >= adaptive_min_size:
                mask_cleaned[labeled == label_id] = 1
        
        # Nếu có nhiều components, giữ component lớn nhất + các components gần nó
        if num_features > 1:
            largest_idx = np.argmax(sizes) + 1
            mask_final = np.zeros_like(mask_cleaned)
            mask_final[labeled == largest_idx] = 1
            
            # Giữ các components lớn khác (top 3)
            sorted_indices = np.argsort(sizes)[::-1][:3]
            for idx in sorted_indices:
                if sizes[idx] >= adaptive_min_size:
                    mask_final[labeled == (idx + 1)] = 1
            
            mask_cleaned = mask_final
    else:
        mask_cleaned = mask_opened
    
    # Final closing nhẹ để smooth biên
    mask_final = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    return mask_final

def generate_pseudo_labels(model, unlabeled_dir, output_dir, device="cuda", window_size=5):
    """Generate pseudo-labels for unlabeled frames
    
    Uses model predictions + optical flow propagation với temporal window
    để tạo pseudo-labels mượt mà và nhất quán theo thời gian.
    
    Args:
        model: Trained baseline model
        unlabeled_dir: Directory containing unlabeled frames
        output_dir: Where to save generated masks
        device: Device to run model on
        window_size: Số frame để temporal smoothing (mặc định 5)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect frame paths
    frame_paths = []
    for root, _, files in os.walk(unlabeled_dir):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                frame_paths.append(os.path.join(root, f))
    frame_paths.sort()
    
    if len(frame_paths) < 2:
        print("Warning: < 2 frames found; generating per-frame masks without optical flow.")
        window_size = 1
        
    saved_count = 0
    with torch.no_grad():
        # Pre-compute TTA transforms
        tta_transforms, reverse_ops = get_tta_transforms()
        
        # Buffer để lưu các frame và predictions gần đây
        # Format: (frame_path, frame, pred_prob_original)
        frame_buffer = []
        
        for idx, frame_path in enumerate(tqdm(frame_paths, desc="Generating pseudo-labels")):
            # Load current frame
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_size = frame_rgb.shape[:2]  # (H, W)
            
            # Apply TTA: predict trên nhiều augmented versions và average
            pred_prob = apply_tta_prediction(model, frame_rgb, tta_transforms, reverse_ops, device)
            
            # Resize probability map về original size với interpolation tốt hơn
            pred_prob_original = cv2.resize(
                pred_prob,
                (original_size[1], original_size[0]),  # (W, H)
                interpolation=cv2.INTER_CUBIC  # Thay INTER_LINEAR bằng INTER_CUBIC để mượt hơn
            )
            
            # Lưu vào buffer (cùng với frame_path để save sau)
            frame_buffer.append((frame_path, frame, pred_prob_original))
            
            # Chỉ xử lý khi có đủ frame trong buffer hoặc là frame cuối
            if len(frame_buffer) < window_size and idx < len(frame_paths) - 1:
                continue
            
            # Lấy frame ở giữa window để process
            center_idx = len(frame_buffer) // 2 if len(frame_buffer) >= window_size else len(frame_buffer) - 1
            center_frame_path, center_frame, center_pred_prob = frame_buffer[center_idx]
            
            # Temporal smoothing: weighted average probability maps từ các frame trong window
            if len(frame_buffer) > 1:
                # Tính optical flow và warp các predictions về center frame
                warped_probs = []
                valid_weights = []
                
                for i, (buf_path, buf_frame, buf_pred_prob) in enumerate(frame_buffer):
                    if i == center_idx:
                        # Frame ở giữa, không cần warp
                        warped_probs.append(buf_pred_prob)
                        valid_weights.append(1.0)  # Weight cao nhất cho center frame
                    else:
                        # Warp prediction về center frame bằng optical flow
                        flow = calculate_optical_flow(buf_frame, center_frame)
                        
                        if flow is not None:
                            # Optical flow hợp lệ, warp probability map
                            warped_prob = warp_mask(buf_pred_prob.astype(np.float32), flow)
                            warped_probs.append(warped_prob)
                            valid_weights.append(0.8)  # Weight cho warped predictions
                        else:
                            # Optical flow không hợp lệ (motion quá lớn), dùng direct prediction
                            warped_probs.append(buf_pred_prob)
                            valid_weights.append(0.5)  # Weight thấp hơn cho direct prediction
                
                # Weighted average: frame ở giữa có weight cao hơn
                n_frames = len(warped_probs)
                if n_frames > 1:
                    # Gaussian weights: center frame có weight cao nhất
                    center_weight = n_frames // 2
                    gaussian_weights = np.exp(-0.5 * ((np.arange(n_frames) - center_weight) / (n_frames/4))**2)
                    # Combine với valid_weights
                    combined_weights = gaussian_weights * np.array(valid_weights)
                    combined_weights = combined_weights / combined_weights.sum()  # Normalize
                    smoothed_prob = np.average(warped_probs, axis=0, weights=combined_weights)
                else:
                    smoothed_prob = warped_probs[0]
            else:
                smoothed_prob = center_pred_prob
            
            # Gaussian smoothing nhẹ hơn để giữ biên tốt hơn
            from scipy.ndimage import gaussian_filter
            smoothed_prob = gaussian_filter(smoothed_prob, sigma=0.2)  # Giảm từ 0.3 xuống 0.2
            
            # Adaptive confidence threshold dựa trên số frames trong buffer
            # Frames đầu (thiếu temporal context) cần threshold thấp hơn
            if len(frame_buffer) < window_size:
                confidence_threshold = 0.50  # Thấp hơn cho frames đầu
                min_size = 100  # Nhỏ hơn để giữ lại mask nhỏ
            else:
                confidence_threshold = 0.55  # Giảm từ 0.60 xuống 0.55 để giữ nhiều vùng hơn
                min_size = 150  # Giảm từ 300 xuống 150 để giữ lại vessel mỏng
            
            pred_mask = (smoothed_prob > confidence_threshold).astype(np.uint8)
            
            # Post-process với min_size adaptive
            pred_mask = post_process_pseudo_mask(pred_mask, min_size=min_size, image_size=original_size)
            
            # Save mask cho frame ở giữa window
            center_frame_name = os.path.basename(os.path.splitext(center_frame_path)[0])
            out_path = os.path.join(output_dir, center_frame_name + '_mask.png')
            cv2.imwrite(out_path, pred_mask * 255)
            saved_count += 1
            
            # Xóa frame đầu tiên khỏi buffer (sliding window)
            if len(frame_buffer) >= window_size:
                frame_buffer.pop(0)
        
        # Xử lý các frame còn lại trong buffer (ở cuối video)
        while len(frame_buffer) > 0:
            center_idx = len(frame_buffer) // 2
            center_frame_path, center_frame, center_pred_prob = frame_buffer[center_idx]
            
            # Temporal smoothing với các frame còn lại (weighted average)
            if len(frame_buffer) > 1:
                warped_probs = []
                for i, (buf_path, buf_frame, buf_pred_prob) in enumerate(frame_buffer):
                    if i == center_idx:
                        warped_probs.append(buf_pred_prob)
                    else:
                        flow = calculate_optical_flow(buf_frame, center_frame)
                        warped_prob = warp_mask(buf_pred_prob.astype(np.float32), flow)
                        warped_probs.append(warped_prob)
                
                # Weighted average
                n_frames = len(warped_probs)
                center_weight = n_frames // 2
                weights = np.exp(-0.5 * ((np.arange(n_frames) - center_weight) / (n_frames/4))**2)
                weights = weights / weights.sum()
                smoothed_prob = np.average(warped_probs, axis=0, weights=weights)
            else:
                smoothed_prob = center_pred_prob
            
            # Gaussian smoothing rất nhẹ để giữ biên
            from scipy.ndimage import gaussian_filter
            smoothed_prob = gaussian_filter(smoothed_prob, sigma=0.3)  # Giảm từ 0.5 xuống 0.3 để giữ biên tốt hơn
            
            # Apply threshold (giảm để giữ vùng biên)
            confidence_threshold = 0.60  # Giảm từ 0.65 xuống 0.60 để giữ nhiều vùng biên hơn
            pred_mask = (smoothed_prob > confidence_threshold).astype(np.uint8)
            pred_mask = post_process_pseudo_mask(pred_mask, min_size=300, image_size=original_size)  # Giảm từ 500 xuống 300
            
            # Save
            center_frame_name = os.path.basename(os.path.splitext(center_frame_path)[0])
            out_path = os.path.join(output_dir, center_frame_name + '_mask.png')
            cv2.imwrite(out_path, pred_mask * 255)
            saved_count += 1
            
            # Xóa frame đã xử lý
            frame_buffer.pop(center_idx)

    return saved_count