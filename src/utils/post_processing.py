import numpy as np
import cv2

def keep_largest_connected_component(mask):
    """
    Keeps only the largest connected component in a binary mask.
    Args:
        mask: A binary mask (numpy array).
    Returns:
        A binary mask with only the largest connected component.
    """
    # Keep the largest connected component to remove noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Find the label of the largest component (ignoring the background, label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # Create a new mask with only the largest component
        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest_label] = 255
        mask = clean_mask
    return mask
