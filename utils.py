import numpy as np
import cv2

def get_overlap(mask1, mask2):
    """Calculate the overlap between two masks as the intersection over union."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0
    return np.sum(intersection) / np.sum(union), np.sum(intersection)
def convert_format_body_points(pose_landmarks,image_width,image_height):
    """Convert normalized landmark coordinates to pixel coordinates"""
    pixel_coordinates = []
    if pose_landmarks:
        for landmark in pose_landmarks.landmark:
            x_pixel = int(landmark.x * image_width)
            y_pixel = int(landmark.y * image_height)
            pixel_coordinates.append([x_pixel, y_pixel])
    return pixel_coordinates


def filter_masks(masks, full_body_mask, overlap_threshold=0.5, size_threshold=0.25):
    """Filter out masks based on size, overlap, and containment within the full body mask."""
    full_body_size = np.sum(full_body_mask)
    min_size = full_body_size * size_threshold
    filtered_masks = []
    # First pass: filter by size and prepare for overlap checking
    for mask in masks:
        if np.sum(mask) > min_size:
            filtered_masks.append(mask)
    # Second pass: filter overlapping masks
    keep_mask_indices = set()
    mask_areas = np.array([np.sum(mask) for mask in filtered_masks])
    for i, mask in enumerate(filtered_masks):
        if i in keep_mask_indices:
            continue
        overlapping_masks = []
        for j, other_mask in enumerate(filtered_masks):
            if i != j:
                overlap_ratio, intersection_area = get_overlap(mask, other_mask)
                if overlap_ratio > overlap_threshold or intersection_area == mask_areas[i] or intersection_area == \
                        mask_areas[j]:
                    overlapping_masks.append(j)
        overlapping_masks.append(i)
        largest_mask_index = max(overlapping_masks, key=lambda idx: mask_areas[idx])
        keep_mask_indices.add(largest_mask_index)
    filtered_masks = [filtered_masks[idx] for idx in keep_mask_indices]
    # Third pass: ensure masks are within the full body mask and less than 60%
    contained_masks = []
    for mask in filtered_masks:
        containment = np.sum(np.logical_and(mask, full_body_mask)) / np.sum(mask)
        containment_full_body = np.sum(np.logical_and(mask, full_body_mask)) / np.sum(full_body_mask)
        if containment > 0.9 and containment_full_body < 0.6:
            contained_masks.append(mask)
    return contained_masks

def find_two_largest_masks(masks):
    mask_areas = [np.sum(mask) for mask in masks]
    sorted_indices = np.argsort(mask_areas)[::-1]
    if len(masks) >= 2:
        return [mask_with_largest_contour(masks[sorted_indices[0]]), mask_with_largest_contour(masks[sorted_indices[1]])]

def find_mask_centers(masks):
    centers = []
    for mask in masks:
        moments = cv2.moments(mask.astype(np.uint8)[0])
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            centers.append((cx, cy))
        else:
            centers.append(None)
    return centers
def find_bbox(mask):
    if mask.dtype != np.uint8 or np.max(mask) > 1:
        mask = (mask > 0).astype(np.uint8) * 255
    mask = mask[0,:,:]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y, x = np.where(rows)[0][[0, -1]], np.where(cols)[0][[0, -1]]
    top_left_x = x[0]
    top_left_y = y[0]
    width = x[1] - x[0] + 1
    height = y[1] - y[0] + 1
    return (top_left_x, top_left_y, width, height)
def separate_masks(masks, centers):
    """Decide which mask is upper part and which is lower part"""
    if centers[0][1] < centers[1][1]:
        upper_mask, lower_mask = masks[0], masks[1]
    else:
        upper_mask, lower_mask = masks[1], masks[0]
    upper_bbox = find_bbox(upper_mask)
    lower_bbox = find_bbox(lower_mask)
    return upper_bbox, lower_bbox

def crop_and_save_image(image, bbox, save_path):
    """Crop the image using the bounding box and save the result."""
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(save_path, cropped_image)
    return cropped_image


def mask_with_largest_contour(mask):
    """Isolate the largest contour in a binary mask"""
    mask = mask[0,:,:]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_mask = np.zeros_like(mask)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return largest_contour_mask[np.newaxis, :, :]