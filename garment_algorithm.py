from body_detection.mediapipe_body_detection import *
from segment_clothes.segment_cloths import *
from utils import *
def main(image_path):
    image = cv2.imread(image_path)
    body_points = detect_body_landmarks(image)
    width, height = image.shape[1], image.shape[0]
    body_points_sam_format = convert_format_body_points(body_points, width, height)
    segment_full_body = segment_body(image,body_points_sam_format)
    segments = segment_clothes(image,body_points_sam_format)
    final_segments = filter_masks(segments,segment_full_body)
    upper_and_lower_masks = find_two_largest_masks(final_segments)
    final_center_points = find_mask_centers(upper_and_lower_masks)
    final_bbox = separate_masks(upper_and_lower_masks,final_center_points)
    upper_bbox, lower_bbox = final_bbox
    upper_image = crop_and_save_image(image, upper_bbox, 'upper_body.jpg')
    lower_image = crop_and_save_image(image, lower_bbox, 'lower_body.jpg')

if "__main__" == __name__:
    main("test_images/person_A.jpg")