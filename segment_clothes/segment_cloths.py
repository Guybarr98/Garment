import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def segment_clothes(image,body_points,model="vit_h"):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[model](checkpoint="models/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    body_points = np.array(body_points)
    masks = []
    for body_point in body_points:
        segments, _, _ = predictor.predict(point_coords=np.array([body_point]),
                                        point_labels=np.array([1]),
                                        multimask_output=False,)

        masks.append(segments)
    return masks


def segment_body(image,body_points,model="vit_h"):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[model](checkpoint="models/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    body_points = np.array(body_points)
    masks, _, _ = predictor.predict(point_coords=body_points,
                                    point_labels=np.array([1]*len(body_points)),
                                    multimask_output=False,)
    return masks
