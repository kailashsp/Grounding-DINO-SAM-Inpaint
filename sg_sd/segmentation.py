import cv2
from PIL import Image

from config import SAM_CHECKPOINT_PATH , SAM_MODEL_TYPE, DEVICE
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

class Segment():
    def __init__(self,img_path,bbox) -> None:
        self.img_path =img_path
        self.bbox = bbox

    def segment(self):
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)

        sam.to(device=DEVICE)

        predictor = SamPredictor(sam)

        image_bgr = cv2.imread(self.img_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        input_box = np.array(self.bbox)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )


        mask = Image.fromarray(masks[0, : , :])

        mask.save("mask.png")

        return mask
