import os
import sys 
import cv2
from config import BASE_PATH, GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CONFIG_PATH
# sys.path.insert(0, os.path.join(BASE_PATH,"GroundingDINO"))       
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate,
    Model,
)

class DetectionModel():
    def __init__(self,img_path,box_threshold=0.35,text_threshold=0.25,text_prompt="") -> None:
        self.img_path = img_path
        # BOX_THRESHOLD = 0.35
        # TEXT_THRESHOLD = 0.25
        # TEXT_PROMPT = "person . background"
        self.box_threshold=box_threshold
        self.text_threshold=text_threshold
        self.text_prompt=text_prompt

        self.grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
    def detect(self):

        # IMAGE_PATH = "/home/kailash/SG_SD/data/54166371.jpeg"
        image_bgr = cv2.imread(self.img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image_source, image = load_image(IMAGE_PATH)
        # detect objects
        detections, phrases = self.grounding_dino_model.predict_with_caption(
            image=image_bgr,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        detections.class_id = phrases
        print(detections.xyxy,phrases)

        for bbox in detections.xyxy:
            print(bbox)
            x1,y1,x2,y2 = map(int,bbox)
            cv2.rectangle(img=image_bgr,pt1=(x1,y1),pt2=(x2,y2),color=(255,0,0))
            cv2.imwrite("detect.png",image_bgr)

        return detections.xyxy

if __name__ == "__main__":
    det = DetectionModel(img_path="/home/kailash/SG_SD/data/54166371.jpeg",text_prompt="person")
    det.detect()