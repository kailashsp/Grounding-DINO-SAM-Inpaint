import os
import sys
import logging

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
                            
GROUNDING_DINO_CONFIG_PATH = os.path.join(BASE_PATH,"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

logging.info(f"{GROUNDING_DINO_CONFIG_PATH} ; exist: {os.path.isfile(GROUNDING_DINO_CONFIG_PATH)}")

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(BASE_PATH, "weights", "G_DINO","groundingdino_swint_ogc.pth")

logging.info(f"{GROUNDING_DINO_CHECKPOINT_PATH} ; exist: {os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH)}")

SAM_CHECKPOINT_PATH = os.path.join(BASE_PATH, "weights", "SAM","sam_vit_l_0b3195.pth")

SAM_MODEL_TYPE = "vit_l"

DEVICE = "cpu"

logging.info(f"{SAM_CHECKPOINT_PATH} ; exist: {os.path.isfile(SAM_CHECKPOINT_PATH)}")