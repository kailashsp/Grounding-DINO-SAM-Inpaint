from object_detector import DetectionModel
from segmentation import Segment
from inpaint import StableDiffusionInpaint


path = "/home/kailash/SG_SD/data/Untitled.jpeg" # input images

det = DetectionModel(img_path=path,text_prompt="router") # text_prompt denotes the object to detect from image
box = det.detect()

seg = Segment(img_path=path,bbox=box) #segments the object from images

mask = seg.segment()

prompt = "represent the router with slow speed" # prompt on how to change the object
sd = StableDiffusionInpaint(img_path=path,mask_img=mask,prompt=prompt)

sd.inpaint()