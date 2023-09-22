from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# image = Image.open("/home/kailash/SG_SD/data/54166371.jpeg")

class StableDiffusionInpaint:
    def __init__(self,img_path,mask_img,prompt="") -> None:
        self.img_path = img_path
        self.mask_img = mask_img
        self.prompt = prompt
    def inpaint(self):

        # mask = Image.open("mask.png")

        image = Image.open(self.img_path)
        image  = image.resize((512,512))
        mask = self.mask_img.resize((512,512))

        pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",)
                                                            # torch_dtype=torch.float16)
        # pipe.to("cuda")
        # pipe.enable_attention_slicing() 
        # prompt ="turn the shirt to red color"

        output = pipe(prompt=self.prompt, image=image, mask_image= mask).images[0]

        # output_img = Image.fromarray(output)

        output.save("inpaint.png")