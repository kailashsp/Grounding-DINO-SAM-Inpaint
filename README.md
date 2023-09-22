Leveraging Grounding DINO , SAM and stable diffusion to create a pipeline \
to detect the text automatically, segment the image and applying inpaint \
 on the object by a prompt

save the [grounding dino](https://github.com/IDEA-Research/GroundingDINO)  and [SAM](https://github.com/facebookresearch/segment-anything) weight in the weights/ folder

run the [pipeline](sg_sd/test_pipeline.py)

### Requirements
`pip3 install torch torchvision`

follow the install instructions of :
[grounding dino](https://github.com/IDEA-Research/GroundingDINO) 
[SAM](https://github.com/facebookresearch/segment-anything)

`pip install diffusers`