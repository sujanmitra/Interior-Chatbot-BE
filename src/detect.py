import cv2
from ultralytics import YOLO
import numpy as np

from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

def crop_image(img, y1, y2, x1, x2):
    result = np.zeros(img.shape, dtype=np.uint8)
    img[y1:y2, x1:x2] = result[y1:y2, x1:x2]
    return result

def mask_image(img, y1, y2, x1, x2):
    result = np.full(img.shape, 255, dtype=np.uint8)
    img[y1:y2, x1:x2] = result[y1:y2, x1:x2]
    return result

def process_image(img, yolo, prompt):

    results = yolo.track(img, stream=False)

    mask = np.zeros(img.shape, dtype=np.uint8)

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is there
            if box.conf[0] > 0.6:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #crop image
                crop_image(img, y1, y2, x1, x2)

                # mask image
                mask_image(mask, y1, y2, x1, x2)
                    
        print("Image Cropped")
        cv2.imwrite('../testdata/crop.jpg', img)

        print("Image Masked")
        cv2.imwrite('../testdata/mask.jpg', mask)

    # TODO: Train the model on your own dataset
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            ).to("cuda")  # Use GPU for better performance
    pipe.enable_model_cpu_offload()

    image = Image.open("../testdata/sofa2.webp").convert("RGB")
    mask = Image.open("../testdata/mask.jpg").convert("L")  # White areas are to be filled

    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    result.save("../testdata/final.jpg")

def process_img(path, prompt):
    yolo = YOLO('yolov8l.pt')
    process_image(cv2.imread(path), yolo, prompt)