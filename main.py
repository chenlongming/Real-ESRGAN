import os
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
from tqdm import tqdm


def main() -> int:
    scale = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    for i, image in tqdm(enumerate(os.listdir("inputs"))):
        image = Image.open(f"inputs/{image}").convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(f'results/{i}_x{scale}.png')
        

if __name__ == '__main__':
    main()