import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import torchvision.datasets as dsets

class ImageTransformer:
    def __init__(self, image_size):
        self.imsize = image_size

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        transform = transforms.Compose(options)
        return transform
    

def load_lsun(source_dir,transformer):
    transforms = transformer.transform(True, True, False, False)
    dataset = dsets.LSUN(source_dir, classes=['church_outdoor_train'], transform=transforms)
    return dataset

if __name__ == '__main__':
    source_dir = 'data/LSUN'
    target_dir = 'data/transformed_images_LSUN'
    image_size = 64

    image_transformer = ImageTransformer(image_size)

    dataset=load_lsun(source_dir,image_transformer)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for i,image in enumerate(dataset):
        save_image(image[0],target_dir+'/'+str(i)+'.png')
