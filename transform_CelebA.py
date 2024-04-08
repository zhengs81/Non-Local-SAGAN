import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

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

def process_images(source_dir, target_dir, transformer):
    to_pil = ToPILImage()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    i=0
    for filename in os.listdir(source_dir):
        i += 1
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check for image files
            file_path = os.path.join(source_dir, filename)
            image = Image.open(file_path).convert('RGB')
            transformed_tensor = transformer.transform(True, True, False, True)(image)
            save_image(transformed_tensor,os.path.join(target_dir, filename))
            # transformed_image = to_pil(transformed_tensor)  # Convert the tensor to a PIL image
            # transformed_image.save(os.path.join(target_dir, filename))  # Save the PIL image

        if i % 10000 == 0:
            print(i)

if __name__ == '__main__':
    # Set your source and target directories
    source_directory = 'data/CelebA/img_align_celeba/img_align_celeba'
    target_directory = 'data/transformed_images'
    image_size = 64 # Set this to your desired image size

    # Create an instance of the transformer
    image_transformer = ImageTransformer(image_size)

    # Process the images
    process_images(source_directory, target_directory, image_transformer)
