from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os
import argparse
import zipfile
import tarfile
import pickle


parser = argparse.ArgumentParser(
    description="Preprocesses the dataset")
parser.add_argument('--file_path', default="",
                    help="path to the downloaded .zip file, "
                         "if left blank the file will be downloaded "
                         "from kaggle first")
parser.add_argument('--extract_path', default='data',
                    help='downloaded data will be extracted to this path')
parser.add_argument('--image_size', default=64, type=int,
                    help='size of each side')
parser.add_argument('--output_path', default='dataset.pckl',
                    help='path where the preprocessed pickled dataset '
                         'will be stored')
parser.add_argument('--verbose', '-v', action='store_true')


def preprocess_img(image_path, annotation_path):
    """
    Loads an image and the corresponding annotation
    Returns a cropped version of the image.
    NOTE: The returned image will always be a square
          with the target dog in the middle

    Arguments:
        image_path {str} -- path to the image file
        annotation_path {str} -- path to the xml file

    Returns:
        PIL.Image
    """
    img = Image.open(image_path)
    annotation = ET.parse(annotation_path)

    # Read information about the bounding box
    root = annotation.getroot()
    xmax = int(root.find("object/bndbox/xmax").text)
    xmin = int(root.find("object/bndbox/xmin").text)
    ymax = int(root.find("object/bndbox/ymax").text)
    ymin = int(root.find("object/bndbox/ymin").text)

    # Crop the image so it becomes a square
    width = xmax - xmin
    width_half = width // 2
    height = ymax - ymin
    height_half = height // 2
    side = max(width, height)

    img = img.crop((xmin + width_half - (side // 2),
                    ymin + height_half - (side // 2),
                    xmin + width_half + (side // 2),
                    ymin + height_half + (side // 2)))

    # assert that every image has 3 channels
    img = img.convert('RGB')
    return img


def image_to_normalized_array(img):
    """
    Converts the given image into a numpy array with each element
    scaled between -1.0 and 1.0

    Arguments:
        img {PIL.Image}

    Returns:
        np.Array
    """
    img = np.array(img, dtype=np.float32)
    img -= 127.5
    img /= 127.5
    return img


def array_to_image(array):
    """
    Converts the given array into a Pillow Image

    Arguments:
        array {np.array}

    Returns:
        PIL.Image
    """
    array = np.copy(array)
    array *= 127.5
    array += 127.5
    return Image.fromarray(array)


def download_dataset():
    """
    runs 'kaggle datasets download -d jessicali9530/stanford-dogs-dataset'
    """
    os.system("kaggle datasets download -d jessicali9530/stanford-dogs-dataset")


def extract_file(file_path, extract_path):
    """
    Extracts the downloaded .zip file (and both nested .tar files)
    to the given directory

    Arguments:
        file_path {str} -- path to the downloaded zip file
        extract_path {str} -- path where the files will be extracted
    """
    with zipfile.ZipFile(file_path, 'r') as f:
        f.extractall(extract_path)

    images = extract_path + "/images.tar"
    annotations = extract_path + "/annotations.tar"

    with tarfile.TarFile(images, 'r') as f:
        f.extractall(extract_path)
    os.remove(images)

    with tarfile.TarFile(annotations, 'r') as f:
        f.extractall(extract_path)
    os.remove(annotations)


def process_single_image(image_path,
                         annotation_path,
                         image_size):
    """
    Loads, resizes and preprocesses a single image

    Arguments:
        image_path {str} -- path to the image file
        annotation_path {str} -- path to the annotation file
        image_size {int} -- size of one side

    Returns:
        np.array
    """
    image = preprocess_img(image_path=image_path,
                           annotation_path=annotation_path)
    image = image.resize((image_size, image_size))
    image = image_to_normalized_array(image)
    return image


def process_all_images(data_path='data/',
                       image_size=64,
                       verbose=True):
    """
    Loads, resizes and preprocesses all images

    Keyword Arguments:
        data_path {str} -- path where the downloaded .zip file got extracted
        image_size {int} -- size of one side (default: {64})
        verbose {bool} -- (default: {True})

    Returns:
        dict -- {breed {str}: np.array}
    """
    images_path = data_path + "/Images"
    annotation_path = data_path + "/Annotation"

    dog_breeds = [os.path.split(i)[1]
                  for i in glob.glob(f'{annotation_path}/*')]

    dataset = {}
    for breed in dog_breeds:
        breed_array = []

        if verbose:
            print(f"Working on '{breed}'...")

        # get list of all files
        filenames = [os.path.split(i)[1]
                     for i in glob.glob(f'{annotation_path}/{breed}/*')]

        # process every image
        for name in filenames:
            img_array = process_single_image(
                image_path=images_path + "/" + breed + "/" + name + ".jpg",
                annotation_path=annotation_path + "/" + breed + "/" + name,
                image_size=image_size)
            breed_array.append(img_array)

        dataset[breed] = np.array(breed_array)

    return dataset


if __name__ == '__main__':
    args = vars(parser.parse_args())

    if args['file_path'] == "":
        download_dataset()
        file_path = "stanford-dogs-dataset.zip"
    else:
        file_path = args['file_path']

    if args['verbose']:
        print("Extracting files ...")

    extract_file(file_path=file_path,
                 extract_path=args['extract_path'])

    dataset = process_all_images(data_path=args['extract_path'],
                                 image_size=args['image_size'],
                                 verbose=args['verbose'])
    with open(args['output_path'], "wb") as f:
        pickle.dump(dataset, f)
