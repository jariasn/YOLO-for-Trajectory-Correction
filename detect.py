import cv2 as cv
import os
import argparse
from pathlib import Path
import sys


def is_image_blurry(image, threshold):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Calculate the variance of the Laplacian
    fm = cv.Laplacian(gray, cv.CV_64F).var()
    # Check if the image is blurry based on the threshold
    return fm < threshold


def main(input_folder, blurry_folder, not_blurry_folder, threshold):
    # Convert input paths to Path objects
    data_folder = Path(input_folder)
    blurry_folder = Path(blurry_folder)
    good_folder = Path(not_blurry_folder)

    # Check if the provided paths are valid directories
    if not data_folder.is_dir():
        print(f"{data_folder} is not a directory")
        sys.exit(1)

    if not blurry_folder.is_dir():
        print(f"{blurry_folder} is not a directory")
        sys.exit(1)

    if not good_folder.is_dir():
        print(f"{good_folder} is not a directory")
        sys.exit(1)

    # Recognize jpg or jpeg images
    image_extensions = ('.jpg', '.jpeg')
    images = [f for f in data_folder.glob('*') if f.suffix.lower() in image_extensions]

    # Go through all images in the data folder
    for image_path in images:
        print(f"Processing image {image_path}")

        # Load the image
        image = cv.imread(str(image_path))

        # Check if the image is blurry
        if is_image_blurry(image, threshold):
            # Move blurry image to the blurry folder
            blurry_path = blurry_folder.joinpath(image_path.name)
            cv.imwrite(str(blurry_path), image)
        else:
            # Move non-blurry image to the good folder
            good_path = good_folder.joinpath(image_path.name)
            cv.imwrite(str(good_path), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eliminate blurry pictures")
    parser.add_argument("inputFolder", help="Path of folder containing images to classify", type=str)
    parser.add_argument("blurryFolder", help="Path of folder where blurry images will be sent", type=str)
    parser.add_argument("notBlurryFolder", help="Path of folder where non-blurry images will be sent", type=str)
    parser.add_argument("--threshold", help="Threshold for blurry detection, default is 200", type=int, default=200)
    args = parser.parse_args()

    main(args.inputFolder, args.blurryFolder, args.notBlurryFolder, args.threshold)