import argparse
from pathlib import Path
import random
import os
import shutil


def get_indices(num_examples, train_pct):
    indices = list(range(num_examples))
    random.shuffle(indices)

    num_train = int(num_examples * train_pct / 100)
    num_valid = int(num_examples * (100 - train_pct) / 200)

    train_indices = indices[:num_train]
    valid_indices = indices[num_train:num_train + num_valid]
    test_indices = indices[num_train + num_valid:]

    return train_indices, valid_indices, test_indices


def copy_dataset(list_indices, source_files, destination_dir):
    sorted_source_files = sorted(source_files, key=lambda f: f.stem)
    for idx in list_indices:
        source_path = sorted_source_files[idx]
        destination_path = destination_dir / source_path.name
        shutil.copy(str(source_path), str(destination_path))

    print("Copied {} files to {}".format(len(list_indices), destination_dir))


def main():
    parser = argparse.ArgumentParser(description="Create datasets to train YOLO")
    parser.add_argument("datadir", help="Directory containing the subfolders frames and labels", type=str)
    parser.add_argument("outputdir", help="Directory that will contain the datasets", type=str)
    parser.add_argument("train_pct", help="Percentage to use to create the training dataset", type=int)

    args = parser.parse_args()

    datadir = Path(args.datadir)
    outputdir = Path(args.outputdir)
    if not outputdir.exists():
        os.mkdir(str(outputdir))
    train_pct = args.train_pct

    frames_dir = datadir / "images/train"
    labels_dir = datadir / "labels/train"

    image_files = list(frames_dir.glob("*.jpg"))
    labels_files = list(labels_dir.glob("*.txt"))

    num_image_files = len(image_files)
    num_labels_files = len(labels_files)

    if num_image_files != num_labels_files:
        print("The number of jpg files and txt files must be equal")
        return

    train_indices, valid_indices, test_indices = get_indices(num_image_files, train_pct)

    images_dir = outputdir / "images"
    labels_dir = outputdir / "labels"

    os.mkdir(str(images_dir))
    os.mkdir(str(labels_dir))

    image_train_dir = images_dir / "train"
    image_val_dir = images_dir / "val"
    image_test_dir = images_dir / "test"

    os.mkdir(str(image_train_dir))
    os.mkdir(str(image_val_dir))
    os.mkdir(str(image_test_dir))

    copy_dataset(train_indices, image_files, image_train_dir)
    copy_dataset(valid_indices, image_files, image_val_dir)
    copy_dataset(test_indices, image_files, image_test_dir)

    label_train_dir = labels_dir / "train"
    label_val_dir = labels_dir / "val"
    label_test_dir = labels_dir / "test"

    os.mkdir(str(label_train_dir))
    os.mkdir(str(label_val_dir))
    os.mkdir(str(label_test_dir))

    copy_dataset(train_indices, labels_files, label_train_dir)
    copy_dataset(valid_indices, labels_files, label_val_dir)
    copy_dataset(test_indices, labels_files, label_test_dir)


if __name__ == "__main__":
    main()
