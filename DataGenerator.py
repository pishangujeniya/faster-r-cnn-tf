import wget
import os
import pandas as pd
from concurrent import futures
from typing import List, Set, Any
import cv2
import sys
from matplotlib import pyplot as plt


class DataGenerator:
    OPEN_IMAGES_DATASET_DIR = "OpenImagesDataset"
    BBOX_CSV = "oidv6-train-annotations-bbox.csv"
    IMAGES_LINKS_CSV = "train-images-boxable-with-rotation.csv"
    CLASS_LABELS_CSV = "class-descriptions-boxable.csv"
    SUBSET_IMAGES_TRAIN_FILE_NAME = "subset_images_train.csv"
    SUBSET_BBOX_TRAIN_FILE_NAME = "subset_bbox_train.csv"
    SUBSET_IMAGES_TEST_FILE_NAME = "subset_images_test.csv"
    SUBSET_BBOX_TEST_FILE_NAME = "subset_bbox_test.csv"
    ANNOTATION_TRAIN_CSV = "annotation_train.csv"
    ANNOTATION_TEST_CSV = "annotation_test.csv"
    ANNOTATION_TRAIN_TXT = "annotation_train.txt"
    ANNOTATION_TEST_TXT = "annotation_test.txt"

    class_labels_df: pd.DataFrame = None
    bbox_df: pd.DataFrame = None
    image_links_df: pd.DataFrame = None
    annotation_df: pd.DataFrame = None

    def __init__(self, dataset_dir_path: str):
        self.OPEN_IMAGES_DATASET_DIR = dataset_dir_path
        if not os.path.exists(self.OPEN_IMAGES_DATASET_DIR):
            os.mkdir(path=self.OPEN_IMAGES_DATASET_DIR)

    def download_open_images_dataset_csv(self):
        # Collecting Data from https://opensource.google/projects/open-images-dataset
        if not os.path.exists(self.OPEN_IMAGES_DATASET_DIR + os.path.sep + self.BBOX_CSV):
            print("Downloading of BBOX csv file starting...")
            wget.download(
                url='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
                out=self.OPEN_IMAGES_DATASET_DIR
            )
            print("Downloading of BBOX csv file completed")
        if not os.path.exists(self.OPEN_IMAGES_DATASET_DIR + os.path.sep + self.IMAGES_LINKS_CSV):
            print("Downloading of Image Details csv file starting...")
            wget.download(
                url='https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv',
                out=self.OPEN_IMAGES_DATASET_DIR
            )
            print("Downloading of Image Details csv file completed")
        if not os.path.exists(self.OPEN_IMAGES_DATASET_DIR + os.path.sep + self.CLASS_LABELS_CSV):
            print("Downloading of Class Labels csv file starting...")
            wget.download(
                url="https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
                out=self.OPEN_IMAGES_DATASET_DIR
            )
            print("Downloading of Class Labels csv file completed")

    def read_class_labels_csv(self):
        self.class_labels_df = pd.read_csv(self.OPEN_IMAGES_DATASET_DIR + os.path.sep + self.CLASS_LABELS_CSV)
        self.class_labels_df.columns = ["label_value", "class_value"]

    def read_image_links_csv(self):
        self.image_links_df = pd.read_csv(self.OPEN_IMAGES_DATASET_DIR + os.path.sep + self.IMAGES_LINKS_CSV)

    def read_bbox_df_csv(self):
        self.bbox_df = pd.read_csv(self.OPEN_IMAGES_DATASET_DIR + os.path.sep + self.BBOX_CSV)

    def read_annotation_train_df_csv(self):
        self.annotation_df = pd.read_csv(os.path.join(self.OPEN_IMAGES_DATASET_DIR, self.ANNOTATION_TRAIN_CSV))

    def read_annotation_test_df_csv(self):
        self.annotation_df = pd.read_csv(os.path.join(self.OPEN_IMAGES_DATASET_DIR, self.ANNOTATION_TEST_CSV))

    def get_class_value(self, label_values: Set[str]) -> List[str]:
        if self.class_labels_df is None:
            self.read_class_labels_csv()
        return self.class_labels_df.loc[self.class_labels_df["label_value"].isin(label_values)]["class_value"].values

    def get_label_value(self, class_values: Set[str]) -> List[str]:
        if self.class_labels_df is None:
            self.read_class_labels_csv()
        return self.class_labels_df.loc[self.class_labels_df["class_value"].isin(class_values)]["label_value"].values

    def create_subset(self, class_values: Set[str], max_images_per_class: int = 5, for_training: bool = True):
        if self.class_labels_df is None:
            self.read_class_labels_csv()
        if self.bbox_df is None:
            self.read_bbox_df_csv()
        if self.image_links_df is None:
            self.read_image_links_csv()

        subset_images_file_name = ""
        subset_bbox_file_name = ""
        start_offset = 0
        end_offset = 0
        if for_training:
            start_offset = 0
            end_offset = start_offset + max_images_per_class
            subset_images_file_name = self.SUBSET_IMAGES_TRAIN_FILE_NAME
            subset_bbox_file_name = self.SUBSET_BBOX_TRAIN_FILE_NAME
        else:
            start_offset = max_images_per_class + 1
            end_offset = start_offset + max_images_per_class
            subset_images_file_name = self.SUBSET_IMAGES_TEST_FILE_NAME
            subset_bbox_file_name = self.SUBSET_BBOX_TEST_FILE_NAME

        label_values = self.get_label_value(class_values=class_values)

        bbox_subset_df: pd.DataFrame = pd.DataFrame()
        for label in label_values:
            bbox_of_label_df = self.bbox_df.loc[self.bbox_df["LabelName"] == label]
            image_label_ids = bbox_of_label_df["ImageID"].unique()[start_offset:end_offset]
            bbox_subset_df = pd.concat(
                [bbox_subset_df, pd.DataFrame(bbox_of_label_df.loc[bbox_of_label_df["ImageID"].isin(image_label_ids)])],
                axis=0)

        bbox_subset_df["class_value"] = None
        bbox_subset_df["class_value"] = bbox_subset_df.apply(lambda row: self.get_class_value({row["LabelName"]})[0],
                                                             axis=1)

        image_ids = bbox_subset_df['ImageID'].unique()
        image_links_subset_df = pd.DataFrame(self.image_links_df.loc[self.image_links_df["ImageID"].isin(image_ids)])

        image_links_subset_df.to_csv(
            path_or_buf=os.path.join(self.OPEN_IMAGES_DATASET_DIR, subset_images_file_name), sep=",", index=False)
        bbox_subset_df.to_csv(
            path_or_buf=os.path.join(self.OPEN_IMAGES_DATASET_DIR, subset_bbox_file_name), sep=",", index_label="No",
            index=True)

    def download_images(self, max_parallel=8, for_training: bool = True):
        print("Downloading images for " + str("training" if for_training else "testing") + " is starting...")
        if for_training:
            subset_images_file_name = self.SUBSET_IMAGES_TRAIN_FILE_NAME
            out_dir = "train"
        else:
            subset_images_file_name = self.SUBSET_IMAGES_TEST_FILE_NAME
            out_dir = "test"

        out_path: str = os.path.join(self.OPEN_IMAGES_DATASET_DIR, "images", out_dir)
        in_csv_path: str = os.path.join(self.OPEN_IMAGES_DATASET_DIR, subset_images_file_name)
        subset_images_details_df = pd.read_csv(in_csv_path)
        images_links = subset_images_details_df["OriginalURL"].values

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            for image_link in images_links:
                def download_single_image() -> Any:
                    try:
                        wget.download(url=image_link, out=out_path)
                    finally:
                        return True

                executor.submit(
                    download_single_image()
                )

    def generate_annotation(self, for_training: bool = True):
        print("Generating annotation for " + str("training" if for_training else "testing") + " is starting...")
        if for_training:
            annotation_txt_file_name = self.ANNOTATION_TRAIN_TXT
            subset_images_file_name = self.SUBSET_IMAGES_TRAIN_FILE_NAME
            subset_bbox_file_name = self.SUBSET_BBOX_TRAIN_FILE_NAME
            out_dir = "train"
        else:
            annotation_txt_file_name = self.ANNOTATION_TEST_TXT
            subset_images_file_name = self.SUBSET_IMAGES_TEST_FILE_NAME
            subset_bbox_file_name = self.SUBSET_BBOX_TEST_FILE_NAME
            out_dir = "test"
        downloaded_images_dir: str = os.path.join(self.OPEN_IMAGES_DATASET_DIR, "images", out_dir)
        subset_bbox_csv_path: str = os.path.join(self.OPEN_IMAGES_DATASET_DIR, subset_bbox_file_name)
        subset_images_csv_path: str = os.path.join(self.OPEN_IMAGES_DATASET_DIR, subset_images_file_name)

        subset_images_df = pd.read_csv(subset_images_csv_path)
        subset_bbox_df = pd.read_csv(subset_bbox_csv_path)

        annotations_df = pd.DataFrame()
        for ind in subset_images_df.index:
            image_id = subset_images_df["ImageID"][ind]
            file_name = str(subset_images_df["OriginalURL"][ind]).split("/")[-1]
            if file_name in os.listdir(os.path.abspath(downloaded_images_dir)):
                bbox_rows = pd.DataFrame(subset_bbox_df[subset_bbox_df["ImageID"] == image_id])
                bbox_rows["ImagePath"] = None
                bbox_rows["ImagePath"] = os.path.abspath(os.path.join(downloaded_images_dir, file_name))
                annotations_df = pd.concat(
                    [annotations_df, bbox_rows],
                    axis=0
                )

        annotations_df = annotations_df.set_index('No')

        if for_training:
            annotations_df.to_csv(os.path.join(self.OPEN_IMAGES_DATASET_DIR, self.ANNOTATION_TRAIN_CSV))
        else:
            annotations_df.to_csv(os.path.join(self.OPEN_IMAGES_DATASET_DIR, self.ANNOTATION_TEST_CSV))

        with open(os.path.join(self.OPEN_IMAGES_DATASET_DIR, annotation_txt_file_name), "w+") as f:
            for idx, row in annotations_df.iterrows():
                sys.stdout.write(str(idx) + '\r')
                sys.stdout.flush()
                img = cv2.imread(row["ImagePath"])
                height, width = img.shape[:2]
                x1 = int(row['XMin'] * width)
                x2 = int(row['XMax'] * width)
                y1 = int(row['YMin'] * height)
                y2 = int(row['YMax'] * height)
                f.write(
                    row["ImagePath"] + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + row[
                        'class_value'] + '\n')

    def plot_bbox(self, img_id: str):
        if self.annotation_df is None:
            self.read_annotation_train_df_csv()

        img_url = self.annotation_df[self.annotation_df["ImageID"] == img_id]['ImagePath'].values[0]
        # img = cv2.imread(img_url)
        img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        bboxs = self.annotation_df[self.annotation_df["ImageID"] == img_id]
        for index, row in bboxs.iterrows():
            xmin = row['XMin']
            xmax = row['XMax']
            ymin = row['YMin']
            ymax = row['YMax']
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            class_value = row['class_value']
            print(f"Coordinates: {xmin, ymin}, {xmax, ymax}")
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_value, (xmin, ymin - 10), font, 1.5, (0, 255, 0), 5)
        plt.figure(figsize=(15, 10))
        plt.title('Image with Bounding Box')
        plt.imshow(img)
        plt.axis("on")
        plt.show()
