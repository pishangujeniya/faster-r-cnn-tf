import os.path

from DataGenerator import DataGenerator
from train_frcnn import FasterRCNN


class Main:

    def __init__(self, MAIN_DIR_PATH: str):
        self.MAIN_DIR_PATH = MAIN_DIR_PATH

    @staticmethod
    def download_open_images_dataset_csv(data_generator: DataGenerator):
        data_generator.download_open_images_dataset_csv()

    @staticmethod
    def generate_dataset(data_generator: DataGenerator):
        data_generator.create_subset(class_values={
            "Grape", "Apple", "Potato"
        }, max_images_per_class=150, for_training=True)  # Training Dataset generator
        data_generator.create_subset(class_values={
            "Grape", "Apple", "Potato"}, max_images_per_class=15, for_training=False)  # Testing Dataset generator
        data_generator.download_images(for_training=True)
        data_generator.download_images(for_training=False)
        data_generator.generate_annotation(for_training=True)
        data_generator.generate_annotation(for_training=False)

    def main(self, resume_weights_model_name: str or None, download_open_images_dataset: bool = True,
             generate_subset: bool = True):

        open_images_dataset_dir = os.path.join(self.MAIN_DIR_PATH, "OpenImagesDataset")

        if resume_weights_model_name is not None:
            resume_weights_path = os.path.join(
                open_images_dataset_dir, "models", resume_weights_model_name)
        else:
            resume_weights_path = None

        data_generator = DataGenerator(dataset_dir_path=open_images_dataset_dir)

        if download_open_images_dataset:
            self.download_open_images_dataset_csv(data_generator=data_generator)

        if generate_subset:
            self.generate_dataset(data_generator=data_generator)

        annotation_path = os.path.join(open_images_dataset_dir, "annotation_train.txt")

        model_path = os.path.join(
            open_images_dataset_dir, "models", "model_frcnn.vgg.hdf5")

        base_net_weights = os.path.join(
            open_images_dataset_dir, "vgg16_weights_tf_dim_ordering_tf_kernels.h5")

        config_output_path = os.path.join(
            open_images_dataset_dir, "config.pickle"
        )

        FasterRCNN.train_frcnn(
            annotation_path=annotation_path,
            model_output_path=model_path,
            base_net_weights_path=base_net_weights,
            resume_weights_model_path=resume_weights_path,
            config_output_path=config_output_path,
            num_rois=64,
            is_vgg=True,
            num_epochs=300
        )


if __name__ == '__main__':
    main = Main("./")
    main.main(resume_weights_model_name=None, download_open_images_dataset=True, generate_subset=True)
