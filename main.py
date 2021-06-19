import os.path

from DataGenerator import DataGenerator
from train_frcnn import FasterRCNN
from test_frcnn import FasterRCNN


class Main:

    def __init__(self, MAIN_DIR_PATH: str):
        self.MAIN_DIR_PATH = MAIN_DIR_PATH

    @staticmethod
    def download_open_images_dataset_csv(data_generator: DataGenerator):
        data_generator.download_open_images_dataset_csv()

    def main(self, resume_weights_model_name: str or None, download_open_images_dataset: bool = True,
             generate_subset: bool = True, download_subset_images: bool = True, generate_annotation: bool = True):

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
            data_generator.create_subset(class_values={
                "Grape", "Apple", "Potato"
            }, max_images_per_class=150, for_training=True)  # Training Dataset generator
            data_generator.create_subset(class_values={
                "Grape", "Apple", "Potato"}, max_images_per_class=15, for_training=False)  # Testing Dataset generator

        if download_subset_images:
            data_generator.download_images(for_training=True)
            data_generator.download_images(for_training=False)

        if generate_annotation:
            data_generator.generate_annotation(for_training=True)
            data_generator.generate_annotation(for_training=False)

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

    def test(self, trained_model_path: str, config_path: str, input_images_dir_path: str, output_dir_path: str):
        FasterRCNN.test_frcnn(config_file_path=config_path, trained_model_path=trained_model_path,
                              input_images_dir_path=input_images_dir_path, output_images_dir_path=output_dir_path)


if __name__ == '__main__':
    main = Main("./")
    main.main(resume_weights_model_name=None, download_open_images_dataset=True, generate_subset=True,
              download_subset_images=True, generate_annotation=True)

    main.test(
        trained_model_path=r"C:\Users\pitbox\Documents\GitHub\faster-r-cnn-tf\OpenImagesDataset\models\model_frcnn.vgg_0218.hdf5",
        config_path=r"C:\Users\pitbox\Documents\GitHub\faster-r-cnn-tf\OpenImagesDataset\config.pickle",
        input_images_dir_path=r"C:\Users\pitbox\Documents\GitHub\faster-r-cnn-tf\OpenImagesDataset\images\test",
        output_dir_path=r"C:\Users\pitbox\Documents\GitHub\faster-r-cnn-tf\OpenImagesDataset\images\output")
