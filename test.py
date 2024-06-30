
import os
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

from model import Generator, Discriminator
from processingDataset import ProcessingDataset

class Test:
    def __init__(self):
        self.imgFilePath = './preprocessing_dataset_COL/0045/left/0045_2m_0P_10V_-15H.jpg'
        self.imgInfoPath = './preprocessing_dataset_COL/0045/info/left/0045_2m_0P_10V_-15H.txt'
        # self.imgFilePath = './preprocessing_dataset_CelebA/0/left/anna-jackson-2.jpg'
        # self.imgInfoPath = './preprocessing_dataset_CelebA/0/info/left/anna-jackson-2.txt'

        # self.file_path_l = './training_inputs_COL/left_data.pkl'
        # self.file_path_r = './training_inputs_COL/right_data.pkl'
        self.file_path_l = './training_inputs_COL/left_data.pkl'
        self.file_path_r = './training_inputs_COL/right_data.pkl'

    def loadData(self):
        process_dataset = ProcessingDataset()

        # Load the test image
        eye_landmarks = process_dataset.read_landmarks_from_txt(self.imgInfoPath )
        landmarks = tf.cast(eye_landmarks, tf.float32)
        landmarks = tf.expand_dims(landmarks, axis=0)

        test_image = process_dataset.load_and_preprocess_image(self.imgFilePath)
        test_image = tf.cast(test_image, tf.float32)
        test_image = tf.expand_dims(test_image, axis=0)  # Add batch dimension

        # Example target gaze direction
        gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
        gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)

        gaze_target = tf.expand_dims(gaze_target, axis=0)

        pose = 0.0
        pose = tf.convert_to_tensor(pose, dtype=tf.float32)
        pose = tf.expand_dims(pose, axis=0)

        return test_image, pose, gaze_target, landmarks

    def loadDataFromPKL(self):
        process_dataset = ProcessingDataset()

        batch_size = 64

        test_data = process_dataset.load_pickle_data(self.file_path_l)
        test_dataset = process_dataset.create_dataset(test_data, batch_size)

        for (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks) in test_dataset.take(1):
            break

        img = tf.cast(img, tf.float32)
        img_t = tf.cast(img_t, tf.float32)

        gaze_target = tf.cast(gaze_target, tf.float32)
        gaze_real = tf.cast(gaze_real, tf.float32)

        return img, p, gaze_target, landmarks


    def plotIMGs(self, test_image, generated_image):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
        axes[0].imshow(test_image[0].numpy())
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(generated_image[0].numpy())
        axes[1].set_title("Generated Image")
        axes[1].axis('off')

        plt.savefig('./result_CA.png')
        plt.close(fig)

    def run(self):
        # Define the generator and discriminator
        generator = Generator()
        discriminator = Discriminator()

        # Define the optimizers
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Load the checkpoint
        checkpoint_dir = './training_checkpoints'
        # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(generator=generator,
                                        discriminator=discriminator,
                                        gen_optimizer=generator_optimizer,
                                        disc_optimizer=discriminator_optimizer
                                        )

        # Restore the latest checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        test_image, pose, gaze_target, landmarks = self.loadData()
        # test_image, pose, gaze_target, landmarks = self.loadDataFromPKL()
        
        # Generate the output image
        generated_image = generator(test_image, pose, gaze_target, landmarks, training=False)

        test_image = (test_image + 1.0) / 2.0
        generated_image = (generated_image + 1.0) / 2.0

        self.plotIMGs(test_image, generated_image)

if __name__ == '__main__':
    test = Test()
    test.run()