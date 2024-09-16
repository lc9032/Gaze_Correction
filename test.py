
import os
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt # type: ignoreabout:blank#blocked
import numpy as np # type: ignore

from Model.model import Generator, Discriminator
from processingDataset import ProcessingDataset

from tensorflow.keras import backend as K

from Model.transformation import Transformation

import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_DATA_DIR_PKL_SWITCH = 1

class Test:
    def __init__(self):

        # Define the generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()

        # Define the optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.imgFilePath = './DataSets/preprocessing_dataset_COL_0814/0032/right/0032_2m_0P_-10V_5H.jpg'
        self.imgInfoPath = './DataSets/preprocessing_dataset_COL_0814/0032/info/right/0032_2m_0P_-10V_5H.txt'
        # self.imgFilePath = './preprocessing_dataset_CelebA/0/left/anna-jackson-2.jpg'
        # self.imgInfoPath = './preprocessing_dataset_CelebA/0/info/left/anna-jackson-2.txt'


        self.pkl_folder_path = './DataSets/training_inputs_COL_0824'
        # self.pkl_folder_path = './DataSets/training_inputs_U2_0824'


        # self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_0809'
        self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_N_0912_2'
        # self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_N_0813'
        self.trans = Transformation()

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

        batch_size = 1

        # test_data = process_dataset.load_pickle_data(self.file_path_l)
        test_data = process_dataset.load_pickle_data(self.pkl_folder_path)

        test_dataset = process_dataset.create_dataset(test_data[0], batch_size)

        for (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap in test_dataset.take(1):
            break

        img = tf.cast(img, tf.float32)
        img_t = tf.cast(img_t, tf.float32)

        gaze_target = tf.cast(gaze_target, tf.float32)
        gaze_real = tf.cast(gaze_real, tf.float32)

        return img, p, gaze_target, landmarks, img_t


    def plotIMGs(self, test_image, generated_image):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
        axes[0].imshow(test_image[0].numpy())
        axes[0].set_title("Original Image", fontsize=36)
        axes[0].axis('off')

        axes[1].imshow(generated_image[0].numpy())
        axes[1].set_title("Generated Image", fontsize=36)
        axes[1].axis('off')

        plt.savefig('./result_CA.png')
        plt.close(fig)

    def plotIMGs_target(self, test_image, target_image, generated_image):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
        axes[0].imshow(test_image[0].numpy())
        axes[0].set_title("Original Image", fontsize=22)
        axes[0].axis('off')

        axes[1].imshow(target_image[0].numpy())
        axes[1].set_title("Target Image", fontsize=22)
        axes[1].axis('off')

        axes[2].imshow(generated_image[0].numpy())
        axes[2].set_title("Generated Image", fontsize=22)
        axes[2].axis('off')

        plt.savefig('./result_CA.png')
        plt.close(fig)


    def calculate_mse(self, original_image, generated_image):
        return tf.reduce_mean(tf.square(original_image - generated_image))
    
    def calculate_average_mse(self):
        total_mse = 0
        total_images = 0

        # Assuming the dataset is loaded from a PKL file
        process_dataset = ProcessingDataset()
        test_data = process_dataset.load_pickle_data(self.pkl_folder_path)
        test_dataset = process_dataset.create_dataset(test_data[0], batch_size=1)

        # Iterate through the dataset
        for (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap in test_dataset:
            corr_flow, corr_brightness_map = self.generator(img, p, gaze_target, landmarks, training=False)
            warped_image = self.trans.apply_transformation(corr_flow, img)
            generated_image = self.trans.apply_lcm(warped_image, corr_brightness_map)

            # Calculate MSE for this image
            mse = self.calculate_mse(img_t, generated_image)
            total_mse += mse.numpy()
            total_images += 1

            print(f'Image: {total_images}, MSE: {mse.numpy()}')

        # Compute the average MSE
        average_mse = total_mse / total_images
        print(f'Average MSE over the dataset: {average_mse}')

        return average_mse
    

    def calculate_mse_for_angles(self, save_path='./mse_angles.png'):
        angle_bins = np.arange(-30, 31, 10)  # Adjust the range and bin size as needed
        mse_values = {angle: [] for angle in angle_bins}

        process_dataset = ProcessingDataset()
        test_data = process_dataset.load_pickle_data(self.pkl_folder_path)
        test_dataset = process_dataset.create_dataset(test_data[0], batch_size=1)

        for (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap in test_dataset:
            corr_flow, corr_brightness_map = self.generator(img, p, gaze_target, landmarks, training=False)
            warped_image = self.trans.apply_transformation(corr_flow, img)
            generated_image = self.trans.apply_lcm(warped_image, corr_brightness_map)

            # Calculate MSE for this image
            mse = self.calculate_mse(img_t, generated_image).numpy()

            # Determine the angle bin
            # gaze_angle = np.degrees(np.arctan2(gaze_real[0, 0, 1], gaze_real[0, 0, 0]))  # Assuming gaze_real is in radians
            # print(gaze_real[0, 0, 1],'      ', gaze_real[0, 0, 0],  '\n')

            h_degree = gaze_real[0, 0, 1].numpy()  # Horizontal angle
            v_degree = gaze_real[0, 0, 0].numpy()  # Vertical angle
            gaze_angle = v_degree*2#(h_degree*h_degree + v_degree*v_degree) ** 0.5

            print(gaze_angle)


            closest_bin = min(angle_bins, key=lambda x: abs(x - gaze_angle))
            mse_values[closest_bin].append(mse)

        # Calculate the average MSE for each angle bin
        average_mse = {angle: np.mean(mse_values[angle]) if mse_values[angle] else 0 for angle in mse_values}

        # Plotting the results
        angles = sorted(average_mse.keys())
        mse_values = [average_mse[angle] for angle in angles]

        plt.figure(figsize=(10, 6))
        plt.plot(angles, mse_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Gaze Angle (degrees)')
        plt.ylabel('MSE')
        plt.title('MSE vs. Gaze Angle')
        plt.grid(True)
        # plt.show()
        plt.savefig(save_path)

        return average_mse
    

    def calculate_mse_for_multiple_checkpoints(self, save_path='./mse_comparison.png'):

        # checkpoint_paths = ['./TrainingCheckPoints/training_checkpoints_N_0829',
        #                    './TrainingCheckPoints/training_checkpoints_S_0829',
        #                    './TrainingCheckPoints/training_checkpoints_SN_0829']
        
        # checkpoint_paths = ['./TrainingCheckPoints/training_checkpoints_S_0831_SIN',
        #                    './TrainingCheckPoints/training_checkpoints_S_0831_BI',
        #                    './TrainingCheckPoints/training_checkpoints_S_0831_BI55']
        
        checkpoint_paths = ['./TrainingCheckPoints/training_checkpoints_N_0902',
                           './TrainingCheckPoints/training_checkpoints_N_0902_gan']

        angle_bins = np.arange(-30, 31, 10)  # Adjust the range and bin size as needed
        all_mse_values = {checkpoint: {angle: [] for angle in angle_bins} for checkpoint in checkpoint_paths}
        overall_avg_mse = {}

        process_dataset = ProcessingDataset()

        for checkpoint_path in checkpoint_paths:
            # Load the checkpoint
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                            discriminator=self.discriminator,
                                            gen_optimizer=self.generator_optimizer,
                                            disc_optimizer=self.discriminator_optimizer)
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

            test_data = process_dataset.load_pickle_data(self.pkl_folder_path)
            test_dataset = process_dataset.create_dataset(test_data[0], batch_size=1)

            total_mse = 0
            total_images = 0

            for (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap in test_dataset:
                corr_flow, corr_brightness_map = self.generator(img, p, gaze_target, landmarks, training=False)
                warped_image = self.trans.apply_transformation(corr_flow, img)
                generated_image = self.trans.apply_lcm(warped_image, corr_brightness_map)

                # Calculate MSE for this image
                mse = self.calculate_mse(img_t, generated_image).numpy()

                # Accumulate total MSE and image count
                total_mse += mse
                total_images += 1

                # Determine the angle bin
                h_degree = gaze_real[0, 0, 1].numpy()  # Horizontal angle
                v_degree = gaze_real[0, 0, 0].numpy()  # Vertical angle
                gaze_angle = v_degree * 2  # Example: Adjust according to your needs

                closest_bin = min(angle_bins, key=lambda x: abs(x - gaze_angle))
                all_mse_values[checkpoint_path][closest_bin].append(mse)

            # Calculate the overall average MSE for this checkpoint
            avg_mse_checkpoint = total_mse / total_images if total_images > 0 else 0
            overall_avg_mse[checkpoint_path] = avg_mse_checkpoint

            print(f'Checkpoint: {checkpoint_path}, Average MSE: {avg_mse_checkpoint}')

        # Calculate the average MSE for each angle bin for each checkpoint
        average_mse = {checkpoint: {angle: np.mean(all_mse_values[checkpoint][angle]) if all_mse_values[checkpoint][angle] else 0
                                    for angle in all_mse_values[checkpoint]} for checkpoint in checkpoint_paths}

        # Plotting the results
        plt.figure(figsize=(10, 6))

        # Plotting the results
        plt.figure(figsize=(10, 6))

        # Adjust text sizes
        plt.rc('font', size=11)          # Default text size
        plt.rc('axes', titlesize=12)     # Title size
        plt.rc('axes', labelsize=16)     # Label size (x and y labels)
        plt.rc('xtick', labelsize=12)    # X-axis tick size
        plt.rc('ytick', labelsize=12)    # Y-axis tick size
        plt.rc('legend', fontsize=12)    # Legend font size

        for checkpoint_path in checkpoint_paths:

            angles = sorted(average_mse[checkpoint_path].keys())
            mse_values = [average_mse[checkpoint_path][angle] for angle in angles]

            if checkpoint_path == './TrainingCheckPoints/training_checkpoints_N_0902':
                plt.plot(angles, mse_values, marker='o', linestyle='-', label=os.path.basename('Training without Discriminator'))
            else:
                plt.plot(angles, mse_values, marker='o', linestyle='-', label=os.path.basename('Training with Discriminator'))

        plt.xlabel('Gaze Angle (degrees)')
        plt.ylabel('MSE')
        plt.title('MSE Comparison of Gaze Correction Models Across Different Gaze Angles for different Training Methods')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)

        return average_mse, overall_avg_mse
    
    def calculate_mse_for_headpose(self, save_path='./headpose_mse_comparison.png'):
        # checkpoint_paths = ['./TrainingCheckPoints/training_checkpoints_N_0829',
        #                     './TrainingCheckPoints/training_checkpoints_S_0829',
        #                     './TrainingCheckPoints/training_checkpoints_SN_0829']
        
        # checkpoint_paths = ['./TrainingCheckPoints/training_checkpoints_S_0831_SIN',
        #                    './TrainingCheckPoints/training_checkpoints_S_0831_BI',
        #                    './TrainingCheckPoints/training_checkpoints_S_0831_BI55']
        
        checkpoint_paths = ['./TrainingCheckPoints/training_checkpoints_N_0909_WWW',
                    './TrainingCheckPoints/training_checkpoints_N_0909_WWW']

        angle_bins = np.arange(-30, 31, 15)  # Adjust the range and bin size as needed
        all_mse_values = {checkpoint: {angle: [] for angle in angle_bins} for checkpoint in checkpoint_paths}
        overall_avg_mse = {}

        process_dataset = ProcessingDataset()

        for checkpoint_path in checkpoint_paths:
            # Load the checkpoint
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                            discriminator=self.discriminator,
                                            gen_optimizer=self.generator_optimizer,
                                            disc_optimizer=self.discriminator_optimizer)
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

            test_data = process_dataset.load_pickle_data(self.pkl_folder_path)
            test_dataset = process_dataset.create_dataset(test_data[0], batch_size=1)

            total_mse = 0
            total_images = 0

            for (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap in test_dataset:
                corr_flow, corr_brightness_map = self.generator(img, p, gaze_target, landmarks, training=False)
                warped_image = self.trans.apply_transformation(corr_flow, img)
                generated_image = self.trans.apply_lcm(warped_image, corr_brightness_map)

                # Calculate MSE for this image
                mse = self.calculate_mse(img_t, generated_image).numpy()

                # Accumulate total MSE and image count
                total_mse += mse
                total_images += 1

                # Determine the angle bin based on head pose
                h_pose_degree = p[0, 0].numpy()  # Assuming p contains the head pose in degrees
                # v_pose_degree = p[0, 1].numpy() if p.shape[1] > 1 else 0  # Vertical angle (optional, use 0 if not available)
                
                headpose_angle = h_pose_degree  # Example: Adjust according to your needs

                closest_bin = min(angle_bins, key=lambda x: abs(x - headpose_angle))
                all_mse_values[checkpoint_path][closest_bin].append(mse)

            # Calculate the overall average MSE for this checkpoint
            avg_mse_checkpoint = total_mse / total_images if total_images > 0 else 0
            overall_avg_mse[checkpoint_path] = avg_mse_checkpoint

            print(f'Checkpoint: {checkpoint_path}, Average MSE: {avg_mse_checkpoint}')

        # Calculate the average MSE for each angle bin for each checkpoint
        average_mse = {checkpoint: {angle: np.mean(all_mse_values[checkpoint][angle]) if all_mse_values[checkpoint][angle] else 0
                                    for angle in all_mse_values[checkpoint]} for checkpoint in checkpoint_paths}

        # Plotting the results
        plt.figure(figsize=(10, 6))

        # Adjust text sizes
        plt.rc('font', size=11)          # Default text size
        plt.rc('axes', titlesize=12)     # Title size
        plt.rc('axes', labelsize=16)     # Label size (x and y labels)
        plt.rc('xtick', labelsize=12)    # X-axis tick size
        plt.rc('ytick', labelsize=12)    # Y-axis tick size
        plt.rc('legend', fontsize=12)    # Legend font size

        for checkpoint_path in checkpoint_paths:
            angles = sorted(average_mse[checkpoint_path].keys())
            mse_values = [average_mse[checkpoint_path][angle] for angle in angles]

            # plt.plot(angles, mse_values, marker='o', linestyle='-', label=os.path.basename(checkpoint_path))
            if checkpoint_path == './TrainingCheckPoints/training_checkpoints_N_0902':
                plt.plot(angles, mse_values, marker='o', linestyle='-', label=os.path.basename('Training without Discriminator'))
            else:
                plt.plot(angles, mse_values, marker='o', linestyle='-', label=os.path.basename('Training with Discriminator'))

        plt.xlabel('Head Pose Angle (degrees)')
        plt.ylabel('MSE')
        plt.title('MSE vs. Head Pose Angle for Multiple Checkpoints')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)

        return average_mse, overall_avg_mse


    


    def visualize_flow(self, flow, save_path='./flow_visualization.png'):
        """
        Visualizes the optical flow field as a color image with optional arrows to indicate direction.

        Args:
        - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2).
        - save_path (str): Path to save the visualization image.
        - max_magnitude (float): Maximum magnitude for normalization (optional).

        Returns:
        - None: Saves the visualization image to the specified path.
        """

        max_magnitude=None

        flow = flow[0].numpy()  # Assuming batch size of 1 for visualization
        height, width = flow.shape[:2]
        
        # Compute magnitude and angle of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize the magnitude to [0, 1]
        if max_magnitude is not None:
            magnitude = np.clip(magnitude / max_magnitude, 0, 1)
        else:
            magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert angle to degrees
        angle = angle * 180 / np.pi / 2
        
        # Create HSV image: hue corresponds to angle, saturation to 1, value to normalized magnitude
        hsv = np.zeros((height, width, 3), dtype=np.float32)
        hsv[..., 0] = angle  # Hue: direction of flow
        hsv[..., 1] = 1.0    # Saturation: full
        hsv[..., 2] = magnitude  # Value: magnitude of flow
        
        # Convert HSV to RGB (OpenCV uses BGR by default)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Optional: Draw arrows to indicate flow direction (downsampled for clarity)
        step = 10
        for y in range(0, height, step):
            for x in range(0, width, step):
                fx, fy = flow[y, x]
                if magnitude[y, x] > 0.1:  # Draw only if the flow is significant
                    cv2.arrowedLine(rgb_flow, (x, y), (int(x + fx), int(y + fy)), (0, 0, 0), 1, tipLength=0.3)

        # Save the image
        cv2.imwrite(save_path, (rgb_flow * 255).astype(np.uint8))

        print(f"Flow visualization saved to {save_path}")


    def visualize_flow_with_arrows(self, flow, save_path='./flow_with_arrows.png', step=2):
        """
        Visualizes the optical flow field using small arrows to indicate direction and magnitude.
        
        Args:
        - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2).
        - save_path (str): Path to save the visualization image.
        - step (int): Step size for downsampling the flow field for arrow plotting.
        
        Returns:
        - None: Saves the visualization image to the specified path.
        """
        flow = flow[0].numpy()  # Assuming batch size of 1 for visualization
        height, width = flow.shape[:2]
        
        # Create a grid of coordinates
        Y, X = np.mgrid[0:height:step, 0:width:step]
        U = flow[::step, ::step, 0]  # X component of flow
        V = flow[::step, ::step, 1]  # Y component of flow

        V = -V  # Flip the Y component
        U = -U  # Flip the X component
        
        plt.figure(figsize=(10, 10))
        plt.quiver(X - width // 2, Y - height // 2, U, -V, color='red')  # Quiver plot with arrows

        plt.xlim([-width // 2, width // 2])
        plt.ylim([-height // 2, height // 2])
        plt.gca().set_aspect('equal', 'box')
        
        plt.savefig(save_path)
        # plt.show()
        print(f"Flow with arrows visualization saved to {save_path}")

    def run(self):


        # Load the checkpoint
        # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(generator=self.generator,
                                        discriminator=self.discriminator,
                                        gen_optimizer=self.generator_optimizer,
                                        disc_optimizer=self.discriminator_optimizer
                                        )

        # Restore the latest checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        if LOAD_DATA_DIR_PKL_SWITCH == 0:
            test_image, pose, gaze_target, landmarks = self.loadData()
        else:
            test_image, pose, gaze_target, landmarks, target_image = self.loadDataFromPKL()
        
        # Generate the output image
        corr_flow, corr_brightness_map = self.generator(test_image, pose, gaze_target, landmarks, training=False)
        warped_image = self.trans.apply_transformation(corr_flow, test_image)
        generated_image = self.trans.apply_lcm(warped_image, corr_brightness_map)

        # dir
        # _, gaze_d = self.discriminator(test_image, pose, landmarks, training=False)
        # print(gaze_d)

        test_image = (test_image + 1.0) / 2.0
        generated_image = (generated_image + 1.0) / 2.0

        if LOAD_DATA_DIR_PKL_SWITCH == 0:
            self.plotIMGs(test_image, generated_image)
        else:
            target_image = (target_image + 1.0) / 2.0
            # self.plotIMGs_target(test_image, target_image, generated_image)
            self.plotIMGs(test_image, generated_image)


        # self.visualize_flow(corr_flow, save_path='./flow_visualization.png')
        self.visualize_flow_with_arrows(corr_flow, save_path='./flow_with_arrows.png')

        # self.calculate_average_mse()
        # average_mse, overall_avg_mse = self.calculate_mse_for_multiple_checkpoints()
        # average_mse, overall_avg_mse = self.calculate_mse_for_headpose()
        # self.calculate_mse_for_angles()

        # print("Overall Average MSE for each checkpoint:")
        # for checkpoint, mse in overall_avg_mse.items():
        #     print(f'{checkpoint}: {mse}')



if __name__ == '__main__':
    if tf.config.list_physical_devices('GPU'):
        # Sets the peak memory to the current memory.
        tf.config.experimental.reset_memory_stats('GPU:0')
        # Creates the first peak memory usage.
        x1 = tf.ones(1000 * 1000, dtype=tf.float64)
        del x1 # Frees the memory referenced by `x1`.
        peak1 = tf.config.experimental.get_memory_info('GPU:0')['peak']
        # Sets the peak memory to the current memory again.
        tf.config.experimental.reset_memory_stats('GPU:0')
        # Creates the second peak memory usage.
        x2 = tf.ones(1000 * 1000, dtype=tf.float32)
        del x2
        peak2 = tf.config.experimental.get_memory_info('GPU:0')['peak']
        assert peak2 < peak1  # tf.float32 consumes less memory than tf.float64.

    test = Test()
    test.run()