
import tensorflow as tf # type: ignore
from Model.model import Generator, GazeRedirectGAN, Discriminator
from processingDataset import ProcessingDataset

import matplotlib.pyplot as plt

import json
import os



class LossHistory(tf.keras.callbacks.Callback):
    # def on_train_begin(self, logs={}):
    #     self.losses = []
    #     self.val_losses = []

    # def on_epoch_end(self, epoch, logs={}):
    #     self.losses.append(logs.get('cl_img'))
    #     self.val_losses.append(logs.get('L_total'))

    # def save_loss_data(self, save_path):
    #     import json
    #     data = {
    #         'cl_img': self.losses,
    #         'cl_tot': self.val_losses
    #     }
    #     with open(save_path, 'w') as f:
    #         json.dump(data, f)
    def __init__(self, save_path='./loss_data.json'):
        super().__init__()
        self.save_path = save_path
        # Initialize an empty dictionary to store losses if the file doesn't exist
        if not os.path.exists(self.save_path):
            with open(self.save_path, 'w') as f:
                json.dump({'cl_img': [], 'cl_tot': []}, f)

    def on_epoch_end(self, epoch, logs={}):
        # Get the current losses
        cl_img_loss = logs.get('cl_img')
        cl_tot_loss = logs.get('L_total')

        # Append the losses to the JSON file
        if cl_img_loss is not None and cl_tot_loss is not None:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            # Append new losses to the existing list
            data['cl_img'].append(cl_img_loss)
            data['cl_tot'].append(cl_tot_loss)

            # Write back the updated data to the JSON file
            with open(self.save_path, 'w') as f:
                json.dump(data, f)

    def plot_loss(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='MSE Loss')
        # if self.val_losses:
        #     plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (MSE) Over Time')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()




class Train():
    def __init__(self):
        # self.file_path_l = './DataSets/training_inputs_COL_0813/left_data.pkl'
        # self.file_path_r = './DataSets/training_inputs_COL_0813/right_data.pkl'
        # self.file_path_l = './DataSets/training_inputs_U2_0823/left_data.pkl'
        # self.file_path_r = './DataSets/training_inputs_U2_0823/right_data.pkl'
        self.pkl_folder_path_N = './DataSets/training_inputs_DIRL_0824'
        self.pkl_folder_path_S = './DataSets/training_inputs_U2_0824'
        # self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_N_0813'
        self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_N_0912_2'
        self.batch_size = 256#128
        self.epochs = 500

    def run(self):
        process_dataset = ProcessingDataset()

        
        # data_l = process_dataset.load_pickle_data(self.file_path_l)
        # data_r = process_dataset.load_pickle_data(self.file_path_r)
        # data = {**data_l, **data_r}
        dataN_list = process_dataset.load_pickle_data(self.pkl_folder_path_N)
        dataS_list = process_dataset.load_pickle_data(self.pkl_folder_path_S)
        # data = {**dataN, **dataS}

        train_datasets_N = [process_dataset.create_dataset(data, self.batch_size) for data in dataN_list]
        train_datasets_S = [process_dataset.create_dataset(data, self.batch_size) for data in dataS_list]

        all_datasets = train_datasets_N#train_datasets_N + train_datasets_S

        train_dataset = tf.data.Dataset.sample_from_datasets(all_datasets)


        generator = Generator()
        discriminator = Discriminator()

        gan_model = GazeRedirectGAN(generator, discriminator)
        gan_model.compile(
            gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9),
            disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9),
            loss_fn = tf.keras.losses.MeanSquaredError()
        )

        # Define checkpoint directory and checkpoint objects
        
        # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(generator=gan_model.generator,
                                        discriminator=gan_model.discriminator,
                                        gen_optimizer=gan_model.gen_optimizer,
                                        disc_optimizer=gan_model.disc_optimizer
                                        )

        checkpoint_manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_dir, max_to_keep=10)

        # Restore the latest checkpoint if it exists
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f'Restored from checkpoint: {checkpoint_manager.latest_checkpoint}')
        else:
            print('No checkpoint found, starting from scratch.')

        # Define a callback to save checkpoints
        class CheckpointSaver(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                checkpoint_manager.save()
                print(f'\nCheckpoint saved at epoch {epoch + 1}')

        loss_history = LossHistory()

        gan_model.fit(train_dataset, epochs=self.epochs, callbacks=[CheckpointSaver(), loss_history])

        # Plot and save loss after training
        # loss_history.plot_loss(save_path='./training_loss.png')
        # loss_history.save_loss_data('./loss_data.json')

if __name__ == '__main__':
    train = Train()
    train.run()
    print("Training Completed!!!")