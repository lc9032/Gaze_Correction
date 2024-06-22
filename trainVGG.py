import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda

from tensorflow.keras.layers import Concatenate



# Load pickle data
def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Preprocess the data
def preprocess_data(img, p, h, v):
    img = tf.cast(img, tf.float32)
    # Normalize the images to [-1, 1]
    img = (img / 127.5) - 1.0

    p = tf.cast(p, tf.float32)
    h = tf.cast(h, tf.float32)
    v = tf.cast(v, tf.float32)

    gaze_real = tf.stack([h, v], axis=-1)

    return img, p, gaze_real

# Create dataset
def create_dataset(data, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['p'], data['h'], data['v']))
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset.map(lambda img, p, gaze: ((img, p), gaze))

# Load data
file_path_l = './training_inputs_COL/left_data.pkl'
batch_size = 64
data_l = load_pickle_data(file_path_l)
gaze_dataset = create_dataset(data_l, batch_size)

# Function to build VGG model with fully connected layers
def build_model(input_shape=(32, 64, 3)):
    # Image input
    img_input = Input(shape=input_shape, name='img_input')
    pose_input = Input(shape=(1,), name='pose_input')

    # Reshape and tile pose to match image input dimensions
    pose = Lambda(lambda p: tf.tile(tf.reshape(p, (-1, 1, 1, 1)), [1, input_shape[0], input_shape[1], 1]))(pose_input)

    # Concatenate image and pose inputs
    concatenated = Concatenate()([img_input, pose])

    # Convolutional layers for feature extraction
    x = Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')(concatenated)


    x = Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=4, strides=2, padding='same', activation='relu')(x)

    x = Flatten()(x)

    # Dense layers for regression
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer for gaze prediction
    x = Dense(3, activation='linear')(x)  # Outputs in range [-1, 1]


    gan, gaze = tf.split(x, num_or_size_splits=[1, 2], axis=-1)

    # outputs = Lambda(lambda x: x * 20.0)(gaze)  # Scaling to range [-20, 20]

    # model = tf.keras.Model(inputs=[img_input, pose_input], outputs=outputs, name='gaze_prediction_model')
    model = tf.keras.Model(inputs=[img_input, pose_input], outputs=[gaze], name='gaze_prediction_model')
    return model


# Build the model
# model = build_vgg_model()

model = build_model()
model.summary()

# model = tf.keras.models.load_model('disc_model.h5')

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss='mean_squared_error',
              metrics=['mae'])

# Train the model
epochs = 100
history = model.fit(gaze_dataset, epochs=epochs)

# Save the model
model.save('disc_model.h5')

# Predict gaze direction
def predict_gaze(model, img, pose):
    img, pose, _ = preprocess_data(img, pose, 0, 0)  # Only need to preprocess the image and pose
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    pose = tf.expand_dims(pose, axis=0)  # Add batch dimension
    prediction = model.predict([img, pose])
    return prediction



# Load and preprocess your input image
input_image = cv2.imread('./preprocessing_dataset_COL/0009/left/0009_2m_0P_-10V_10H.jpg')  # Load your image using OpenCV
pose = 0.0 
prediction = predict_gaze(model, input_image, pose)
print('Predicted gaze:', prediction)


# input_image = cv2.resize(input_image, (64, 32))  # Resize to match the input shape
# input_image = (input_image / 127.5) - 1.0  # Normalize to [-1, 1]

# # Predict gaze vector
# predicted_gaze = vgg_gaze_model.predict(np.expand_dims(input_image, axis=0))  # Expand dims for batch size 1
# print("Predicted gaze vector:", predicted_gaze)






