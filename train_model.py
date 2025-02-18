import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

# Initialize TensorBoard callback
tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))

# Set the dataset directory
dataset_dir = r"C:\Users\ARUNIMA\OneDrive\Desktop\Team11_Virtual_Diagnosis of Skin Disorder_icv"

# Check if the directory exists
if not os.path.exists(dataset_dir):
    raise ValueError(f"The dataset directory does not exist: {dataset_dir}")

# ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,        # Random rotation
    width_shift_range=0.2,    # Random width shift
    height_shift_range=0.2,   # Random height shift
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,      # Random horizontal flip
    fill_mode='nearest',       # Fill pixels with nearest valid value
    validation_split=0.2
)

# Generators
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=8,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    subset='training',  # Set as training data
    shuffle=True  # Shuffle the training data
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    subset='validation'  # Set as validation data
)

# Check the number of images found
print(f"Training images found: {train_generator.samples}")
print(f"Validation images found: {validation_generator.samples}")

# Create a simple CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # First convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # First pooling layer
    model.add(Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second pooling layer
    model.add(Conv2D(128, (3, 3), activation='relu'))  # Third convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Third pooling layer
    model.add(Flatten())  # Flatten the output for the fully connected layer
    model.add(Dense(128, activation='relu'))  # Fully connected layer
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(len(train_generator.class_indices), activation='softmax'))  # Output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
    return model

# Create the model
model = create_model()

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[tensorboard])

# Save the model
model.save('skin_disorder_model.h5')  # Save the model as HDF5 file
print("Model saved to skin_disorder_model.h5")

