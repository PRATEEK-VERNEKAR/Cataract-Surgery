from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(window_size, num_frames, num_phases):
  """
  Creates a TensorFlow Keras model for cataract surgery phase detection.

  Args:
      window_size: Number of frames in each sequence.
      num_frames: Number of frames per video (used for data shaping).
      num_phases: Number of possible phases.

  Returns:
      A TensorFlow Keras model.
  """

  model = Sequential([
    # 3D CNN for feature extraction
    TimeDistributed(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(window_size, num_frames, num_frames, 3))),
    TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))),
    TimeDistributed(Conv3D(64, kernel_size=(3, 3, 3), activation='relu')),
    TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))),
    TimeDistributed(Flatten()),

    # LSTM for sequence modeling
    LSTM(128, return_sequences=True),  # Process each frame sequence
    LSTM(64),

    # Output layer
    Dense(num_phases, activation='softmax')
  ])

  model.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
  return model

def train_model(data_dir, csv_file, window_size, batch_size, epochs):
  """
  Trains the model on cataract surgery video data.

  Args:
      data_dir: Path to the directory containing video files.
      csv_file: Path to the CSV file containing annotation data.
      window_size: Number of frames in each sequence.
      batch_size: Number of sequences to process in each training step.
      epochs: Number of training epochs.
  """

  x_train, y_train, x_val, y_val = load_and_split_data(data_dir, csv_file, window_size)

  # Get video dimensions for input shape consistency
  num_frames, height, width, _ = x_train[0].shape

  model = create_model(window_size, num_frames, len(set(y_train)))

  # Early stopping to prevent overfitting
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)

  # Model checkpoint to save the best model
  model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_accuracy')

  history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[early_stopping, model_checkpoint])

  return history, model

# Example usage
data_dir = "./videos/"
csv_file = "annotations.csv"
window_size = 16
batch_size = 32
epochs = 10

history, model = train_model(data_dir, csv_file, window_size, batch_size, epochs)

# You can further evaluate the model performance and use it for predictions
