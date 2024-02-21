# Load the preprocessed video sequence
new_video_sequence = preprocess_video("path/to/new_video.mp4")

# Load the trained model
model.load_weights('best_model.h5')

# Predict the phase
predicted_phases = model.predict(new_video_sequence)

# Get the phase with the highest probability
predicted_phase = np.argmax(predicted_phases[0])

print(f"Predicted phase: {predicted_phase}")
