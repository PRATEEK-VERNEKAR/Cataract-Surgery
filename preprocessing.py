import cv2
from os import listdir
from os.path import join

def preprocess_data(video_dir, csv_file, window_size):
  """
  Preprocesses video data for cataract surgery phase detection.

  Args:
      video_dir: Path to the directory containing video files.
      csv_file: Path to the CSV file containing annotation data.
      window_size: Number of consecutive frames to include in each sequence.

  Returns:
      sequences: List of frame sequences.
      labels: List of corresponding phase labels.
  """

  # Read phase labels from CSV
  phase_labels = {}
  with open(csv_file, 'r') as f:
    for line in f.readlines()[1:]:  # Skip header
      video_id, frame_no, phase = line.strip().split(',')
      phase_labels[(int(video_id), int(frame_no))] = int(phase)

  # Initialize empty lists
  sequences = []
  labels = []

  # Loop through all video files
  for filename in listdir(video_dir):
    if filename.endswith('.mp4'):
      video_path = join(video_dir, filename)

      # Load video frames
      cap = cv2.VideoCapture(video_path)
      frames = []
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames.append(frame)
      cap.release()

      # Normalize pixel values
      frames = [frame / 255.0 for frame in frames]

      # Create frame sequences
      for i in range(window_size, len(frames)):
        sequence = frames[i - window_size:i]
        label = phase_labels.get((int(filename[:-4]), frames[i - window_size][0]))  # Match video ID and starting frame
        if label is not None:  # Skip frames without annotation
          sequences.append(sequence)
          labels.append(label)

  return sequences, labels

# Example usage
video_dir = "./videos/"
csv_file = "annotations.csv"
window_size = 16

sequences, labels = preprocess_data(video_dir, csv_file, window_size)

print("Number of sequences:", len(sequences))
print("Number of labels:", len(labels))

# Use sequences and labels for training your model
