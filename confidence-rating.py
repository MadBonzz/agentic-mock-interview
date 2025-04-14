import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB

# Load the saved model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "confidence-model.pth"
model = torch.load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

class AudioTransform(torch.nn.Module):
    def __init__(self, input_freq=8000, resample_freq=16000, n_fft=1024, win_length=512, hop_length=128, n_mels=128):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.mel_spec = MelSpectrogram(sample_rate=resample_freq, n_fft=n_fft, win_length=win_length,
                                       hop_length=hop_length, n_mels=n_mels)
        self.amplitude_to_db = AmplitudeToDB(stype='power', top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        mel = self.mel_spec(resampled)
        mel_db = self.amplitude_to_db(mel)
        return mel_db

# Initialize audio transformation
audio_transform = AudioTransform(input_freq=16000)

# Load and preprocess audio file
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)  # Load audio file
    transformed_audio = audio_transform(waveform)  # Apply transformations
    return transformed_audio.unsqueeze(0)  # Add batch dimension

# Perform inference on the audio file
def predict_confidence(audio_tensor):
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        output = model(audio_tensor)
        return output.item()

audio_file_path = "path/to/audio.wav"
preprocessed_audio = load_audio(audio_file_path)
confidence_score = predict_confidence(preprocessed_audio)

print(f"Confidence Score: {confidence_score}")
