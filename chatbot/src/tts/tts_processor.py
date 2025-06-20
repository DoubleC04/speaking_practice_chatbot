from src.tts.csm.generator import load_csm_1b, Segment
import torchaudio
import torch
import re

class TTSProcessor:
    def __init__(self, device=None):
        """
        Initialize model and decide which device to use.
        
        Args:
            device (str): Device to run model (CPU, CUDA, MPS).
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.generator = load_csm_1b(device=self.device)
    
    def estimate_audio_length_ms(self, text: str, ms_per_word: int = 450):
        """
        Estimate the audio length in milliseconds based on the length of text.

        Args:
            text (str): The input text to estimate audio length for.
            ms_per_word (int, optional): Milliseconds per word. Defaults to 450.

        Returns:
            int: Estimated audio length in milliseconds.
        """
        words = [word.strip(".,!?\"'") for word in text.split()]
        word_count = len(words)
        audio_length_ms = word_count * ms_per_word
        
        return audio_length_ms
    
    def preprocess_text(self, text):
        """
        Preprocesses raw text and splits it into cleaned sentences.

        This method performs the following steps:
        - Removes HTML tags and unwanted special characters.
        - Normalizes whitespace by replacing multiple spaces with a single space.
        - Converts all characters to lowercase.
        - Splits the text into individual sentences using punctuation marks (. ! ?).

        Args:
            text (str): The input raw text to preprocess.

        Returns:
            list[str]: A list of cleaned and lowercased sentences.
        """
        html_and_special_chars = re.compile(r'<[^>]+>|[^\w\s.,!?]')
        whitespace = re.compile(r'\s+')
        sentence_splitter = re.compile(r'[.!?]+')

        text = html_and_special_chars.sub('', text)
        
        text = whitespace.sub(' ', text).strip()
        
        text = text.lower()
        
        sentences = [s.strip() for s in sentence_splitter.split(text) if s.strip()]
        
        return sentences

    def generate_audio(self, text, speaker, context=None):
        """
        Generate audio from text.

        Args:
            text (str): The input text to be converted into speech.
            speaker (int): ID of the speaker.
            context (list): List of Segment objects providing conversational context (optional).

        Returns:
            torch.Tensor: The generated audio waveform.
        """
        
        audio_length_ms = self.estimate_audio_length_ms(text=text)
        sentences = self.preprocess_text(text)
        
        audio_segments = []
        for sentence in sentences:
            with torch.inference_mode():
                audio = self.generator.generate(
                    text=sentence,
                    speaker=speaker,
                    context=context or [],
                    max_audio_length_ms=audio_length_ms,
                    skip_watermark=True
                )
            audio_segments.append(audio)
        
        if audio_segments:
            return torch.cat(audio_segments, dim=-1)
        return torch.tensor([])

    def save_audio(self, file_path, audio, sample_rate):
        """
        Save audio to a WAV file.

        Args:
            file_path (str): Path where the audio file will be saved.
            audio (torch.Tensor): The audio tensor to be saved.
            sample_rate (int): Sample rate of the audio (e.g., generator.sample_rate).
        """

        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), sample_rate)

    def load_audio(self, audio_path):
        """
        Load audio from file and conversion of frequency samples (resample) to suit with sample_rate of model.

        Args:
            audio_path (str): Audio filepath.
        
        Returns:
            torch.Tensor: Audio after resample.
        """
        audio_tensor, original_sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=original_sample_rate, new_freq=self.generator.sample_rate
        )
        return audio_tensor
    
    def create_segments(self, transcripts, speakers, audio_paths):
        """
        Create segments from transcripts and audio files.

        Args:
            transcripts (list): List of text transcripts.
            speakers (list): List of speaker IDs.
            audio_paths (list): List of audio file paths.

        Returns:
            list: List of Segment objects.
        """

        segments = [
            Segment(text=transcript, speaker=speaker, audio=self.load_audio(audio_path))
            for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
        ]
        return segments





