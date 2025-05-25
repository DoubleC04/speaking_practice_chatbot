import time
import threading
import keyboard
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from io import BytesIO

from .stt_utils import audio_callback, audio_queue
from .model import load_model
from src.utils.load_config import load_config
from src.utils.log_config import setup_logging


logger = setup_logging()

class STTProcessor:
    def __init__(self):
        self.config = load_config("config.yaml")
        stt_config = self.config["stt"]
        
        self.sample_rate = stt_config["sample_rate"]
        self.chunk_duration = stt_config["chunk_duration"]
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.beam_size = stt_config["beam_size"]
        self.vad_parameters = stt_config["vad_parameters"]

        self.recording = True
        self.full_transcript = ""
        self.model = load_model()
        
    def process_audio(self, audio_data, sample_rate):
        """
        Process audio data into text.

        Args:
            audio_data (np.ndarray): Numpy array containing raw audio data.
            sample_rate (int): The sampling rate of the audio data.

        Returns:
            str: The transcribed text from the audio data.
        """
        try:
            buffer = BytesIO()
            wavfile.write(buffer, sample_rate, audio_data)
            buffer.seek(0)
            
            segments, info = self.model.transcribe(
                buffer, 
                beam_size=self.beam_size,
                language="en",
                vad_filter=True,
                vad_parameters=self.vad_parameters
            )
            text = " ".join([seg.text for seg in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Error when process audio: {e}")
            return ""
        
    def check_stop_key(self):
        """
        Check if 'S' key is pressed to stop recording.
        """
        while self.recording:
            if keyboard.is_pressed('s'):
                logger.info("Stopping recording.")
                self.recording = False
                break
            time.sleep(0.1)
        
    def run(self):
        """
        Run the audio recording and processing loop to capture and transcribe speech.

        Returns:
            str: The final transcribed text from the recorded audio.
        """
        chunk_samples = self.chunk_samples
        sample_rate = self.sample_rate
        
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                callback=audio_callback,
            ):
                logger.info("Start recording... Press 'S' to stop...")
                
                stop_thread = threading.Thread(target=self.check_stop_key)
                stop_thread.daemon = True
                stop_thread.start()
                
                audio_data = np.empty((0,), dtype=np.float32)
                
                while self.recording:
                    if audio_queue.qsize() == 0:
                        time.sleep(0.1)
                        continue
                        
                    new_data = audio_queue.get()
                    audio_data = np.concatenate((audio_data, new_data.flatten()), axis=0)
                    
                    while len(audio_data) >= chunk_samples:
                        chunk = audio_data[:chunk_samples]
                        text = self.process_audio(chunk, sample_rate)
                        if text:
                            self.full_transcript += text + " "
                            print(text, end=" ", flush=True)
                        audio_data = audio_data[chunk_samples:]
                        
                    if len(audio_data) > 0:
                        text = self.process_audio(audio_data, sample_rate)
                        if text:
                            self.full_transcript += text + " "
                            print(text, end=" ", flush=True)
                
        except Exception as e:
            logger.error(f"Error when initialize micro: {e}")
            
        final_transcript = self.full_transcript.strip()
        self.full_transcript = ""
        audio_queue.queue.clear()
            
        return final_transcript
    

            