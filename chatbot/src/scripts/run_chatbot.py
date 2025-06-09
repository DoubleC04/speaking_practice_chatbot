import time
from src.stt.stt_processor import STTProcessor
from src.utils.load_config import load_config
from ..tts.tts_processor import TTSProcessor
from ..rag.rag_pipeline import RAGPipeline
from ..utils.log_config import setup_logging
import sounddevice as sd
import scipy.io.wavfile as wavfile
import sys

config = load_config("config.yaml")
logger = setup_logging()

stt = STTProcessor()
rag = RAGPipeline()
try:
    tts_processor = TTSProcessor()
    logger.info("Initialize TTSProcessor successed")
except Exception as e:
    logger.error(f"Error when initialize TTSProcessor: {e}")
    sys.exit(1)
    
    
speakers = [6, 6, 6, 6, 6]
transcripts = [
    "Excuse me.",
    "Where do I find him.",
    "Yes John Wick that's right.",
    "Yes John Wick.",
    "You keen on earning a coin.",
]
audio_paths = [
    "data/audio/Excuse_me.wav",
    "data/audio/Where_do_i_find_him_.wav",
    "data/audio/Yes_john_wick_that_s_right.wav",
    "data/audio/Yes_john_wick.wav",
    "data/audio/You_keen_on_earning_a_coin.wav",
]

segment = tts_processor.create_segments(transcripts, speakers, audio_paths)

i = 0

while True:
    question = stt.run()
    
    if not question:
        logger.warning("No question captured, continuing to listen...")
        stt.recording = True
        continue
    
    logger.info(f"Question received: {question}")
    print(f"User: {question}")
    
    logger.info("Invoking LLM for response...")
    start_time = time.time()
    response = rag.rag_invoke(question)
    logger.info(f"LLM response took {time.time() - start_time:.2f} seconds")
    print(f"Bot: {response}")
    
    audio = tts_processor.generate_audio(
        text=response,
        speaker=6,
        context=segment
    )
    
    audio_path = f"chatbot_{i}.wav"

    tts_processor.save_audio(audio_path, audio, tts_processor.generator.sample_rate)

    sample_rate, data = wavfile.read(audio_path)

    sd.play(data, sample_rate)
    
    i = i + 1
    sd.wait()
    
    time.sleep(1)











