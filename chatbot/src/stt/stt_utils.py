import queue

from src.utils.log_config import setup_logging

audio_queue = queue.Queue()
logger = setup_logging()

def audio_callback(indata, frames, time, status):
    if status:
        logger.info(f"Status of audio: {status}")
    audio_queue.put(indata.copy())
    

    
