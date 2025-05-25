import logging

from faster_whisper import WhisperModel
from src.utils.load_config import load_config
from src.utils.log_config import setup_logging

logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logger = setup_logging()
config = load_config("config.yaml")

stt_config = config["stt"]

model_size = stt_config["model_size"]
device = stt_config["device"]
compute_type = stt_config["compute_type"]

_model = None

def load_model(force_reload=False):
    """
    Load the Speech to Text model

    Args:
        force_reload (bool, optional): If True, force reload the model even if already loaded. Defaults to False.
                                       Use when you want to change the configuration of the model.

    Returns:
        WhisperModel: Loaded Speech to Text model
    """
    global _model
    
    if _model is not None and not force_reload:
        logger.info("Speech to Text model already loaded, reusing instance")
        return _model
    
    logger.info(f"Loading model {model_size} on {device}")
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            local_files_only=False
        )
        _model = model
        logger.info("Speech to Text model is ready")
        return model
    except Exception as e:
        logger.error(f"Error when loading Speech to Text model: {e}")
        raise e