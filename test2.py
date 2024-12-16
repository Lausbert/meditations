from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from pathlib import Path
import os

def initialize_models():
    """Initialize and cache the TTS models"""
    # Create cache directory if it doesn't exist
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'tts'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache dir: {cache_dir}")
    
    # Initialize processor and model with cache
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=cache_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir=cache_dir)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=cache_dir)
    
    # Load speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=cache_dir)
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    return processor, model, vocoder, speaker_embeddings

def text_to_speech(text, output_path="output2.wav"):
    """Convert text to speech and save to file"""
    # Initialize models (they'll be cached after first run)
    processor, model, vocoder, speaker_embeddings = initialize_models()
    
    # Prepare inputs
    inputs = processor(text=text, return_tensors="pt")
    
    # Generate speech with volume normalization
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Normalize audio to prevent clipping
    speech = speech.numpy()
    speech = speech / max(abs(speech)) * 0.95
    
    # Save the audio file
    sf.write(output_path, speech, samplerate=16000)
    print(f"Audio saved to: {os.path.abspath(output_path)}")

# Example usage
if __name__ == "__main__":
    # Text to synthesize
    text = "Find a comfortable position in a quiet place. Sit upright or lie on your back. Your arms and legs should be relaxed."
    
    try:
        text_to_speech(text)
        print("Speech synthesis completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")