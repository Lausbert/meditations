import torch
from datetime import datetime
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

def text_to_speech(prompt):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-multilingual-v1.1")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    description = "Nicole's german voice is flat, seductive, slow, soft and meditative without any emotions. Make a breath pause of three seconds between each sentence. Use a very close recording that almost has no background noise."

    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sf.write("parler_tts_out-" + timestamp + ".wav", audio_arr, model.config.sampling_rate)

if __name__ == "__main__":

    text = "Schließen Sie sanft Ihre Augen. Spüren Sie, wie Ihr Körper von der Unterlage getragen wird. Lassen Sie alle Anspannung los. Richten Sie nun Ihre Aufmerksamkeit auf Ihren natürlichen Atem. Beobachten Sie, wie die Luft durch Ihre Nase ein- und ausströmt. Sie müssen nichts verändern. Beobachten Sie einfach."
 
    # Funktion aufrufen
    text_to_speech(text)
