from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    attn_implementation="eager"
)

# 데이터 전처리
inputs = processor(
    text=[
        "2000s jazz pop with saxophone", "90s rock with loud guitars"
    ],
    padding=True,
    return_tensors="pt",
)
    
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=10, max_new_tokens=256) #512 = 8초


sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out1-2.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())