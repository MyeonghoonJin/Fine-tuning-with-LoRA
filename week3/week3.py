from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import AutoTokenizer
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers import TrainingArguments, Trainer
from datasets import Dataset

import torch
import scipy
import json
import os

# model 설정
model_name = "facebook/musicgen-small"
model = MusicgenForConditionalGeneration.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터 셋
dataset = []
genres = ["classical", "hiphop", "jazz", "metal", "pop"]
data_dir = os.path.abspath("./week3/DataSet")

# 장르별 데이터 로드
for genre in genres:
    genre_dir = os.path.join(data_dir, genre)
    for filename in os.listdir(genre_dir):
        if filename.endswith(".wav"):   
            # 파일명에서 텍스트 설명 생성
            description = f"A {genre} melody with unique characteristics."
            dataset.append({"audio": os.path.join(genre_dir, filename), "description": description})

# 데이터셋 JSON 저장 
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

# 데이터셋 로드
datasets = Dataset.from_list(dataset)


linear_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_modules.append(name)

# LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=linear_modules
)
# 모델에 LoRA적용
model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# 데이터 전처리
def preprocess_function(data):
    inputs = tokenizer(data["description"], padding="max_length", truncation=True, max_length=128)
    return inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True)
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)


# 출력 디렉토리 설정
output_dir = os.path.abspath("./week3/output_dir/")

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir = output_dir,
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)
# Trainer 설정 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"], 
)
trainer.train()

test_description = "A soothing jazz melody with saxophone."
inputs = processor(
    test_description,
    padding=True,
    return_tensors="pt",
    )

audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out_test.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())