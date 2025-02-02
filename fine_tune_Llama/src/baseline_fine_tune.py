import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm

# 하이퍼파라미터 설정
model_name = "LLaMA3.2-1B"  # 실제 모델 id로 변경하세요.
batch_size = 4
learning_rate = 5e-5
num_epochs = 3
max_length = 512   # 최대 토큰 길이 (필요에 따라 조정)

# Accelerator 초기화
accelerator = Accelerator()

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터셋 로드
dataset = load_dataset("coastral/korean-writing-style-instruct")

# 프롬프트 생성 함수
def generate_prompt(example):
    # 'input' 필드가 비어있지 않은 경우와 비어있는 경우를 분기
    if example.get("input", "").strip():
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return prompt

# 전처리 함수: 각 샘플에 대해 프롬프트를 생성하고 토큰화합니다.
def preprocess_function(examples):
    # 각 예제마다 프롬프트 생성
    prompts = [generate_prompt(example) for example in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    )]
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    # causal language modeling에서는 label로 input_ids를 그대로 사용합니다.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 데이터셋 전처리 적용 (train 데이터셋 사용)
# 기존 필드("instruction", "input", "output")는 필요 없으므로 제거합니다.
columns_to_remove = dataset["train"].column_names
tokenized_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=columns_to_remove
)

# DataLoader 생성
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

# 옵티마이저 및 학습 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Accelerator를 이용해 모델, 옵티마이저, DataLoader, 스케줄러 준비
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# 학습 시작
global_step = 0
model.train()

progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 이미 accelerator에 의해 올바른 device로 이동된 배치를 사용합니다.
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        global_step += 1
        progress_bar.update(1)
        
        if global_step % 100 == 0:
            accelerator.print(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f}")

# 학습 완료 후 모델 저장 (Accelerator 환경에 맞게 저장)
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("finetuned-llama3.2-1B")
tokenizer.save_pretrained("finetuned-llama3.2-1B")
