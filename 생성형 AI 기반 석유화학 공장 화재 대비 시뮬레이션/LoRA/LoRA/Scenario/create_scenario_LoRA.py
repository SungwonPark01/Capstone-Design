import os
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    pipeline, 
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
    PeftModel
)
from trl import SFTTrainer
import json
import glob

# GPU 환경 확인 및 attention 메커니즘 설정 - attention은 나중에 더 알아보고 결정
if torch.cuda.get_device_capability()[0] >= 8:
#    !pip install -qqq flash-attn
#    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
#    attn_implementation = "eager"
    torch_dtype = torch.float16
    
# BitsAndBytesConfig 설정
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, # 모델을 4비트 정밀도로 로드
    bnb_4bit_quant_type=“nf4”, # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
    bnb_4bit_compute_dtype=torch_dtype, # 연산 속도를 높이기 위해 사용 (default: torch.float32)
    bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
)

# 1. LLaMA 모델 로드 및 토크나이저 불러오기
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False # 캐시 비활성화
model.config.pretraining_tp = 1 # 단일 GPU에서 실행

# 2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # 시퀀스 길이를 맞추기 위해 문장 끝에 eos_token 사용
tokenizer.padding_side = "right" # 오른쪽에 패딩 토큰 추가

# 3. 데이터 불러오기 및 데이터셋 생성
train_files = glob.glob("datasets/scenario_dataset/*.json")
data_files = {"train": train_files}
dataset = load_dataset("json", data_files=data_files, split="train")

# 포지션, 로테이션, 스케일 등을 한 줄로 포맷
def format_vectors_inline(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k in {"position", "rotation", "scale", "size", "PlayerStart"} and isinstance(v, list):
                new_obj[k] = f"[{', '.join(map(str, v))}]"
            else:
                new_obj[k] = format_vectors_inline(v)
        return new_obj
    elif isinstance(obj, list):
        return [format_vectors_inline(i) for i in obj]
    else:
        return obj

# JSON 문자열로 변환하되, vector는 한 줄로 유지
def json_pretty_with_inline_vectors(data):
    formatted = format_vectors_inline(data)
    json_str = json.dumps(formatted, ensure_ascii=False, indent=2)
    return json_str.replace('"[', '[').replace(']"', ']')

# 최종 프롬프트 + 라벨 생성 함수
def create_prompt_and_label(example):
    user_prompt = example["user_prompt"]
    rag_result = example["rag_result"]
    output_json = example["output"]

    # RAG 정보 포맷팅
    rag_objects_str = json_pretty_with_inline_vectors(rag_result["objects"])
    rag_effects_str = json_pretty_with_inline_vectors(rag_result["effects"])
    rag_text = f"오브젝트 정보:\n{rag_objects_str}\n\n이펙트 목록:\n{rag_effects_str}"

    # 프롬프트
    prompt = f"""### 지시사항:
당신은 화재 시뮬레이션 전문가입니다. 사용자 상황과 RAG 정보를 참고해 JSON 시나리오를 만들어주세요.

### 사용자 제시 상황:
{user_prompt}

### RAG 검색 결과:
{rag_text}

### 응답:"""

    # 정답 JSON 포맷팅
    label_str = json_pretty_with_inline_vectors(output_json)

    # 통합
    full_text = prompt + "\n" + label_str  + "<|im_end|>"

    return {"text": full_text}

# 데이터셋 프롬프트 템플릿 적용
remove_column_keys = dataset.features.keys()
#remove_column_keys

# 시나리오 데이터셋이 1000개 단위로 스타일이 달라서 shuffle함.
dataset_cvted = dataset.shuffle().map(create_prompt_and_label, remove_columns = remove_column_keys)
#dataset_cvted


# 4. 데이터 토크나이징
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

remove_column_keys = dataset_cvted.features.keys()
dataset_tokenized = dataset_cvted.map(tokenize_function, batched=True, remove_columns=remove_column_keys)

# 5. LoRA 설정
lora_config = LoraConfig(
    r=4, # 8, 16, 32, 64, 128로 테스트
    lora_alpha=8, # LoRA 스케일링 계수
    #target_modules=["q_proj", "v_proj"], # loRA를 적용할 레이어.
    lora_dropout=0.05, # 과적합 방지를 위한 드롭아웃 확률. 기본값: 0.1
    bias="none", # LoRA 어댑터 행렬에 대한 편향을 추가할지 여부
    task_type="CAUSAL_LM", # LoRA가 적용될 작업 유형
)

# 양자화된 모델을 학습하기 전, 전처리를 위해 호출
model = prepare_model_for_kbit_training(model)

# LoRA 학습을 위해서 peft를 사용하여 모델 wrapping
model = get_peft_model(model, lora_config)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2, # 각 GPU당 배치 사이즈, 기본값: 8
    gradient_accumulation_steps=4, # 가중치 업데이트 수행 빈도, 기본값: 1
    #num_train_epochs=1, # 전체 학습 데이터셋 반복횟수, 기본값: 3 / max_steps를 우선
    logging_dir="./logs",
    logging_steps=25,# 해당 스탭마다 로그 기록, 저장
    save_steps=500, # 해당 스텝마다 모델 기록, 저장
    learning_rate=2e-4, # 학습률, 라마3 참고해서 설정
    weight_decay=0.001, # 가중치 감소 계수 설정, 기본값: 0
    max_grad_norm=0.3, # 그레디언트의 최대 Norm 설정, 그레디언트 폭발 방지
    max_steps=1000, # 
    #evaluation_strategy="no" - eval
    optim="paged_adamw_8bit", # 8비트 AdamW 옵티마이저
    fp16=False, # 3080ti는 fp16 연산을 지원함
    bf16=True, # 3080ti에서 지원되는지 확실히 모름. 일단 위키피디아에 3000 시리즈가 지원된다는 말이 있다 해서 일단 얘를 키고, fp16 끔
)

# 7. 학습 실행
# Data Collator 역할
# 각 입력 시퀀스의 input_ids(토큰) 길이를 계산하고, 가장 긴 길이를 기준으로 길이가 짧은 시퀀스에는 패딩 토큰 추가
def collate_fn(examples):
    examples_batch = tokenizer.pad(examples, padding='longest', return_tensors='pt')
    examples_batch['labels'] = examples_batch['input_ids'] # 모델 학습 평가를 위한 loss 계산을 위해 입력 토큰을 레이블로 사용
    return examples_batch

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_tokenized,
    max_seq_length=256, # 최대 시퀀스 길이
    args=training_args,
    dataset_text_field="text",
    data_collator=collate_fn
)

trainer.train()

# 8. 모델 저장
model.save_pretrained("./scenario_lora")
tokenizer.save_pretrained("./scenario_lora")

