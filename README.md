# Human Rights Demand LLM Model

This project involves training a large language model (LLM) to generate human rights-related responses using pre-quantized models from Unsloth. The model is fine-tuned with custom datasets, employing low-rank adaptation (LoRA) for efficient training on limited hardware resources. The inference pipeline enables fast generation of responses to specific prompts related to human rights topics.

## Getting Started

### Prerequisites

Ensure you have the following packages installed:

- Python 3.7+
- PyTorch (with CUDA support for GPU acceleration)
- Unsloth library
- Transformers
- Datasets
- TRL (Training with Reinforcement Learning)

The required packages can be installed using the following commands:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes triton
```

### Model Selection and Configuration

The model utilizes 4-bit quantization to optimize memory usage. You can choose from a variety of pre-quantized models available in the `fourbit_models` list.

```python
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    # More models available
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    max_seq_length = 2048,
    dtype = None, # Auto detection
    load_in_4bit = True,
)
```

### LoRA Adapter Configuration

The model is further optimized using LoRA adapters, which significantly reduce the computational overhead during training.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Low-rank adaptation factor
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

### Dataset Preparation

You can load and format your custom dataset using the following steps:

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="converted_data.json", split="train")

# Apply formatting
dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Training the Model

Training is performed using the `SFTTrainer` from the `trl` library with specified training arguments.

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
```

### Inference

To generate responses using the trained model, the following inference pipeline is used:

```python
inputs = tokenizer(
    [alpaca_prompt.format("Instruction here", "Input context here", "")],
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128)
response = tokenizer.batch_decode(outputs)
```

### Saving the Model

After training and inference, you can save the model locally or push it to Hugging Face Hub:

```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Push to Hugging Face Hub
model.push_to_hub("your_username/Human_Rights_Demand", use_auth_token=True)
tokenizer.push_to_hub("your_username/Human_Rights_Demand", use_auth_token=True)
```

### Usage Example

```python
inputs = tokenizer(
    [alpaca_prompt.format("Continue the Fibonacci sequence.", "1, 1, 2, 3, 5, 8", "")],
    return_tensors = "pt"
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

response = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
```

## Acknowledgments

This project utilizes the `unsloth` library and models from Hugging Face. Special thanks to the contributors of open-source libraries that made this project possible.

---

Feel free to adjust any sections or details according to your specific needs.
