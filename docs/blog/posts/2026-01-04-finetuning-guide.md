---
title: "The Complete Guide to LLM Fine-Tuning in 2025: From Theory to Production"
date:
    created: 2026-01-04
    updated: 2026-01-04
tags: [ai-engineering, llm, fine-tuning, lora, qlora, peft, unsloth, axolotl]
description: A comprehensive, beginner-friendly guide to LLM fine-tuning covering PEFT methods (LoRA, QLoRA, DoRA), frameworks (Unsloth, Axolotl), practical code examples, output formats, and deployment strategies.
author: Viacheslav Dubrov
---

# The Complete Guide to LLM Fine-Tuning in 2025: From Theory to Production

Fine-tuning has become the secret weapon for building specialized AI applications. While general-purpose models like GPT-4 and Claude excel at broad tasks, fine-tuning transforms them into laser-focused experts for your specific domain. This guide walks you through everything you need to know—from understanding when to fine-tune to deploying your custom model.

<!-- more -->

## When Should You Fine-Tune?

Before investing GPU hours and engineering time, you need to answer a fundamental question: **is fine-tuning the right solution for your problem?**

![Decision Flowchart](../assets/2026-01-04-finetuning-guide/decision_flowchart.svg)

Here's the decision framework:

| Challenge             | Best Solution      | Why?                                                                      |
| --------------------- | ------------------ | ------------------------------------------------------------------------- |
| **Missing knowledge** | RAG                | Models hallucinate facts. Retrieval provides grounded, up-to-date context |
| **Wrong format/tone** | Prompt Engineering | Modern models follow style instructions well via few-shot examples        |
| **Complex behavior**  | Fine-Tuning (SFT)  | When you need consistent "modes" without massive prompts                  |
| **Safety/preference** | Alignment (DPO)    | When outputs are correct but don't match preferences                      |
| **Latency/cost**      | Fine-Tuning        | Distill a large model into a smaller, faster one                          |

### The Economic Case

Fine-tuning shines in **high-volume, stable-requirement scenarios**. Consider this: a robust RAG system might require 2,000 tokens of context on every call (system prompt + retrieved docs + few-shot examples). That's your "context tax" on every request.

A fine-tuned model can internalize those instructions, reducing your prompt from 2,000 tokens to 50. At scale, this pays for the training compute within weeks.

!!! tip "The Hybrid Approach"
The industry sweet spot is often a fine-tuned smaller model (8B params) combined with lightweight RAG for facts. This often outperforms prompting a massive model (70B+) in both accuracy and cost.

---

## Understanding Fine-Tuning Methods

### Full Fine-Tuning vs PEFT

**Full Fine-Tuning (FFT)** updates all model weights. For a 7B model at 16-bit precision, you need roughly 112GB of VRAM just for training. This is prohibitive for most teams.

**Parameter-Efficient Fine-Tuning (PEFT)** changes the game by updating only a small subset of parameters while freezing the rest.

### LoRA: The Industry Standard

LoRA (Low-Rank Adaptation) is the foundational PEFT technique. The key insight: weight changes during fine-tuning have low "intrinsic rank."

![LoRA Architecture](../assets/2026-01-04-finetuning-guide/lora_architecture.svg)

Instead of updating a massive weight matrix **W** (dimension d×d), LoRA learns two smaller matrices:

- **A** (d × r) — down-projection
- **B** (r × d) — up-projection

The update becomes: **ΔW = B × A**

With rank `r=16`, this reduces trainable parameters by **~10,000x**, dropping VRAM from 120GB to 16GB.

### PEFT Methods Compared

| Method    | How It Works                                   | Memory Savings | Best For                           |
| --------- | ---------------------------------------------- | -------------- | ---------------------------------- |
| **LoRA**  | Low-rank matrices injected into frozen weights | ~10x           | General fine-tuning                |
| **QLoRA** | LoRA + 4-bit base model quantization           | ~20x           | Consumer GPUs (16-24GB)            |
| **DoRA**  | LoRA with magnitude/direction decomposition    | ~10x           | When LoRA hits performance ceiling |

#### When to Use Which?

- **LoRA**: Start here. It's fast, memory-efficient, and widely supported
- **QLoRA**: When you need to fine-tune 70B models on consumer hardware
- **DoRA**: When you need to match full fine-tuning quality on complex reasoning tasks

---

## Training Stages

Fine-tuning isn't a single step—it's a pipeline:

### 1. Supervised Fine-Tuning (SFT)

SFT teaches the model **how to behave**. You provide (prompt, response) pairs, and the model learns to generate responses matching your examples.

```python
# Example SFT data format
{
    "instruction": "Convert this to SQL",
    "input": "Get all users who signed up last month",
    "output": "SELECT * FROM users WHERE signup_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
}
```

**Data quality matters more than quantity.** 500-1,000 carefully curated examples often outperform 50,000 noisy ones.

### 2. Preference Alignment (DPO/ORPO)

When SFT isn't enough—the model technically answers correctly but in "wrong" ways (too verbose, unsafe, wrong tone)—you need preference alignment.

**DPO (Direct Preference Optimization)** has replaced complex RLHF pipelines. You provide preference pairs:

```python
{
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing uses qubits...",   # Preferred response
    "rejected": "Well, it's complicated..."        # Non-preferred response
}
```

**ORPO** and **SimPO** are newer methods that simplify this further—ORPO combines SFT and alignment into one stage, while SimPO removes the need for a reference model entirely.

---

## Fine-Tuning Frameworks

The ecosystem has consolidated around four major tools:

### Unsloth — Speed & Efficiency Champion

```python
from unsloth import FastLanguageModel

# 2x faster training, 60% less VRAM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

**Best for:** Single-GPU training, prototyping, Colab notebooks, anyone paying for GPU hours.

**Key advantage:** Custom Triton kernels make it 2-5x faster than standard HuggingFace implementations.

### Axolotl — Config-Driven Production

```yaml
# config.yaml - no code required
base_model: meta-llama/Meta-Llama-3-8B
adapter: qlora
lora_r: 32
lora_alpha: 16
datasets:
    - path: data/my_data.jsonl
      type: alpaca
sample_packing: true
```

Run with: `accelerate launch -m axolotl.cli.train config.yaml`

**Best for:** Production pipelines, multi-GPU clusters, reproducible experiments.

**Key advantage:** YAML configs are version-controllable and shareable.

### Framework Comparison

| Feature       | Unsloth            | Axolotl         | TRL       | Torchtune       |
| ------------- | ------------------ | --------------- | --------- | --------------- |
| **Strength**  | Speed & efficiency | Multi-GPU scale | Ecosystem | PyTorch native  |
| **Speed**     | Fastest (2-5x)     | High            | Moderate  | High            |
| **Multi-GPU** | Growing            | Excellent       | Good      | Excellent       |
| **Config**    | Python             | YAML            | Python    | Python          |
| **Best for**  | Local/Colab        | Clusters        | Research  | PyTorch purists |

---

## Practical Demo: Fine-Tuning with Unsloth

Let's walk through a complete example using my [unsloth-finetune-demo](https://github.com/slavadubrov/unsloth-finetune-demo) repository. This demo fine-tunes Nemotron-Nano for function calling.

![Training Pipeline](../assets/2026-01-04-finetuning-guide/training_pipeline.svg)

### Quick Start

```bash
# Clone and setup
git clone https://github.com/slavadubrov/unsloth-finetune-demo.git
cd unsloth-finetune-demo

# Install with uv (recommended)
uv sync

# Run fine-tuning (quick test)
uv run finetune --max-samples 1000
```

### Configuration Deep Dive

The key configuration lives in `config.py`:

```python
# Model & Dataset
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"  # 4B params, 128K context
DATASET_NAME = "glaiveai/glaive-function-calling-v2"   # 113K examples

# LoRA Configuration
LORA_R = 16        # Rank - higher = smarter but more VRAM
LORA_ALPHA = 32    # Scaling factor - usually 2x LORA_R
MAX_SEQ_LENGTH = 4096

# Target all linear layers for best quality
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

!!! note "The Alpha/Rank Ratio"
Industry best practice in 2025: set **alpha = 2 × rank** (e.g., rank=16, alpha=32). This provides stronger weight updates without destabilizing training.

### Core Training Code

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # Magic sauce for memory
)

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=4096,
    packing=True,  # Crucial for efficiency!
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
    ),
)
trainer.train()
```

---

## Fine-Tuning with Axolotl

For production and multi-GPU setups, Axolotl's config-first approach excels:

```yaml
# axolotl_config.yaml
base_model: meta-llama/Meta-Llama-3-8B
model_type: LlamaForCausalLM

# QLoRA configuration
load_in_4bit: true
adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

# Dataset
datasets:
    - path: data/training_data.jsonl
      type: alpaca

# Training settings
sequence_len: 4096
sample_packing: true # Critical for speed!
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0002
num_epochs: 3

# Hardware
bf16: true
flash_attention: true
```

Run training:

```bash
accelerate launch -m axolotl.cli.train axolotl_config.yaml
```

---

## Output Formats: LoRA vs Merged vs GGUF

After training, you have three export options:

![Output Formats](../assets/2026-01-04-finetuning-guide/output_formats.svg)

### 1. LoRA Adapter (Default)

```bash
uv run finetune  # Saves ~100-500MB adapter
```

- **Size:** ~100-500 MB
- **Best for:** Development, testing, multiple adapters on one base model
- **Flexibility:** Swap adapters without re-downloading the base model

### 2. Merged Model

```bash
uv run finetune --merge  # Creates standalone ~8-16GB model
```

- **Size:** ~8-16 GB (full 16-bit weights)
- **Best for:** Sharing on HuggingFace, vLLM serving, simple deployment
- **Trade-off:** Larger storage, but no separate base model needed

### 3. GGUF Format

```bash
uv run finetune --gguf q4_k_m  # Creates ~2-4GB quantized model
```

- **Size:** ~2-4 GB (Q4_K_M quantization)
- **Best for:** CPU inference, Ollama, llama.cpp, edge deployment
- **Options:** `q4_k_m` (balanced), `q5_k_m` (higher quality), `q8_0` (near-lossless)

---

## Serving Your Fine-Tuned Model

### With vLLM (Production)

```bash
# Requires merged model format
vllm serve ./outputs/unsloth-nemotron-function-calling-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

Query via OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="unsloth-nemotron-function-calling-merged",
    messages=[{"role": "user", "content": "Book a flight to Tokyo"}]
)
```

### With Ollama (Local)

```bash
# Create Modelfile
echo 'FROM ./outputs/unsloth-nemotron-function-calling-gguf/model-q4_k_m.gguf' > Modelfile

# Import to Ollama
ollama create my-function-model -f Modelfile

# Run
ollama run my-function-model
```

### With llama.cpp (CPU)

```bash
./main -m ./outputs/model-q4_k_m.gguf \
    -p "What's the weather in Tokyo?" \
    --ctx-size 4096
```

---

## Evaluation

Training is easy; knowing if it worked is hard. Here's the evaluation stack:

### Automated Benchmarks

Use `lm-evaluation-harness` for standardized testing:

```bash
lm_eval --model hf \
    --model_args pretrained=./outputs/merged-model \
    --tasks hellaswag,arc_easy,mmlu \
    --batch_size 8
```

### LLM-as-Judge

For subjective quality, use a larger model to evaluate:

```python
judge_prompt = """
Rate this response from 1-5 on:
- Relevance
- Accuracy
- Formatting

Response: {model_output}
Expected: {ground_truth}
"""
```

### Domain-Specific Eval

Create a held-out test set of real examples from your use case. This is the most important evaluation—generic benchmarks won't tell you if your function-calling model actually works.

---

## Key Takeaways

1. **Don't default to fine-tuning.** Try RAG and prompting first
2. **Use QLoRA** to fine-tune 70B models on consumer GPUs
3. **Quality over quantity** in training data—1K great examples > 50K noisy ones
4. **Sample packing** is the single biggest training speedup
5. **Start with Unsloth** for prototyping, **Axolotl** for production
6. **Export to GGUF** for local/edge deployment

---

## References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Demo Repository](https://github.com/slavadubrov/unsloth-finetune-demo)
- [Research Notebook](https://notebooklm.google.com/notebook/f6bfdb56-8949-4929-87e4-ab6dee31a4a8) - NotebookLM notebook created during article research
