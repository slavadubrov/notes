---
title: Choosing the Right Open-Source LLM Variant & File Format
date: 2025-05-11
tags: [guide, llm]
summary: A practical guide to navigating open-source LLM variants and file formats, covering model types (Base, Instruct, Distill, QAT, MoE), quantization formats (GGUF, GPTQ, AWQ), and hardware-specific recommendations for optimal performance.
---

# Choosing the Right Open-Source LLM Variant & File Format

---

## 1. Why all these tags exist

Open-source LLMs are shipped in **two axes of variation**:

1. **Training / fine-tuning style** – the suffixes you see in model names (`-Instruct`, `-Distill`, `-A3B`, …) tell you how the checkpoint was produced and what it's good at.
2. **File & quantization format** – the extension (`.gguf`, `.gptq`, …) tells you how the weights are packed for inference on different hardware.

Understanding both axes lets you avoid downloading 20 GB for nothing or fighting CUDA errors at 3 a.m.

<!-- more -->

---

## 2. Model-type cheat sheet

| Tag                     | What it means (plain English)                                                                            | Typical use-cases                                                                                                | Trade-offs                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Base**                | Raw pre-trained LLM, not aligned to human instructions.                                                  | Further fine-tuning, research, creative (unfiltered) generation.                                                 | Needs prompts wrapped carefully; can refuse or ramble.         |
| **Instruct / Chat**     | Extra supervised + RLHF rounds so the model _follows instructions_.                                      | Agents, RAG, chatbots, function-calling, day-to-day coding help.                                                 | Slightly larger, a bit slower; may be less creative.           |
| **Distill / Distilled** | A smaller "student" model trained to mimic a big "teacher."                                              | Mobile / edge devices, cost-sensitive SaaS, latency-critical endpoints.                                          | Some loss in reasoning depth; great token-per-watt ratio.      |
| **QAT**                 | _Quantization-Aware Training_: the model was re-trained while already in low-bit form.                   | When you need 4- or 8-bit weights **and** near-FP16 accuracy (e.g., on consumer GPUs/CPUs).                      | Training cost is higher than plain post-training quantization. |
| **A3B / A22B** (MoE)    | _Mixture-of-Experts_ with **A**ctivated **X B** parameters (e.g., "A3B" = 3 B active out of 30 B total). | You want "big-brain" performance but only pay inference for a slice of it - ideal for local GPUs with ≤24 GB VRAM. | Heavier disk size; not every framework supports MoE yet.       |

> **Rule of thumb:**
>
> * _Start with an **Instruct** model.
> * If you hit latency or memory limits, drop to a **Distill** or **A3B** MoE variant.
> * If you still need more speed, look for models explicitly tagged **QAT** or download a lower-bit quantized file._

---

## 3. Picking a **file / quantization** format

| Format                                  | Best for                                                                    | Highlights                                                                                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **GGUF** (`.gguf`)                      | **General local inference** (CPU or GPU) via `llama.cpp`, Ollama, LM Studio | Successor to GGML; single binary with rich metadata & prompt template; supports 2-8 bit "K-quants"; now the _de-facto_ community standard. |
| **GPTQ** (`.safetensors` + `gptq.json`) | Fast **GPU** inference at 3-/4-bit (NVIDIA/AMD)                             | Block-wise second-order quantization; large ecosystem in `autoGPTQ`.                                                                       |
| **AWQ** (`awq.safetensors`)             | 4-bit **CPU/GPU** with very low memory                                      | Activation-aware; fewer outliers than GPTQ ⇒ higher accuracy; easy via `autoawq`.                                                          |
| **EXL2** (`.exl2`)                      | **GPU** power-users who want mixed-precision layers (2–8 bit)               | Allows per-layer bit-mixing; stellar throughput with ExLlama v2.                                                                           |
| **PyTorch / Safetensors (FP16/BF16)**   | Cloud GPUs, further fine-tuning                                             | Full fidelity, biggest VRAM & disk footprint.                                                                                              |
| **ONNX / TFLite**                       | Edge devices, mobile                                                        | Graph optimisations, hardware-specific accelerators.                                                                                       |

> **Tip:** If you're uncertain, grab a **Q4_K_M GGUF** first; it fits 13 B dense or 30 B-A3B MoE models into ~6–8 GB and runs on an 8-GB VRAM GPU or a modern CPU.

---

## 4. Decision flow (quick & opinionated)

1. **What's my hardware?**

      * `<8 GB VRAM / only CPU` → GGUF Q4_K_M or AWQ 4-bit.
      * `≥16 GB VRAM` → GPTQ 3-bit or GGUF Q3_K_S.
      * `H100/A100 class` → FP16 Safetensors or ONNX.

2. **What's my task?**

      * **Chat / retrieval-augmented generation** → Instruct or Chat.
      * **Batch creative writing** → Dense Base model with a creative prompt.
      * **Private edge device** → Distill or A3B + QAT/AWQ quantization.

3. **Do I care about devops simplicity?**
      * Yes → GGUF (single file, works everywhere).
      * No, I'll optimise every ms → GPTQ / EXL2 on GPU.

---

## 5. Gotchas to remember

* **Not all quantizers understand MoE gating.** Today `llama.cpp` (and forks) handle A3B/A22B in GGUF; `autoGPTQ` & `autoAWQ` support is still evolving.
* **QAT ≠ plain quantization.** A QAT model in 4-bit often matches an 8-bit PTQ model's accuracy - don't double-quantize it.
* **Distill ≠ small parameter count only.** A distilled 7 B model may outperform a vanilla dense 13 B because knowledge from a stronger teacher was baked in.
* **File format ≠ quantization method.** You can have a GGUF in 8-bit (near-FP16) or 2-bit (experimental); likewise, GPTQ can hold 4-bit or 8-bit blocks.

---

## 6. TL;DR

> **If you just want something that works:**
>
> * Download a **`<model-name>-Instruct.Q4_K_M.gguf`** build.
> * Run it with `llama.cpp`, LM Studio, or Ollama.
> * If it's still slow, try an **A3B** MoE flavour or an **AWQ** 4-bit file.
