---
title: "Scaling Large Language Models - Practical Multi-GPU and Multi-Node Strategies for 2025"
date:
  created: 2025-05-04
  updated: 2025-05-04
tags: [LLM, Distributed Training, Deep Learning, GPU, Parallelism]
description: A practical guide to scaling large language models across multiple GPUs and nodes, with real-world strategies from Hugging Face's Ultra-Scale Playbook.
author: Viacheslav Dubrov
---

# Scaling Large Language Models - Practical Multi-GPU and Multi-Node Strategies for 2025

The race to build bigger, better language models continues at breakneck speed. Today's state-of-the-art models require massive computing resources that no single GPU can handle. Whether you're training a custom LLM or deploying one for inference, understanding how to distribute this workload is essential.

This guide walks through practical strategies for scaling LLMs across multiple GPUs and nodes, incorporating insights from Hugging Face's [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).

<!-- more -->

## Why Scaling Matters

Modern LLMs have outgrown single GPUs. Here's why scaling is no longer optional:

- **Model size**: A 70B parameter model needs ~140GB in FP16 format - that's nearly 2x what an A100 (80GB) can hold
- **Training time**: Even with 8 top-tier A100 GPUs, training a 13B model from scratch takes weeks
- **Context length**: Long contexts (32k+ tokens) easily exceed single-GPU memory limits
- **Inference speed**: For production workloads, distributing inference reduces latency and increases throughput

The solution? Split the workload across multiple GPUs. Let's explore how.

## 1. Parallelism Techniques Explained Simply

### 1.1 Data Parallelism (DP)

**The idea:** Multiple workers with identical instruction manuals (the model), each working on different examples.

**How it works:**

1. Each GPU gets a complete copy of the model
2. Each GPU processes different batches of data
3. After computing gradients, all GPUs synchronize by averaging their gradients
4. Everyone updates their model copy with the averaged gradients

**When to use it:**

- Your model fits comfortably on a single GPU
- You want to process more data faster
- You need the simplest distributed setup with minimal code changes

**Limitation:** Memory inefficient - every GPU stores the full model, so you're not saving memory, just increasing throughput.

```kroki-mermaid
flowchart LR
    subgraph DataLoader
        D[Global batch] --> |split| MB1[Micro-batch 1]
        D[Global batch] --> |split| MB2[Micro-batch 2]
        D[Global batch] --> |split| MBN[Micro-batch N]
    end
    subgraph GPU1
        MB1[Micro-batch 1] --> M1[Model copy]
    end
    subgraph GPU2
        MB2[Micro-batch 2] --> M2[Model copy]
    end
    subgraph GPUN
        MBN[Micro-batch N] --> MN[Model copy]
    end
    M1[Model copy] & M2[Model copy] & MN[Model copy] --> G[All-reduce -> average gradients]
    G[All-reduce -> average gradients] --> U[Synchronised weight update]
```

**Tools**: [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html), [Horovod](https://horovod.ai/).

### 1.2 Fully Sharded Data Parallelism (FSDP)

**The idea:** Like Data Parallelism, but memory-efficient. Each worker keeps only part of the instruction manual and borrows pages from colleagues when needed.

**How it works:**

1. Model parameters, gradients, and optimizer states are **sharded** (split) across all GPUs
2. During forward pass: each GPU gathers the parameters it needs from other GPUs
3. After using them, it discards those borrowed parameters to save memory
4. During backward pass: same gathering happens for gradient computation
5. After backward pass: gradients are reduced and each GPU updates only its own parameter shard

**When to use it:**

- Your model is too large for a single GPU (typically >10B parameters)
- You want to train bigger models without changing your code much
- You're working on a single machine with multiple GPUs

**Real-world impact:** FSDP lets you train models 4-8x larger than what fits on one GPU.

```kroki-mermaid
flowchart TD
    %% GPU-local state
    subgraph "GPU 1"
        direction TB
        P1[Param shard P₁]
        G1[Grad shard G₁]
        O1[Opt shard O₁]
    end
    subgraph "GPU 2"
        direction TB
        P2[Param shard P₂]
        G2[Grad shard G₂]
        O2[Opt shard O₂]
    end
    subgraph "GPU N"
        direction TB
        PN[Param shard Pₙ]
        GN[Grad shard Gₙ]
        ON[Opt shard Oₙ]
    end

    %% Mini-batch pipeline
    start([Start micro-batch]) --> gather[Step 1: All-Gather]
    gather --> fwd[Step 2: Forward compute]
    fwd --> reshard[Step 3: Re-shard P]
    reshard --> bwd[Step 4: Backward compute]
    bwd --> reduce[Step 5: Reduce-Scatter]
    reduce --> update[Step 6: Optimizer update]

    %% Collective edges (dotted to indicate broadcast)
    P1 -.-> gather
    P2 -.-> gather
    PN -.-> gather
```

**Tools**: [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html), [DeepSpeed ZeRO-3](https://www.deepspeed.ai/tutorials/zero/).

### 1.3 Tensor Parallelism (TP)

**The idea:** Split individual layers across GPUs - like dividing a massive spreadsheet calculation where each person computes a few columns.

**How it works:**

1. Take a single layer's weight matrix and split it into chunks
2. Each GPU gets one chunk and computes its portion of the output
3. Results are combined (via all-reduce or concatenation) before passing to the next layer
4. This happens at **every** layer in the model

**When to use it:**

- Individual layers are too large even with FSDP (e.g., huge attention or FFN layers)
- You have fast GPU-to-GPU connections (NVLink/NVSwitch)
- You're working within a single node (TP doesn't scale well across nodes due to communication overhead)

**Sweet spot:** TP degree of 2-8 within a single machine with NVLink.

```kroki-mermaid
flowchart LR
    A[X activations] --> |broadcast| X1[GPU1]
    A --> |broadcast| X2[GPU2]
    A --> |broadcast| XN[GPUN]
    subgraph ShardedWeights
        W1[W shard₁] --- X1
        W2[W shard₂] --- X2
        WN[W shardₙ] --- XN
    end
    X1 --> P1[Partial Y₁]
    X2 --> P2[Partial Y₂]
    XN --> PN[Partial Yₙ]
    P1 & P2 & PN --> C[Concat / reduce -> Y]
```

**Tools**: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [ColossalAI](https://github.com/hpcaitech/ColossalAI).

### 1.4 Pipeline Parallelism (PP)

**The idea:** Split the model vertically by layers - like an assembly line where each station handles specific layers.

**How it works:**

1. Divide your model into stages (e.g., layers 1-10, 11-20, 21-30)
2. Assign each stage to a different GPU
3. Send micro-batches through the pipeline: GPU 1 processes batch 1, sends output to GPU 2, then starts on batch 2
4. Multiple micro-batches flow through simultaneously to keep all GPUs busy

**When to use it:**

- Very deep models that don't fit on available GPUs even with FSDP
- Multi-node training where inter-node bandwidth is limited
- Combined with TP and FSDP for massive models

**Challenge:** Pipeline "bubbles" (idle time) at the start and end of each batch. Use multiple micro-batches to minimize this.

```kroki-mermaid
sequenceDiagram
    participant S0 as GPU-Stage 0 (Layers 1-4)
    participant S1 as GPU-Stage 1 (Layers 5-8)
    participant S2 as GPU-Stage 2 (Layers 9-12)
    Note over S0,S2: ← time ->
    S0->>S0: Fwd/Bwd µ-batch 0
    S0->>S1: send activations
    S1->>S1: Fwd/Bwd µ-batch 0
    S1->>S2: send activations
    S0->>S0: Fwd/Bwd µ-batch 1
    S2->>S2: Fwd/Bwd µ-batch 0
```

**Tools**: [DeepSpeed PP](https://www.deepspeed.ai/tutorials/pipeline/), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [GPipe](https://arxiv.org/abs/1811.06965).

### 1.5 Context Parallelism (CP)

**The idea:** For handling extremely long sequences - different people read different paragraphs of a book, then share key information.

**How it works:**

1. Split a long sequence (e.g., 64K tokens) across multiple GPUs (e.g., 4 GPUs × 16K tokens each)
2. Each GPU runs self-attention on its local chunk
3. GPUs exchange keys and values to compute cross-attention (how tokens in one chunk relate to tokens in other chunks)
4. Results are merged to produce the final output

**When to use it:**

- Processing very long contexts (64K, 128K, or even 1M+ tokens)
- Document analysis, long-form code generation, or book-length reasoning
- When context length is the bottleneck, not model size

**Real-world impact:** Context Parallelism enables 100K+ token processing on consumer hardware that would otherwise max out at 8K tokens.

```kroki-mermaid
flowchart LR
    subgraph Input["Input Sequence"]
        S[Sequence 0-8191 tokens]
    end

    subgraph CrossGPU["Cross-GPU Processing"]
        direction LR
        subgraph GPU1["GPU 1"]
            direction TB
            T0[Tokens 0-4095]
            A0[Self-Attention Block]
            T0 --> A0
        end

        subgraph GPU2["GPU 2"]
            direction TB
            T1[Tokens 4096-8191]
            A1[Self-Attention Block]
            T1 --> A1
        end

        GPU1 <-->|Exchange Keys/Values| GPU2
    end

    subgraph Output["Output Processing"]
        M[Merge Logits]
        O[Output Sequence]
        M --> O
    end

    S --> |Split| T0
    S --> |Split| T1

    A0 --> M
    A1 --> M
```

**Tools**: [Picotron](https://github.com/huggingface/picotron), [Nanotron](https://github.com/huggingface/nanotron).

### 1.6 Expert Parallelism (Mixture of Experts - MoE)

**The idea:** Specialized consultants - instead of activating the entire model for every input, each token gets routed only to the "experts" it needs.

**How it works:**

1. Replace dense feed-forward layers with multiple "expert" networks (e.g., 8 or 64 experts)
2. A gating network decides which experts (usually top-2) should process each token
3. Only those selected experts activate for that token
4. Different experts can live on different GPUs

**When to use it:**

- You want a model with 100B+ total parameters but only want to activate 13B per token
- You need better parameter efficiency than dense models
- You're okay with more complex training dynamics

**Real-world examples:** Mixtral-8x7B (56B total params, 13B active), Grok, DeepSeek-V2.

**Trade-off:** More parameters with less compute per token, but training can be trickier due to load balancing between experts.

```kroki-mermaid
flowchart LR
    subgraph Input_Tokens["Input Tokens"]
        T1["T₁"]
        T2["T₂"]
        T3["T₃"]
    end
    G["Gating Network"]
    subgraph Experts["Experts"]
        E1["Expert 1"]
        E2["Expert 2"]
        E3["Expert 3"]
        E4["⋯"]
    end
    T1 --> G
    T2 --> G
    T3 --> G
    G -->|top-k routes| E1
    G -->|top-k routes| E2
    G -->|top-k routes| E3
    E1 & E2 & E3 --> O["Concatenate + Mix"]
```

**Tools**: [Picotron](https://github.com/huggingface/picotron), [Nanotron](https://github.com/huggingface/nanotron).

### Quick Comparison: Which Parallelism Should You Use?

| Technique | What It Splits | Best For | Memory Savings | Communication Cost |
|-----------|---------------|----------|----------------|-------------------|
| **Data Parallelism (DP)** | Data batches | Models that fit on 1 GPU | None (copies model) | Low (only gradients) |
| **FSDP** | Model + optimizer + gradients | Models too big for 1 GPU | High (4-8x) | Medium |
| **Tensor Parallelism (TP)** | Individual layers | Huge layers, fast GPUs | Medium | High (per layer) |
| **Pipeline Parallelism (PP)** | Layer groups (stages) | Very deep models | Medium | Low (between stages) |
| **Context Parallelism (CP)** | Sequence length | Long contexts (64K+ tokens) | High (for activations) | Medium |
| **Expert Parallelism (MoE)** | Experts in MoE layers | Massive sparse models | None (more params, less FLOPs) | Medium |

**Rule of thumb:** Start with FSDP. Add TP if individual layers are too big. Add PP if you need multiple nodes. Add CP if context length is your bottleneck.

## 2. Practical Training Strategies

Now that you understand the techniques, here's what to actually do based on your hardware setup.

### 2.1 Single Machine (2-8 GPUs)

**Recommended approach:** FSDP, optionally + TP

**What to do:**

1. Start with pure FSDP using PyTorch FSDP or DeepSpeed ZeRO-2/ZeRO-3
2. If your model has huge attention or FFN layers that still don't fit, add TP=2
3. Use Hugging Face `accelerate` or PyTorch `torchrun` for easy setup

**Hardware-specific tips:**

- Consumer GPUs (RTX 4090, etc.) with PCIe: Stick to TP=1 or TP=2 max
- Server GPUs (A100, H100) with NVLink: You can efficiently use TP=2 to TP=4
- 8 GPUs in one box: FSDP alone often works great for models up to 70B

### 2.2 Small Cluster (2-16 nodes, ≤128 GPUs)

**Recommended approach:** 2D or 3D parallelism (TP + FSDP, optionally + PP)

**What to do:**

1. Use TP within each node (e.g., TP=4 or TP=8 per node with NVLink)
2. Use FSDP across nodes for data parallelism
3. If your model is extremely deep, add PP to split it vertically across nodes

**Why this works:**

- Fast intra-node connections (NVLink) handle TP's high communication needs
- Slower inter-node connections (InfiniBand) only need to sync FSDP shards
- Minimizes cross-node bandwidth requirements

**Pro tip:** When using Pipeline Parallelism, set your number of micro-batches to at least 4× your pipeline degree to keep GPUs busy and minimize "bubbles."

### 2.3 Large Cluster (Hundreds or Thousands of GPUs)

**Recommended approach:** 4D parallelism (DP × TP × PP × CP)

**What to do:**

1. Combine all four parallelism strategies to handle the largest models
2. Carefully map parallelism strategies to your hardware topology
3. Use tools like Megatron-LM or Nanotron that support 4D parallelism out of the box

**When you need this:**

- Training models with 70B+ parameters and 32K+ context windows
- Pretraining from scratch (not fine-tuning)
- Production-scale model training at big labs

**Performance expectations:**

- With good InfiniBand networking: ~70-80% scaling efficiency
- With excellent setup and tuning: ~85% scaling efficiency possible

**Real-world example:** Training a 70B model with 32K context on 512 GPUs:

- TP=8 (within each 8-GPU node)
- PP=4 (pipeline across 4 nodes)
- CP=4 (split context across 4 chunks)
- DP=4 (data parallelism for throughput)
- Total: 8 × 4 × 4 × 4 = 512 GPUs

## 3. Practical Tools Worth Learning

Here's a quick guide to the most useful tools and when to reach for them:

| Tool | When to Use It | Learning Curve | Best For |
| ---- | -------------- | -------------- | -------- |
| **Hugging Face Accelerate** | Any distributed training with minimal code changes | ★☆☆☆☆ | Beginners, quick prototypes |
| **PyTorch FSDP** | Medium-large models (1-30B) on single node | ★★☆☆☆ | Most common use case |
| **DeepSpeed ZeRO** | Multi-node training with good documentation | ★★★☆☆ | Production training |
| **Megatron-LM** | Very large models (70B+), 3D/4D parallelism | ★★★★☆ | Advanced/production at scale |
| **Nanotron** | Learning/research on modern parallelism strategies | ★★★☆☆ | Education, experimentation |
| **vLLM** | Fast inference with PagedAttention and KV caching | ★★☆☆☆ | Serving models in production |
| **TensorRT-LLM** | Maximum inference speed on NVIDIA GPUs | ★★★★☆ | Production inference optimization |

**My recommendation for getting started:** Start with Hugging Face Accelerate for learning, then graduate to PyTorch FSDP or DeepSpeed when you need more control.

## 4. Making the Right Choice: A Decision Framework

Still not sure what to use? Follow this decision tree:

**Step 1: Does your model fit on a single GPU?**

- ✅ **Yes** → Use standard training (no parallelism needed)
- ❌ **No** → Continue to Step 2

**Step 2: Do you have multiple GPUs in one machine?**

- ✅ **Yes** → Start with FSDP
- ❌ **No** → You'll need a cluster or smaller model (skip to Step 4)

**Step 3: Is FSDP alone enough?**

- ✅ **Yes** → You're done! Use pure FSDP
- ❌ **No, individual layers are too big** → Add TP=2 or TP=4
- ❌ **No, context is too long** → Add CP

**Step 4: Training across multiple nodes?**

- Start with: TP within nodes + FSDP across nodes
- If model is very deep: Add PP to split layers across nodes
- If you have 100+ GPUs and long contexts: Consider 4D parallelism (TP + PP + DP + CP)

**Visual decision tree:**

```kroki-mermaid
flowchart TD
    start([Start: Need to scale LLM?]) --> fit{Model fits on<br/>single GPU?}
    
    fit -->|Yes| done1[✅ Standard training<br/>No parallelism needed]
    fit -->|No| multi{Multiple GPUs<br/>in one machine?}
    
    multi -->|No| cluster[Need cluster or<br/>smaller model]
    multi -->|Yes| fsdp[Start with FSDP]
    
    fsdp --> enough{FSDP enough?}
    enough -->|Yes| done2[✅ Use pure FSDP]
    enough -->|Layers too big| tp[Add TP=2 or TP=4<br/>within node]
    enough -->|Context too long| cp[Add Context<br/>Parallelism]
    
    tp --> done3[✅ Use FSDP + TP]
    cp --> done4[✅ Use FSDP + CP]
    
    cluster --> multinode[Multi-node setup]
    multinode --> hybrid[TP inside nodes<br/>+ FSDP across nodes]
    
    hybrid --> depth{Model very<br/>deep?}
    depth -->|No| done5[✅ Use 2D: TP + FSDP]
    depth -->|Yes| pp[Add Pipeline<br/>Parallelism]
    
    pp --> scale{100+ GPUs +<br/>long context?}
    scale -->|No| done6[✅ Use 3D: TP + PP + FSDP]
    scale -->|Yes| done7[✅ Use 4D: TP + PP + DP + CP]
    
    style done1 fill:#90EE90
    style done2 fill:#90EE90
    style done3 fill:#90EE90
    style done4 fill:#90EE90
    style done5 fill:#90EE90
    style done6 fill:#90EE90
    style done7 fill:#90EE90
```

## 5. The Ultra-Scale Cheatsheet

For a comprehensive visual summary, check out this guide from Hugging Face's team:

![Ultra-Scale LLM Cheatsheet](https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg)

## Conclusion

Scaling LLMs is both an art and a science. The key takeaways:

1. **Start simple:** Most people should begin with FSDP. It handles the majority of use cases.
2. **Add complexity only when needed:** Don't jump straight to 4D parallelism unless you're training at massive scale.
3. **Match strategy to hardware:** TP works best within nodes, FSDP across nodes, PP for extreme depth.
4. **Tools matter:** Use Accelerate to learn, FSDP or DeepSpeed for production.

The techniques here follow logical patterns based on hardware constraints and model architecture. With the right approach, you can scale from a single GPU to thousands, training models that would have been impossible just a few years ago.

**Further resources:**

- [Hugging Face Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Interactive guide with more details
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) - Official getting started guide
- [DeepSpeed Tutorials](https://www.deepspeed.ai/tutorials/) - Comprehensive DeepSpeed documentation
