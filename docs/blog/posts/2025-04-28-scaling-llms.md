---
title: Scaling Large Language Models. Multi-GPU and Multi-Node Strategies in 2025
date: 2025-04-28
tags: [LLM, Distributed Training, Deep Learning, GPU, Parallelism]
summary: A comprehensive guide on scaling large language models using multi-GPU and multi-node strategies, incorporating insights from Hugging Face's Ultra-Scale Playbook.
---

# Scaling Large Language Models. Multi-GPU and Multi-Node Strategies in 2025

As LLMs continue to grow in complexity and size, efficient training and inference require leveraging multiple GPUs and, often, multiple systems. This guide explores prevalent strategies and tools in 2025 that facilitate such scalability, incorporating insights from Hugging Face's [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).

<!-- more -->

## 1. Parallelism Techniques

### 1.1 Data Parallelism (DP)

In classic data-parallel training **every GPU keeps a full copy of the model**.
A large batch is split into _N_ micro-batches; each rank runs forward + backward on its piece and then gradients are **all-reduced (averaged)** so that all replicas stay in sync before the optimizer step.

**Key ideas**

- **Simplicity first** - almost zero code changes; works everywhere.
- **Redundant memory** - O(total params) on every GPU, so model size is bounded by a single card.
- **Communication cost** - one gradient all-reduce per step (~2 x parameter size).
- **Throughput scaling** - global batch = per-GPU batch x _N_; watch out for generalization when scaling batch too far.

**Mermaid Diagram**

```mermaid
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
    M1[Model copy] & M2[Model copy] & MN[Model copy] --> G[All-reduce → average gradients]
    G[All-reduce → average gradients] --> U[Synchronised weight update]
```

- **Tools**: [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html), [Horovod](https://horovod.ai/).

#### 1.1.1 Fully Sharded Data Parallelism (FSDP)

FSDP is a type of data-parallel training, but unlike traditional data-parallel, which maintains a per-GPU copy of a model's parameters, gradients and optimizer states, it shards all of these states across data-parallel workers and can optionally offload the sharded model parameters to CPUs. [[Pytorch](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api)]

**Key ideas**

- **Memory scaling**: O(total params / NGPU) - enables multi-billion-parameter models to fit on 24 GB cards.
- **Zero redundancy**: No GPU ever holds a full copy of the model; identical to DeepSpeed ZeRO-3.
- **Overlap compute & communication**: PyTorch overlaps the all-gather with computation to hide latency.
- **Granularity control**: You can wrap the whole model or nest FSDP wrappers on sub-modules for finer control.

**Mermaid Diagram**

```mermaid
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

> **Note**: In the diagram above, P represents Parameters (model weights), G represents Gradients, and O represents Optimizer states. These are the three main components of model state that are sharded across GPUs in FSDP.

- **Use Case**: Training very large models (> 10 B parameters) that do not fit on a single GPU.
- **Tools**: [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html), [DeepSpeed ZeRO-3](https://www.deepspeed.ai/tutorials/zero/).

### 1.2 Tensor Parallelism (TP)

TP **slices individual weight tensors across GPUs** so each rank stores only a shard (e.g., specific columns or rows). During the forward pass each rank computes its partial matrix multiplication; intermediate activations are **all-gathered or reduced** to produce the layer output.

**Key ideas**

- **Shards compute & memory** - enables layers larger than a single GPU.
- **Orthogonal to DP** - combine TP x DP for higher scale (Megatron uses a 2-D «TP x DP» grid).
- **Best for dense GEMM(General Matrix Multiplication)-heavy blocks** - attention & FFN matrices.

**Mermaid Diagram**

```mermaid
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
    P1 & P2 & PN --> C[Concat / reduce → Y]
```

- **Tools**: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [ColossalAI](https://github.com/hpcaitech/ColossalAI).

---

### 1.3 Pipeline Parallelism (PP)

PP **distributes consecutive blocks of layers to different GPUs** (pipeline stages).
Micro-batches flow through stages like an assembly line, so computation and communication overlap.

**Key ideas**

- **Memory relief** - each rank stores only its slice of the network depth.
- **Bubble latency** - first and last few micro-batches see idle time; mitigate with enough micro-batches or sophisticated scheduling.
- **Composable with DP/TP** - e.g., 2 x TP inside each stage x 4 x PP across depth.

**Mermaid Diagram**

```mermaid
sequenceDiagram
    participant S0 as GPU-Stage 0 (Layers 1-4)
    participant S1 as GPU-Stage 1 (Layers 5-8)
    participant S2 as GPU-Stage 2 (Layers 9-12)
    Note over S0,S2: ← time →
    S0->>S0: Fwd/Bwd µ-batch 0
    S0->>S1: send activations
    S1->>S1: Fwd/Bwd µ-batch 0
    S1->>S2: send activations
    S0->>S0: Fwd/Bwd µ-batch 1
    S2->>S2: Fwd/Bwd µ-batch 0
```

- **Tools**: [DeepSpeed PP](https://www.deepspeed.ai/tutorials/pipeline/), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [GPipe](https://arxiv.org/abs/1811.06965).

### 1.4 Context Parallelism (CP)

CP (a.k.a. **sequence parallelism**) splits the **sequence length / token dimension** across GPUs so each rank handles a contiguous block of tokens, enabling context windows far beyond single-GPU memory.

**Key ideas**

- **Long-context enabler** - reach 32 k, 64 k+ tokens.
- **Attention communication** - GPUs exchange keys/values (all-gather) for cross-token attention each layer.
- **Pairs well with TP & PP** - CP handles tokens while others handle model axes.
- **Early-stage technique** - currently in research code (Picotron / Nanotron).

**Mermaid Diagram**

```mermaid
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

- **Tools**: [Picotron](https://github.com/huggingface/picotron), [Nanotron](https://github.com/huggingface/nanotron).

### 1.5 Expert Parallelism (or Mixture of Experts)

MoE layers contain dozens (or even hundreds) of parallel **experts** (small feed-forward sub-networks).
For every token a lightweight **gating network** selects the top-_k_ experts, so only that subset runs.
This decouples **model capacity** (total parameters) from **per-token compute/FLOPs**.

**Key ideas**

- **Sparse activation** - With _k = 2_ out of 64 experts each token touches ~3 % of the parameters, yet the model still "sees" the full capacity during training.
- **Conditional computation** - Tokens route to different experts, letting each specialize (e.g., code vs poetry).
- **Load-balancing loss** - Extra loss term keeps expert usage uniform to avoid stragglers.
- **Scale to trillions** - Total parameters scale linearly with #experts, compute stays roughly constant.

**Mermaid Diagram**

```mermaid
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

- **Use Case**: Scaling to 100 B-1 T+ parameters without proportional compute cost.
- **Tools**: [DeepSpeed-MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/), [GShard / Switch Transformer](https://arxiv.org/abs/2001.04451).

---

### 1.6 4D-5D Parallelism

- **4D** composes **Data (D)**, **Tensor (T)**, **Pipeline (P)**, and **Context (C)** parallelism so every axis of the workload can be distributed.
  Picture the GPUs as a 4-D lattice: _N = DxTxPxC_ ranks.
- **5D** combines **4D** + **Expert Parallelism**.

**Key ideas**

- **Extreme scale** - Easily maps 10³-10⁴ GPUs for 100 B-parameter, 8 k-context models.
- **Topology aware** - Tune each dimension to match intra-node (NVLink), inter-node (IB), and rack-level bandwidth.
- **Memory & compute balance** - TP shards big matrices, CP splits long sequences, PP handles depth, DP feeds throughput.

**Mermaid Diagram**

```mermaid
graph TD
    %% Example 2x2x2x2 grid (16 GPUs)
    subgraph Stage0["Pipeline Stage 0"]
        subgraph TP0["Tensor Group 0"]
            R0000["GPU D0-C0"]
            R0001["GPU D0-C1"]
        end
        subgraph TP1["Tensor Group 1"]
            R0010["GPU D0-C0"]
            R0011["GPU D0-C1"]
        end
    end
    subgraph Stage1["Pipeline Stage 1"]
        subgraph TP0S1["Tensor Group 0"]
            R0100["GPU D1-C0"]
            R0101["GPU D1-C1"]
        end
        subgraph TP1S1["Tensor Group 1"]
            R0110["GPU D1-C0"]
            R0111["GPU D1-C1"]
        end
    end
    A["Micro-batches (DP)"] --> R0000
    R0000 -->|TP| R0010
    R0010 -->|PP| R0100
    R0100 -->|CP assemble| Z["Output"]
```

- **Use Case**: Training > 100 B-parameter models with multi-node clusters and long context windows.
- **Tools**: [Picotron](https://github.com/huggingface/picotron), [Nanotron](https://github.com/huggingface/nanotron).

## 2. Training Strategies

> **Rule of thumb** - pick the simplest scheme that fits in memory **and** saturates your interconnect.
> Start with a shard-aware data-parallel variant (FSDP/ZeRO-3).
> Add **Tensor ↔ Pipeline ↔ Context** axes only when the model or the sequence length forces you.

| Hardware scope                                  | Fastest link | Go-to recipe                                    | When to switch                      |
| ----------------------------------------------- | ------------ | ----------------------------------------------- | ----------------------------------- |
| **1 node** (2-8 GPUs, NVLink / PCIe Gen5)       | 200-900 GB/s | **FSDP + small TP** via `torchrun` or DeepSpeed | Model > 1 x GPU                     |
| **2-16 nodes** (≤128 GPUs, NVLink + InfiniBand) | 25-200 GB/s  | **"TP inside, DP across" + optional PP**        | Model > 1 x node                    |
| **>16 nodes** (hundreds-thousands GPUs)         | ≤25 GB/s     | **4-D grid (DPxTPxPPxCP)**                      | 70 B + params **and** 32 k + tokens |

### 2.1 Single-Node, Multi-GPU

Combine zeRO-style **Fully-Sharded Data Parallelism (FSDP)** with a low-degree **Tensor Parallelism** group that stays inside the node.

- FSDP shards parameters, gradients, and optimizer states across all GPUs, so each GPU uses only about 1/n of the total model memory — significantly reducing memory usage per GPU.
- TP protects matmul kernels from weight-gather latency; keep `tp<=2` on PCIe, up to `tp<=4` on NVLink.

### 2.2 Multi-Node, Multi-GPU

Start with **Tensor Parallelism inside a node** and **Data Parallelism across nodes**; introduce **Pipeline Parallelism** when the model no longer fits on one node.

- Keep TP collectives inside the node to avoid slow inter-node all-reduces.
- Tune **micro-batch = 4 x PP degree** as recommended by the Ultra-Scale Playbook to limit the pipeline bubble.

### 2.3 4D-5D Parallelism

When **weights** and **sequence length** both exceed a node, use every axis (DP x TP x PP x CP).

Guidelines:

- **TP** groups stay inside nodes; **PP/CP** may span nodes.
- Increase **DP** first when you need a larger global batch; it is the cheapest axis communication-wise.
- Expect ~75 % scaling efficiency up to 512 GPUs on InfiniBand clusters [(HF benchmarks, Feb 2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook).

---

## 3. Recommended Tools and Libraries

| Tool/Library            | Description                                                                         | Link                                                    |
| ----------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------- |
| DeepSpeed               | Optimizes training and inference for large models                                   | [DeepSpeed](https://github.com/microsoft/DeepSpeed)     |
| Megatron-LM             | Framework for training large transformer models with TP and PP                      | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)    |
| ColossalAI              | Provides a unified interface for various parallelism strategies                     | [ColossalAI](https://github.com/hpcaitech/ColossalAI)   |
| Horovod                 | Distributed training framework supporting multiple backends                         | [Horovod](https://github.com/horovod/horovod)           |
| Hugging Face Accelerate | Simplifies training and inference across devices                                    | [Accelerate](https://github.com/huggingface/accelerate) |
| TensorRT-LLM            | High-performance inference library by NVIDIA                                        | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)  |
| vLLM                    | Efficient LLM inference engine                                                      | [vLLM](https://github.com/vllm-project/vllm)            |
| Picotron                | Minimalistic 4D-parallelism distributed training framework for educational purposes | [Picotron](https://github.com/huggingface/picotron)     |
| Nanotron                | Minimalistic large language model 3D-parallelism training framework                 | [Nanotron](https://github.com/huggingface/nanotron)     |

## 4. Choosing the Right Strategy

| Scenario                                        | Recommended Approach                                      |
| ----------------------------------------------- | --------------------------------------------------------- |
| Training on a single machine with multiple GPUs | Combine DP with TP or PP using DeepSpeed or PyTorch FSDP. |
| Training across multiple machines               | Utilize DeepSpeed with a combination of DP, TP, and PP.   |
| Training with very long context windows         | Use Picotron or Nanotron with Context Parallelism.        |
| Training extremely large models                 | Leverage 4D parallelism with Picotron or Nanotron.        |
| Inference with latency constraints              | Deploy using TensorRT-LLM or vLLM.                        |
| Inference for very large models                 | Use DeepSpeed Inference with ZeRO-Offload.                |
| Quick deployment of models                      | Leverage Hugging Face TGI.                                |

### Cheatsheet from HuggingFace Folks

![Ultra-Scale LLM Cheatsheet](https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg)

⸻

By adopting these strategies and tools, you can effectively scale LLM training and inference across multiple GPUs and systems, ensuring optimal performance and resource utilization.
