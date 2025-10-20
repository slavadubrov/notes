---
title: Orchestrating and serving LLM deployments with LoRAX
summary: Stand up multi-LoRA serving in Docker, Kubernetes, and SkyPilot while understanding operational patterns, trade-offs, and advanced tuning knobs.
---

LoRAX (LoRA eXchange) promises that one GPU can host hundreds of task-specialized adapters without the usual deployment sprawl. This guide walks through the why and how, from single-node Docker runs to managed Kubernetes and SkyPilot clusters, while highlighting production lessons learned.

<!-- more -->

## Why LoRAX matters

When your organization standardizes on a single base LLM but explodes into dozens or thousands of fine-tuned LoRA adapters, conventional inference stacks fall over. LoRAX keeps the base weights resident in GPU memory, hot-loads adapters per request, and continuously batches heterogeneous workloads so every token of compute stays productive. You also get OpenAI-compatible APIs, Prometheus metrics, and JSON-constrained decoding out of the box, which shortens the path from experiment to production.

**TL;DR**

- Serve many LoRA adapters from one GPU by dynamically loading weights on demand.
- Cache adapters across GPU, CPU, and disk tiers for balanced cost and latency.
- Reuse existing OpenAI client code paths via compatible REST and Python interfaces.
- Deploy with prebuilt Docker images, Helm charts, and SkyPilot recipes.

## Core concepts behind LoRAX

LoRAX builds on the multi-adapter batching ideas pioneered by Punica and S-LoRA and ships as an open-source inference server forked from Hugging Face Text Generation Inference (TGI v0.9.4). Key design elements include:

- **Dynamic adapter loading** — The base LLM loads once, while adapters are mounted just in time per request. Cold loads incur a few hundred milliseconds, then requests are served at full speed.
- **Tiered weight caching** — LoRAX promotes hot adapters to GPU memory, demotes warm adapters to CPU RAM, and spills cold adapters to disk, balancing performance with capacity.
- **Continuous multi-adapter batching** — Requests targeting different adapters are masked and scheduled together, keeping GPU utilization high even under heterogeneous traffic.
- **Operational tooling** — The project ships a Docker image, Helm chart, SkyPilot recipe, and OpenAI-compatible APIs plus a Python client, making integration and scaling straightforward.

## Hardware and runtime prerequisites

To run LoRAX you need:

- An NVIDIA GPU based on Ampere or newer architectures.
- CUDA 11.8 or later with the NVIDIA Container Toolkit installed.
- A Linux host capable of running Docker.

## Quickstart with Docker

Start by pulling the public container image and launching a server that hosts a base instruct model:

```bash
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
VOLUME="$PWD/data"

docker run --gpus all --shm-size 1g -p 8080:80 -v "$VOLUME:/data" \
  ghcr.io/predibase/lorax:main \
  --model-id "$MODEL_ID"
```

> Tip: swap `:main` for `:latest` to pin a stable release tag once validated. The container listens on port 80 internally, so expose it as desired.

### Issue requests with REST

Generate text against the base weights:

```bash
curl -s http://127.0.0.1:8080/generate -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Summarize the role of LoRA adapters in 2 bullet points",
    "parameters": {"max_new_tokens": 80}
  }'
```

Attach an adapter identifier to hot-load and reuse LoRA weights on demand:

```bash
curl -s http://127.0.0.1:8080/generate -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "Solve: If Natalia sold 48 clips in April and half as many in May, how many in total?",
    "parameters": {
      "max_new_tokens": 64,
      "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"
    }
  }'
```

### Use the Python client

```python
from lorax import Client

client = Client("http://127.0.0.1:8080")

print(client.generate("one sentence on LoRA", max_new_tokens=32).generated_text)

print(client.generate(
    "classify: urgent vs not urgent: 'Please reset my password'",
    max_new_tokens=32,
    adapter_id="predibase/customer_support"
).generated_text)
```

### OpenAI-compatible chat interface

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8080/v1")

resp = client.chat.completions.create(
    model="alignment-handbook/zephyr-7b-dpo-lora",
    messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Write a one-liner about LoRAX."},
    ],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

## Orchestrating at scale

### Option A: Kubernetes with Helm

LoRAX ships a Helm chart that deploys a load-balanced, GPU-backed replica set. After cloning the chart or vendor repository, install it with:

```bash
helm install mistral-7b-release charts/lorax
```

Override settings through a values file when you need specific resource limits or feature flags:

```bash
helm install -f values.yaml lorax charts/lorax
```

A minimal configuration might look like this:

```yaml
image:
  repository: ghcr.io/predibase/lorax
  tag: latest
server:
  args:
    - --model-id
    - mistralai/Mistral-7B-Instruct-v0.1
    - --max-input-length
    - "4096"
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    cpu: "2"
    memory: "8Gi"
service:
  type: ClusterIP
metrics:
  enabled: true
```

Tune flags such as quantization, CUDA graphs, or maximum token limits via `server.args` or environment variables as your workloads evolve.

### Option B: SkyPilot for cross-cloud clusters

When you need managed provisioning across AWS, Azure, or GCP, SkyPilot combines resource selection with reproducible runtime configuration. Define a deployment file:

```yaml
# lorax.yaml
resources:
  cloud: aws
  accelerators: A10G:1
  memory: 32+
  ports: [8080]

envs:
  MODEL_ID: mistralai/Mistral-7B-Instruct-v0.1

run: |
  docker run --gpus all --shm-size 1g -p 8080:80 -v ~/data:/data \
    ghcr.io/predibase/lorax:main \
    --model-id $MODEL_ID
```

Launch the cluster and capture the public endpoint:

```bash
pip install skypilot && sky check
sky launch -c lorax-cluster lorax.yaml
IP=$(sky status --ip lorax-cluster)
curl "http://$IP:8080/generate" -H 'Content-Type: application/json' \
  -d '{"inputs":"hi","parameters":{"adapter_id":"predibase/customer_support"}}'
```

## Advanced knobs you will reach for

1. **Quantize the base model** — Load the backbone weights in 4-bit or 8-bit formats (bitsandbytes, GPTQ, AWQ) to fit larger batches or sequence lengths. Validate quality before shipping to production.
2. **Structured JSON output** — LoRAX performs token-level constrained decoding so you can enforce JSON schemas without brittle regex post-processing.
3. **Adapter merging** — Combine multiple adapters—such as domain specialization plus brand tone—at request time using weighted strategies.
4. **Observability** — Scrape `/metrics` with Prometheus to track queue depths, token throughput, and latency distributions for auto-scaling and alerting.

### Cookbook snippets

**Schema-constrained generation**

```python
from lorax import Client
client = Client("http://127.0.0.1:8080")

schema = {
  "type": "object",
  "properties": {
    "priority": {"type":"string","enum":["low","medium","high"]},
    "category": {"type":"string"},
    "summary":  {"type":"string"}
  },
  "required":["priority","category","summary"]
}

resp = client.generate(
    "Classify and summarize: 'Order not delivered after 5 days'",
    max_new_tokens=128,
    adapter_id="predibase/customer_support",
    json_schema=schema
)
print(resp.generated_text)
```

**Merge adapters for composite behavior**

```python
from lorax import Client, MergedAdapters

client = Client("http://127.0.0.1:8080")

merged = MergedAdapters(
    adapters=[
        {"adapter_id": "my-org/domain-sql"},
        {"adapter_id": "my-org/tone-formal", "alpha": 0.5},
    ],
    strategy="linear"
)

out = client.generate(
    "Write a safe SELECT for yesterday's revenue by channel (Postgres).",
    max_new_tokens=128,
    merged_adapters=merged
)
print(out.generated_text)
```

**Prometheus scrape configuration**

```yaml
endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

## Production patterns that work

- **Multi-tenant routing** — Map `customer_id` to `adapter_id` in an API gateway so each tenant enjoys isolated behavior with shared infrastructure. Use LoRAX's OpenAI-compatible endpoints to reuse existing auth and rate-limiting flows.
- **RAG plus multi-LoRA** — Use retrieval augmented generation for grounding while a LoRA enforces policy or tone. Speculative decoding techniques like Medusa or prefix lookup decoders can coexist if the adapter set remains compatible.
- **LoRAX vs. vLLM mental model** — vLLM shines for monolithic models with optimized KV-cache management. LoRAX leans into multi-adapter efficiency. If you need “N models for the price of one GPU,” LoRAX is purpose-built for it.

## Performance and cost tuning levers

- Quantize the base to free memory for longer contexts or higher concurrency.
- Enable CUDA graph compilation when your GPU memory budget allows.
- Tune `MAX_BATCH_TOTAL_TOKENS`, maximum input lengths, and scheduling windows to avoid out-of-memory errors while preserving throughput.
- Pin hot adapters in the GPU cache and observe eviction behavior under synthetic load before going live.
- Balance throughput versus latency targets by adjusting batching and fairness controls.

## Pros and cons

| Aspect   | 👍 Strengths | 👎 Trade-offs |
|----------|--------------|---------------|
| Cost | Serve hundreds or thousands of adapters from one GPU instead of maintaining N full replicas. | Uniformly hot adapter sets still saturate a single GPU; horizontal scaling remains necessary. |
| Latency | Adapter loading overhead is typically a few hundred milliseconds and amortizes across generations and cache hits. | Cold adapters pay the load penalty, and adversarial traffic can thrash caches, elevating tail latency. |
| Throughput | Continuous multi-adapter batching keeps utilization high even with heterogeneous prompts. | Requires careful tuning of batch and token limits; long prompts can starve smaller jobs without fair scheduling. |
| Developer experience | OpenAI-compatible APIs, Python client, Docker, Helm, and SkyPilot artifacts lower integration friction. | Chart values and CUDA dependencies evolve, so you must track image tags and driver compatibility. |
| Features | Supports quantization, JSON constraints, adapter merging, and Prometheus metrics. | Advanced features interact in complex ways—test speculative decoding, adapter merges, and quantization combinations. |
| Ecosystem | Rooted in TGI and inspired by Punica/S-LoRA, bringing research ideas to practitioners. | vLLM's community is larger; if you only run a single fine-tuned model, LoRAX may be overkill. |

## Operational checklist

- Monitor queue time, tokens per second, and latency percentiles; alert when they drift.
- Manage adapter lifecycles by versioning, A/B testing, and purging stale adapters from the `/data` cache volume.
- Gate tenants through a proxy that injects `adapter_id` per session to maintain isolation.
- Size token limits and quantization strategies based on the active set of hot adapters and expected concurrency.

## When LoRAX is not the right fit

- You only operate a single fine-tuned model and do not anticipate adapter proliferation.
- Your adapters are not LoRA-based or rely on unsupported fine-tuning techniques.
- Ultra-low-latency requirements collide with entirely cold adapter traffic patterns; cache misses dominate response time.

## Closing thoughts

The “one base, many adapters” pattern is rapidly becoming the default for production LLM systems. LoRAX lets you centralize that universe into a single service with predictable ergonomics and economics. If you are juggling multiple fine-tuned variants today, the path to LoRAX is largely about adapter plumbing and routing strategy—far easier than standing up a fleet of bespoke inference stacks.

## References and further reading

- LoRAX documentation covering architecture, REST and Python clients, JSON output, quantization, merging, and metrics.
- Predibase engineering blogs on dynamic adapter loading, tiered caching, and multi-adapter batching internals.
- SkyPilot and Helm quickstarts for cloud and Kubernetes deployment.
- Punica and S-LoRA research papers on efficient multi-adapter serving.
- Community discussions comparing LoRAX and vLLM economics on marketplaces such as Vast.ai.

## Appendix: values.yaml switches to tweak first

- `--quantize {bitsandbytes|gptq|awq}` to trade VRAM for quality.
- `--max-input-length` and `--max-batch-total-tokens` to balance utilization with OOM safety.
- Health and readiness thresholds to enable fast Kubernetes rescheduling.
- Node pool labels and tolerations to guarantee the intended GPU class.
