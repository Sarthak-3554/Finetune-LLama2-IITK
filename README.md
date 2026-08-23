# Fine-Tuning Llama-2-7B on IIT Kanpur CSE Department Data (LoRA & QLoRA)

This notebook fine-tunes `NousResearch/Llama-2-7b-chat-hf` with QLoRA on a
custom instruction dataset built from the publicly scraped
[cse.iitk.ac.in](https://www.cse.iitk.ac.in) website, so the resulting model
can answer questions about the department — faculty, courses, programs,
admissions, research areas, and more.

Adapted from the original
[mlabonne "Fine-tune Llama 2" tutorial](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd)
(originally trained on `mlabonne/guanaco-llama2-1k`), retargeted to a
domain-specific dataset and patched to run on current library versions.

## What this notebook does

1. Loads `NousResearch/Llama-2-7b-chat-hf` in 4-bit precision (QLoRA)
2. Fine-tunes it with LoRA adapters on the IITK CSE instruction dataset
   (formatted as `<s>[INST] {question} [/INST] {answer} </s>`)
3. Runs a quick generation test with the fine-tuned model
4. Merges the LoRA weights into the base model and saves/optionally pushes
   the merged model to the Hugging Face Hub

## Environment

Built and tested on **Kaggle** with a single T4 GPU (16GB).

> **Note on multi-GPU:** if you have 2 GPUs available, this notebook
> deliberately restricts itself to 1 (see Step 0 below). Hugging Face's
> `Trainer` auto-wraps models across multiple visible GPUs using
> `torch.nn.DataParallel`, which is broken with bitsandbytes 4-bit
> quantization and causes a `CUBLAS_STATUS_EXECUTION_FAILED` crash. Real
> multi-GPU speedup would require proper distributed training (DDP via
> `accelerate launch`), which is a separate, larger change not included here.

### Required package versions

```
transformers==4.47.1
trl==0.15.2
peft
accelerate
bitsandbytes
```

**Why pinned, not latest:** `trl` went through several breaking API changes
after `~0.16` (`max_seq_length` renamed to `max_length`, `group_by_length`
and other `SFTConfig` fields dropped/reworked, `tokenizer=` renamed to
`processing_class=`). `trl==0.15.2` is the newest version that still accepts
this notebook's original argument names directly, while being new enough
that `tokenizers` has prebuilt wheels for current Python (avoiding a Rust
build failure that occurs with much older pins on newer Python versions).

`torchao` is explicitly uninstalled (Step 7) — Kaggle ships an old version
that fails `peft`'s minimum-version check during LoRA merging, and this
notebook doesn't use it (it uses bitsandbytes for quantization).

## Dataset

- **Format:** JSONL, one `{"text": "<s>[INST] question [/INST] answer </s>"}`
  object per line — the standard Llama-2 chat template.
- **Source:** built from scraped text on cse.iitk.ac.in (faculty pages,
  course descriptions, admissions/program pages, research areas, department
  info) — some generated via templated extraction, some hand-written by
  reading each page directly.
- **Note:** raw per-student roster data (names + roll numbers) was
  deliberately excluded/aggregated rather than turned into individually
  queryable facts, to avoid baking identifiable student records into a
  model that can be prompted to recite them.
- Update `dataset_name` in the parameters cell to point at your own copy of
  the dataset file (upload it to your Kaggle dataset/Colab session first).

## Step-by-step

| Step | What happens |
|---|---|
| 0 | Restrict to a single GPU (`CUDA_VISIBLE_DEVICES=0`) — must run before any CUDA/torch call |
| 1 | Install pinned packages; remove incompatible `torchao` |
| 2 | Import libraries |
| 3 | Notes on the Llama-2 chat prompt template |
| 4 | Set model name, dataset path, QLoRA/training hyperparameters |
| 5 | Load dataset, load base model in 4-bit, configure LoRA, train |
| 6 | View training loss in TensorBoard |
| 7 | Quick generation test with the fine-tuned model |
| 8 | Merge LoRA weights into the base model (reload in FP16) |
| 9 | (Optional, commented out) push the merged model to the Hugging Face Hub |

## Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `lora_dropout` | 0.1 | |
| `use_4bit` | `True`, NF4 | QLoRA quantization |
| `num_train_epochs` | 1 | |
| `per_device_train_batch_size` | 4 | Lower this if you hit CUDA OOM |
| `learning_rate` | 2e-4 | |
| `lr_scheduler_type` | cosine, 3% warmup | |
| `optim` | `paged_adamw_32bit` | |

## Known limitations (from actual runs of this notebook)

- **The model doesn't reliably stop generating.** `tokenizer.pad_token =
  tokenizer.eos_token` means the loss-masking can end up excluding the real
  EOS token from training, so the model keeps generating fake follow-up
  `[INST]` turns after answering. Quick workaround: pass
  `stop_strings=["[INST]"]` to the generation pipeline. Real fix: add a
  dedicated `<pad>` token (`tokenizer.add_special_tokens({"pad_token":
  "<pad>"})` + `model.resize_token_embeddings(len(tokenizer))`) and retrain.
- **Factual recall is inconsistent at this scale.** With 1 epoch and a
  ~6K-example dataset on a 7B model, some answers reflect genuine IITK CSE
  facts from the training data; others drift back toward Llama 2's generic
  pretrained knowledge about Indian engineering programs in general. Spot-
  check answers against the source data before trusting them.
- **CUDA OOM on reruns within the same kernel session** is usually leftover
  GPU memory from a previous cell (model/tokenizer/trainer/pipeline objects
  not released), not a new problem — restart the kernel and run top-to-
  bottom rather than re-running training on top of an already-loaded model.

## Output

The fine-tuned, merged model is saved locally under `new_model` (default:
`Llama-2-7b-chat-iitk-cse-finetune`) and can optionally be pushed to the
Hugging Face Hub via the commented-out cell in Step 8.
