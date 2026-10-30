# Energy Efficiency of LLM Inference

Benchmark framework for measuring the energy consumption of LLM inference and relating it to output quality. 
It sweeps over **models × quantization levels × tasks** and records different metrics.


**Models:** Qwen2.5-14B, Mistral-7B, Phi-3-mini, Qwen2.5-0.5B
**Quantization:** FP16, INT8 (W8A8), INT4 (GPTQ), using pre-quantized checkpoints
**Tasks:** Summarization XSum, Code HumanEval, Json JsonSchemaBench
**Metrics:** BERTScore F1, pass@1, schema validity

## Project Structure

```
main/              runner.py
project_types/     dataclasses
model_loader/      vLLM model loading
tasks/             task implementations
metrics/           quality metrics
energy.py          Energy_Monitor
```

## Requirements

- Linux with an NVIDIA GPU and CUDA drivers
- Python 3.12
- vllm, torch, datasets, bert-score, rouge-score, jsonschema, nvidia-ml-py

pip install -r requirements.txt

## Usage

```bash
python code/main/run.py.py --task summarization
python code/main/run.py.py --task code --models mistral_7b,phi3_mini --quants fp16,int4
python code/main/run.py.py --task json --max-samples 0 #all samples
```