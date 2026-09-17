#!/bin/bash
# Fix môi trường TRAIN trên container driver CUDA 12.4: bỏ unsloth (bleeding-edge lệch),
# pin stack HuggingFace coherent. torch 2.6.0+cu124 đã cài trước (fix_torch.sh).
set -e
echo START_FIXENV
pip uninstall -y unsloth unsloth_zoo trl 2>/dev/null || true
pip install --no-cache-dir "transformers==4.49.0" "peft==0.14.0" "accelerate==1.4.0" "datasets==3.3.2"
python -c "import torch,transformers,peft,datasets,accelerate; print('torch',torch.__version__,'avail',torch.cuda.is_available(),'| tf',transformers.__version__,'peft',peft.__version__,'ds',datasets.__version__)"
python -c "from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,DataCollatorForSeq2Seq; from peft import LoraConfig,get_peft_model; print('IMPORTS_OK')"
echo FIXENV_DONE
