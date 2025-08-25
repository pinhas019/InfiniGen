import math, json, torch
from transformers import AutoTokenizer
from transformers.models.opt.configuration_opt import OPTConfig

# ✅ Only import from the patched file inside transformers
from transformers.models.opt.modeling_opt_ours import OPTForCausalLM as OPTForCausalLM_ARC

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_id = "facebook/opt-125m"

# 1) Load tokenizer + config
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
config = OPTConfig.from_pretrained(model_id)

# 2) Load weights into your patched class
model = OPTForCausalLM_ARC.from_pretrained(
    model_id,
    config=config,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
model.eval()

# 3) Force ARC eviction policy (just in case)
for layer in model.model.decoder.layers:
    if hasattr(layer.self_attn, "eviction_policy"):
        layer.self_attn.eviction_policy = "arc"
        setattr(layer.self_attn, "arc_beta", 0.5)

print("Layer 0 eviction_policy:",
      getattr(model.model.decoder.layers[0].self_attn, "eviction_policy", "N/A"))

# 4) Quick generation check
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=20)
print("Generated:", tok.decode(out[0], skip_special_tokens=True))

# 5) Optional perplexity demo
path = r"C:\Users\morsh\Desktop\InfiniGen\lm_eval\results\openbookqa-5.jsonl"
try:
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(l)["question_stem"] for l in f]
    enc = tok(lines, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss.item()
    print("Perplexity:", round(math.exp(loss), 3))
except FileNotFoundError:
    print("Skipping perplexity (demo file not found).")
