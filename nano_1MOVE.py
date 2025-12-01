# nano_infer_evalpath.py — NanoGPT free-text continuation via eval path (CPU, read-only, verbose)
import os, pickle, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

from nanogpt.nanogpt_module import NanoGptPlayer

MODEL_NAME = "lichess_200k_bins_16layers_ckpt_with_optimizer.pt"  # must be in nanogpt/out/
META_PATH  = "nanogpt/out/meta.pkl"                                # same meta the eval run loads
SEED = ";1. e4 e5 2. Nf3 "                                         # keep trailing space

print("[info] creating NanoGptPlayer…")
player = NanoGptPlayer(model_name=MODEL_NAME)

# Get the underlying Karpathy-style model
model = getattr(player, "model", None) or getattr(player, "_model", None)
if model is None:
    raise RuntimeError("NanoGptPlayer has no .model / ._model attribute")
model.to("cpu").eval()
print("[info] model on CPU and in eval()")

# Load the exact charset the eval path uses
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
itos = [meta["itos"][i] for i in range(len(meta["itos"]))]
stoi = {c:i for i,c in enumerate(itos)}

print("[debug] itos (spaces as ␠):", ''.join('␠' if c == ' ' else c for c in itos))
print("[debug] len(itos):", len(itos))

# OOV check
oov = [c for c in SEED if c not in stoi]
print("[debug] seed:", repr(SEED))
print("[debug] OOV chars in seed:", oov)

# Encode
ids = torch.tensor([[stoi.get(ch, 0) for ch in SEED]], dtype=torch.long)
print("[debug] encoded ids:", ids.tolist(), "  shape:", tuple(ids.shape))

# Karpathy-style forward returns (logits, loss) when targets is provided
with torch.no_grad():
    logits, loss = model(ids, targets=ids)
print(f"[debug] seed loss (lower is better): {float(loss):.4f}")

# Generate (no 'do_sample' arg in NanoGPT): control randomness via temperature/top_k
print("[info] generating…")
out = model.generate(
    idx=ids,
    max_new_tokens=80,
    temperature=0.6,   # try 0.6 → cleaner
    top_k=32           # slightly tighter than 40
)

decoded = ''.join(itos[int(i)] for i in out[0])
print("\n=== COMPLETION ===")
print(decoded)

