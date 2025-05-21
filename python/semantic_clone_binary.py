#!/usr/bin/env python3
import argparse, os, sys, torch, torch.nn.functional as F
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)
from sentence_transformers import SentenceTransformer, util

EMBED_MODELS = {
    "mchochlov/codebert-base-cd-ft", # Paper: Using a Nearest-Neighbour, BERT-Based Approach for Scalable Clone Detection
    "jiekeshi/CodeBERT-50MB-Clone-Detection", # Paper: Compressing Pre-trained Models of Code into 3 MB
    "jiekeshi/GraphCodeBERT-3MB-Clone-Detection" # Paper: Compressing Pre-trained Models of Code into 3 MB
}
CLASSIFIER_MODELS = {
    "4luc/codebert-code-clone-detector",
    "jiekeshi/GraphCodeBERT-Adversarial-Finetuned-Clone-Detection"
}
BASE_TOKENIZER = {  # map → their pre-training base
    "4luc/codebert-code-clone-detector":      "microsoft/codebert-base", # Paper: CodeBERT for Code Clone Detection: A Replication Study
    "jiekeshi/GraphCodeBERT-Adversarial-Finetuned-Clone-Detection":
                                              "microsoft/graphcodebert-base" # Paper: Natural Attack for Pre-trained Models of Code
}

def load_classifier(name, device):
    base_tok = BASE_TOKENIZER.get(name, name)
    tok  = AutoTokenizer.from_pretrained(base_tok)
    mdl  = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    return tok, mdl

@torch.inference_mode()
def classifier_score(code1, code2, tok, mdl) -> float:
    toks   = tok(code1, code2, truncation=True, padding=True,
                 return_tensors="pt").to(mdl.device)
    logits = mdl(**toks).logits.squeeze(0)
    if logits.numel() == 1
        prob = torch.sigmoid(logits).item()
    else:
        prob = torch.softmax(logits, dim=-1)[1].item()
    return prob

@torch.inference_mode()
def embed_score(code1, code2, model):
    v1 = F.normalize(model.encode(code1, convert_to_tensor=True, truncation=True), p=2, dim=0)
    v2 = F.normalize(model.encode(code2, convert_to_tensor=True, truncation=True), p=2, dim=0)
    return float(util.cos_sim(v1, v2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file1", required=True)
    ap.add_argument("--file2", required=True)
    ap.add_argument("--model", default="4luc/codebert-code-clone-detector")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="probability (classifier) or cosine (embed) cut-off")
    args = ap.parse_args()

    for p in (args.file1, args.file2):
        if not os.path.isfile(p):
            sys.exit(f"❌  Cannot read {p}")

    with open(args.file1, encoding="utf-8") as f: c1 = f.read()
    with open(args.file2, encoding="utf-8") as f: c2 = f.read()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model in CLASSIFIER_MODELS:
        tok, mdl = load_classifier(args.model, device)
        score, kind = classifier_score(c1, c2, tok, mdl), "probability"
    elif args.model in EMBED_MODELS:
        mdl  = SentenceTransformer(args.model, device=device)
        score, kind = embed_score(c1, c2, mdl), "cosine"
    else:
        sys.exit("❌  Unknown model – add it to the list!")

    print(f"\n{kind.capitalize()}: {score:.4f}  ({score*100:.2f}%)")
    if score >= args.threshold:
        print(f"✔  Files ARE semantic clones (≥ {args.threshold}).")
    else:
        print(f"✘  Files are NOT semantic clones (< {args.threshold}).")

if __name__ == "__main__":
    main()
