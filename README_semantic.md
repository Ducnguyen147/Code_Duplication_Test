## Shebang line
Purpose: Directs Unix-like systems to use the python3 interpreter found in the user’s PATH rather than a hard-coded location. This ensures portability across environments where Python may be installed in different paths
Stack Overflow
and selects whichever python3 appears first in $PATH

## Imports and Model Constants
argparse: Builds the CLI and parses flags/arguments from the user
Python documentation.

os, sys: Handle file checks and process exit—standard Python modules.

torch: PyTorch core library for tensor operations and device management.

torch.nn.functional (F): Provides lower-level neural network operations, including normalize for vector normalization
PyTorch.

transformers.AutoTokenizer: Automatically loads the appropriate tokenizer for a given pre-trained model ID
Hugging Face.

transformers.AutoModelForSequenceClassification: Loads a model with a sequence classification head (linear layer on [CLS]) for tasks like binary or multiclass classification
Hugging Face .

sentence_transformers.SentenceTransformer: Loads embedding models optimized for producing fixed-size sentence or code embeddings
SentenceTransformers.

sentence_transformers.util.cos_sim: Computes the cosine similarity between two embeddings

Model lists: Distinguish between models that output embeddings for cosine similarity and models with classification heads.

Tokenizer mapping: Some classifier models share tokenizers with their base pre-training checkpoints.

## Loading a Classifier
Select tokenizer: Uses BASE_TOKENIZER mapping to load the correct tokenizer for a given classification checkpoint
Hugging Face.

Load model: AutoModelForSequenceClassification.from_pretrained fetches weights and architecture for sequence classification and moves it to device (CPU/GPU)
Hugging Face.

.eval(): Sets the model to evaluation mode (disables dropout, etc.).

## Classifier-Based Scoring
@torch.inference_mode(): Context manager that disables autograd and optimizes inference speed and memory by skipping gradient tracking
PyTorch
.

Tokenization: Encodes two code strings as a pair, applies truncation and padding, and returns PyTorch tensors on the model’s device.

Logits extraction: The model’s forward call yields .logits.

Single vs. multi-logit head:
    - If num_labels=1, the model returns a single logit for binary classification; apply sigmoid to get a probability
    PyTorch.

    - If num_labels=2, returns two logits; apply softmax and take the positive‐class index ([1]) as probability
    PyTorch.

## Embedding-based scoring
Embedding: model.encode(...) converts code text into a fixed-dimensional embedding tensor
SentenceTransformers.

Normalization: F.normalize(..., p=2) scales embeddings to unit length under L₂ norm, a common step before cosine similarity
PyTorch.

Cosine similarity: util.cos_sim computes the dot product between normalized vectors, yielding a similarity score in [-1,1] 

## Main Execution Flow
Argument parsing: Defines required file paths, model choice, and similarity threshold.

File validation: Exits if either file is unreadable.

File reading: Loads the source code text.

Device selection: Uses GPU if CUDA is available, otherwise CPU.

Model dispatch: Chooses classifier or embed path based on the selected model ID.

Scoring: Runs the appropriate scoring function and labels the result ("probability" or "cosine").

Threshold comparison: Prints a human-readable verdict, marking files as clones if the score meets or exceeds the threshold.