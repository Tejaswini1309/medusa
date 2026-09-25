# Medusa

## 1. Project Overview

### Objective
Implement MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.

## 2. Project Directory Structure

medusa/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── base_config.yaml          # Standard hyperparameters (learningrate, batch size, target model)
│   └── vicuna_7b_medusa.yaml     # Specific config for a target model + head architecture
├── medusa/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── medusa_head.py        # Definition of single & multiple Medusa head modules (ResBlock/MLP layers)
│   │   ├── medusa_model.py       # Wrapper combining base LLM backbone + Medusa heads
│   │   └── utils.py              # Weight loading, quantization, and model initialization helpers
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── tree.py               # Medusa tree structure definitions & candidate sequence generator
│   │   ├── kv_cache.py           # Modified KV-cache handling for tree-structured attention
│   │   └── decode.py             # Core speculative decoding engine (tree verification & acceptance)
│   ├── train/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Preprocessing input sequences for head training (Self-distillation / Callbacks)
│   │   ├── loss.py               # Cross-entropy loss across multiple heads (weighted sum)
│   │   └── trainer.py            # Training loop for Stage 1 (Heads only) or Stage 2 (Self-distillation)
│   └── utils/
│       ├── __init__.py
│       ├── sampling.py           # Top-k, top-p, temperature sampling for candidate nodes
│       └── metrics.py            # Speedup benchmarking, acceptance rate tracking
├── scripts/
│   ├── train_heads.py            # Entry point to train Medusa heads on top of a frozen backbone
│   ├── evaluate.py               # Benchmark generation latency & measure acceleration factor
│   └── run_inference.py          # Interactive CLI generation script
└── tests/
    ├── test_medusa_head.py       # Unit tests for head shapes & forward passes
    ├── test_tree_decoding.py     # Functional tests for candidate verification logic
    └── test_kv_cache.py          # Verification of tree attention mask application

## 3. System Components

The Medusa system is composed of the following major components:

### 3.1 Base LLM Backbone
File: medusa/model/medusa_model.py
- The original pretrained LLM acts as the target model.
- Its weights remain frozen during Stage 1 Medusa-head training.
- Its weights remian unfreezed during the stage 2 Medusa-head and Backbone LLm training.
- It produces the hidden representations and the standard next-token logits.
- The backbone is shared by all Medusa heads.

### 3.2 Medusa Heads
File: medusa/model/medusa_head.py
- Multiple lightweight prediction heads are attached to the final hidden representation of the backbone.
- Each head predicts a token at a different future position.
- Heads consist of lightweight neural network layers followed by a vocabulary projection.
- During Stage 1, only the Medusa heads are trained while the backbone remains frozen.
- During Stage 2, the system may jointly fine-tune the model using self-distillation.

### 3.3 Medusa Model Wrapper
File: medusa/model/utils.py
- Combines the pretrained backbone and all Medusa heads into a single model.
- Provides a unified interface for training and inference.
- Manages the forward pass through the backbone and the parallel prediction heads.
- Provides access to both the original model logits and Medusa-head predictions.
- The forward pass calls the full backbone (`output_hidden_states=True`) and reads the last
  hidden-state layer and the LM head's own logits from that single call, rather than bypassing
  the LM head via a lower-level "inner transformer" accessor. This keeps the wrapper correct
  whether the backbone is a plain pretrained model or LoRA-wrapped for Stage 2 self-distillation
  (§4.13), since a LoRA-wrapped model's lower-level accessor does not expose the same bypass.

### 3.4 Candidate Tree Generator
File: medusa/generation/tree.py
- Converts the predictions from multiple Medusa heads into candidate token sequences.
- Organizes candidates into a tree structure representing different possible future continuations.
- Uses sampling strategies such as top-k, top-p, and temperature where required.
- Maintains the relationship between parent and child candidate tokens.

### 3.5 Tree Attention / Attention Mask
File: medusa/generation/tree.py and medusa/generation/kv_cache.py
- Constructs an attention structure that allows the target model to verify multiple candidate sequences in a single forward pass.
- Generates an attention mask according to the parent-child relationships in the candidate tree.
- Prevents tokens from attending to unrelated branches of the tree.
- Prevent tokens from attending the tokens in the same brach that appears after it(like causal attention).
- Enables parallel verification of multiple candidate sequences.

### 3.6 KV Cache Manager
File: medusa/generation/kv_cache.py
- Maintains key-value states required during autoregressive generation.
- Extends conventional KV-cache handling to support tree-structured candidate sequences.
- Reuses previously computed states to reduce redundant computation.
- Manages cache positions corresponding to different nodes in the candidate tree.

### 3.7 Speculative / Tree Decoding Engine
File: medusa/generation/decode.py
- Coordinates the complete Medusa inference process.
- Obtains predictions from the Medusa heads.
- Builds the candidate tree.
- Performs a target-model forward pass to evaluate candidate tokens.
- Applies the candidate acceptance procedure based on the target model's prediction probabilities.
- Accepts valid candidate tokens and rejects candidates that do not satisfy the acceptance criterion.
- Updates the generated sequence and KV cache before the next decoding iteration.

### 3.8 Sampling Module
File: medusa/utils/sampling.py
- Provides token-selection mechanisms used during candidate generation.
- Supports top-k sampling, top-p sampling, and temperature scaling.
- Converts model logits into candidate tokens and their probabilities.
- Ensures that candidate generation can be configured independently of the core decoding engine.

### 3.9 Training Pipeline
File: medusa/train/dataset.py, loss.py, trainer.py
- Prepares training sequences and target tokens for Medusa heads.
- Computes the prediction loss for each head.
- Combines losses from multiple heads using configurable weighting.
- Supports Stage 1 head-only training and Stage 2 backbone LLM + medusa head combined training.
- Maintains the appropriate frozen/trainable parameter configuration for each stage.

### 3.10 Configuration and Model Initialization
File: configs/*.yaml + medusa/model/utils.py
- Stores model, training, generation, and decoding hyperparameters in YAML configuration files.
- Loads the target model and initializes Medusa heads according to the selected configuration.
- Handles model weights, device placement, and optional quantization.
- Provides a consistent initialization process for training and inference.

### 3.11 Evaluation and Metrics
File: medusa/utils/metrics.py + scripts/evaluate.py
- Measures baseline and Medusa generation performance.
- Tracks decoding latency, tokens generated per second, and acceleration/speedup.
- Measures Medusa-head acceptance rates.
- Compares Medusa generation against standard autoregressive decoding.

### 3.12 CLI and Experiment Scripts
File: scripts/*.py
- `train_heads.py` starts the Medusa-head training pipeline.
- `run_inference.py` provides interactive generation using Medusa decoding.
- `evaluate.py` runs performance benchmarks and collects evaluation metrics.

### 3.13 Component Interaction Flow

The overall system follows this flow:

Input Prompt
      ↓
Base LLM Backbone
      ↓
Hidden Representation
      ├──────────────→ Original LM Head
      │                      ↓
      │                Base Prediction
      │
      └──────────────→ Multiple Medusa Heads
                             ↓
                    Future Token Predictions
                             ↓
                    Candidate Tree Generator
                             ↓
                     Candidate Token Tree
                             ↓
                  Tree Attention / KV Cache
                             ↓
                  Target Model Verification
                             ↓
                    Acceptance Procedure
                             ↓
                 Accepted Token Sequence
                             ↓
                     Updated KV Cache
                             ↓
                       Next Iteration

# 4. Core Algorithms

This section specifies the mathematical and algorithmic behavior of the
Medusa system, including Medusa-head prediction, candidate generation,
candidate tree construction, tree attention, target-model verification,
candidate acceptance, Medusa-1 training, Medusa-2 joint training, and
self-distillation.

---

## 4.1 Medusa Head Prediction

Each Medusa head receives the hidden representation produced by the
base language model and predicts a probability distribution over the
vocabulary for a future token position.

Let:

- h_t ∈ R^d be the output of the LLM's final hidden layer at position t.
- d be the hidden dimension of the base LLM.
- V be the vocabulary size.
- k denote the Medusa head index.

The k-th Medusa head computes:

p_t^(k) =
softmax(
    W_2^(k) ·
    (SiLU(W_1^(k) · h_t) + h_t)
)

where:

- W_1^(k) ∈ R^(d×d) is the first projection matrix.
- W_2^(k) ∈ R^(d×V) according to the paper's notation.
- SiLU is the activation function.
- h_t is added through a residual connection.
- p_t^(k) is the probability distribution over the vocabulary.

The computation consists of:

h_t
  ↓
Linear W_1^(k)
  ↓
SiLU
  ↓
Residual addition with h_t
  ↓
Linear W_2^(k)
  ↓
Softmax
  ↓
p_t^(k)

Each Medusa head has independent trainable parameters.

### 4.1.1 Medusa Head Initialization

The Medusa heads are initialized so that their initial predictions are
aligned with the original language model.

For each Medusa head k:

- W_2^(k) is initialized identically to the original language model
  head.
- W_1^(k) is initialized to zero.

Therefore:

SiLU(W_1^(k) h_t) = 0

at initialization, giving:

p_t^(k) ≈ softmax(W_2^(k) h_t)

This aligns the initial Medusa-head prediction with the original
language-model prediction.

---

## 4.2 Multi-Head Candidate Generation

During inference, the original language-model head and Medusa heads
generate candidate tokens.

The original language-model head predicts the next token, while the
Medusa heads predict subsequent future tokens.

For each head k, select the top-s_k predictions:

TopK(p_t^(k), s_k)

where s_k is the number of candidate tokens selected from head k.

The values of s_k may differ across heads.

For example:

s_1 = 2
s_2 = 3

means:

- Head 1 contributes 2 candidate tokens.
- Head 2 contributes 3 candidate tokens.

The candidate sequences are formed from the Cartesian product of the
selected predictions.

For K heads:

N_candidates = ∏_(k=1)^K s_k

For example:

s_1 = 2, s_2 = 3

gives:

N_candidates = 2 × 3 = 6

candidate sequences.

Each candidate sequence corresponds to a branch in the candidate tree.

---

## 4.3 Candidate Tree Construction

The candidate sequences are represented using a tree structure.

The root represents the current generated context.

The first prediction stage produces child nodes of the root. Predictions
from subsequent Medusa heads extend these nodes.

For example:

                    Root
                   /    \
                  A      B
                / | \  / | \
               C  D E C  D  E

Each root-to-leaf path represents a candidate sequence.

The candidate tree must maintain:

- token identity,
- parent node,
- child nodes,
- tree depth,
- logical sequence position,
- candidate branch,
- Medusa-head source,
- mapping between tree nodes and flattened model input positions.

The tree representation is used to construct the tree attention mask
and to perform target-model verification.

---

## 4.4 Tree Attention

The candidate tree contains multiple possible future sequences.
Therefore, candidates must be verified concurrently while preserving
the autoregressive dependency structure of each branch.

A tree attention mask is constructed to achieve this.

### 4.4.1 Tree Attention Principle

Each candidate token may attend to:

1. The original context tokens.
2. Its ancestors in the same candidate branch.

A candidate token must not attend to tokens belonging to unrelated
branches.

For example:

Root
 ├── A
 │    ├── C
 │    ├── D
 │    └── E
 │
 └── B
      ├── C
      ├── D
      └── E

The candidate C under A can attend to:

Root → A → C

but cannot attend to:

Root → B
or
Root → B → C

This prevents information from one candidate branch from leaking into
another branch.

### 4.4.2 Tree Attention Mask Construction

For every candidate node:

1. Identify its parent.
2. Traverse the parent chain back to the root.
3. Mark all ancestors as visible.
4. Mark the original context tokens as visible.
5. Mask unrelated candidate nodes.

The resulting mask is a sparse tree-structured causal attention mask.

The implementation must construct this mask from the actual parent-child
relationships of the candidate tree.

### 4.4.3 Positional Encoding

Candidate tokens exist on different branches but may represent the same
future generation position.

Therefore, positional indices must be assigned consistently with the
candidate tree structure.

The implementation must maintain a mapping:

tree node → logical sequence position

so that positional information remains consistent during verification.

---

## 4.5 Tree-Based Target Model Verification

After candidate construction, all candidate branches are evaluated by
the original/target language model in a single verification pass.

The target model receives:

- the original context,
- candidate tokens,
- the tree attention mask,
- corresponding positional information.

The target model produces probability distributions for the candidate
positions.

For candidate token x_(n+k), the relevant probability is:

p_original(
    x_(n+k) |
    x_1, x_2, ..., x_(n+k-1)
)

This target-model probability is used by the acceptance procedure.

The target model is the authoritative model for candidate verification.

---

## 4.6 Candidate Acceptance Procedure

The objective of the acceptance procedure is to select candidate tokens
that are sufficiently probable under the original language model.

Given the context:

x_1, x_2, ..., x_n

and candidate sequence:

(x_(n+1), x_(n+2), ..., x_(n+K+1)),

candidate token x_(n+k) is accepted when:

p_original(
    x_(n+k) |
    x_1, x_2, ..., x_(n+k-1)
)
>
min(
    ε,
    δ exp(
        -H(
            p_original(
                · |
                x_1, x_2, ..., x_(n+k-1)
            )
        )
    )
)

where:

- p_original(· | context) is the target-model probability distribution.
- p_original(x_(n+k) | context) is the probability assigned to the
  candidate token.
- H(·) is the entropy of the target-model distribution.
- ε is the hard acceptance threshold.
- δ controls the entropy-dependent threshold.

The effective threshold is:

threshold =
min(
    ε,
    δ exp(-H(p_original))
)

The candidate is accepted when:

candidate_probability > threshold

---

## 4.7 Entropy-Dependent Acceptance

The entropy of the target-model distribution is:

H(p) = -Σ_x p(x) log p(x)

Entropy measures the uncertainty of the target model.

The acceptance threshold therefore adapts to the uncertainty of the
model.

The two threshold components are:

1. Hard threshold:

   ε

2. Entropy-dependent threshold:

   δ exp(-H(p))

The final threshold is:

threshold =
min(
    ε,
    δ exp(-H(p))
)

This allows the acceptance criterion to account for the shape of the
target-model probability distribution instead of relying only on a
fixed probability threshold.

---

## 4.8 Candidate Acceptance and Tree Traversal

Acceptance is performed over the verified candidate tree.

For each candidate node:

1. Obtain the candidate token.
2. Obtain its probability from the target model.
3. Obtain the target-model probability distribution for the corresponding
   context.
4. Compute the entropy.
5. Compute the acceptance threshold.
6. Compare candidate probability with the threshold.
7. Accept the candidate if the criterion is satisfied.
8. Continue along the corresponding candidate branch.
9. Stop or reject the branch when the criterion fails.

The accepted tokens are appended to the generated sequence.

The accepted path determines the context for the next decoding
iteration.

---

# 4.9 Medusa-1 Training: Frozen Backbone

Medusa-1 trains the Medusa heads while keeping the base language model
backbone frozen.

Given the ground-truth token:

y_(t+k+1)

at position t+k+1, the loss for the k-th Medusa head is:

L_k =
-log p_t^(k)(y_(t+k+1))

where:

p_t^(k)(y)

denotes the probability assigned to token y by the k-th Medusa head.

Predictions become more uncertain for larger future offsets. Therefore,
the individual head losses can be weighted using λ_k.

The total Medusa-1 loss is:

L_Medusa-1 =
Σ_(k=1)^K
    -λ_k log p_t^(k)(y_(t+k+1))

where:

- K is the number of Medusa heads.
- λ_k is the loss weight for head k.
- The base LLM parameters are frozen.
- Only Medusa-head parameters are updated.

### 4.9.1 Medusa-1 Training Flow

Input sequence
      ↓
Frozen base LLM
      ↓
Hidden representation h_t
      ↓
Medusa heads
      ↓
p_t^(1), p_t^(2), ..., p_t^(K)
      ↓
Individual cross-entropy losses
      ↓
Weighted sum
      ↓
L_Medusa-1
      ↓
Backpropagation
      ↓
Update Medusa heads only

---

# 4.10 Medusa-2 Joint Training

Medusa-2 allows the Medusa heads and backbone model to be trained
jointly.

Unlike Medusa-1, backbone parameters are allowed to change.

However, simply training the backbone and Medusa heads on the
ground-truth dataset can degrade the original model's generation
quality.

Therefore, Medusa-2 uses an additional self-distillation procedure.

The self-distillation procedure allows the model's own original output
distribution to act as the training target for the backbone.

---

# 4.11 Self-Distillation Dataset Generation

The self-distillation pipeline is used when the original training
dataset is unavailable or when the available dataset does not match
the target model's output distribution.

This can occur when:

- the original training data is not publicly available,
- only the trained model is available,
- the model has undergone RLHF or other post-training procedures,
- the model's output distribution differs from the original training
  dataset.

Instead of relying only on ground-truth labels, the model itself is used
to generate a dataset matching its output distribution.

### 4.11.1 Seed Dataset

A public seed dataset from a domain similar to the target model is used.

For example:

- conversational models → conversational prompt datasets,
- domain-specific models → prompts from the corresponding domain.

The seed dataset provides prompts rather than requiring the original
training dataset.

### 4.11.2 Synthetic Dataset Generation

The prompts from the seed dataset are fed to the original model.

The model generates responses to these prompts.

For multi-turn conversations, prompts can be sequentially provided to
the model so that the model generates multiple rounds of conversation.

The resulting synthetic conversations form the self-distillation
training dataset.

The generated dataset therefore follows the output distribution of the
model itself.

### 4.11.3 Dataset Generation Flow

Public seed dataset
        ↓
Seed prompts
        ↓
Original model
        ↓
Generated responses
        ↓
Multi-turn conversation generation
        ↓
Synthetic self-distillation dataset
        ↓
Medusa training

The dataset generation process should be implemented separately from
the main optimization loop so that generated training data can be
stored and reused.

---

# 4.12 Self-Distillation Objective

For Medusa-2, using only ground-truth tokens to train the backbone can
lead to degradation of the model's original generation behavior.

Therefore, the original model's probability distribution is used as a
teacher distribution.

Let:

p_original,t^(0)

denote the probability distribution produced by the original model at
position t.

Let:

p_t^(0)

denote the probability distribution produced by the trainable backbone
during Medusa-2 training.

The backbone self-distillation loss is:

L_LM-distill =
KL(
    p_original,t^(0)
    ||
    p_t^(0)
)

where KL denotes the Kullback-Leibler divergence.

The objective is for the trainable model to reproduce the output
distribution of the original model.

The teacher distribution therefore provides a soft target rather than
only a single ground-truth token.

---

# 4.13 Parameter-Efficient Self-Distillation with LoRA

A naive implementation of self-distillation would require maintaining
two complete copies of the model:

1. The original model acting as the teacher.
2. The trainable model acting as the student.

This increases memory requirements significantly.

To avoid maintaining two complete model copies, the system uses a
parameter-efficient adapter such as LoRA.

The base model parameters are shared.

Two states are used:

### Teacher / Original Model

The LoRA adapter is disabled.

The resulting model represents the original model:

Base model + LoRA OFF
        ↓
Original model distribution

### Student / Trainable Model

The LoRA adapter is enabled.

The resulting model is trained during Medusa-2:

Base model + LoRA ON
        ↓
Trainable model distribution

Therefore, the same base model weights can be reused for both teacher
and student computations.

Conceptually:

                 Shared Base Model
                  /            \
             LoRA OFF        LoRA ON
                ↓               ↓
          Teacher Model    Student Model
                ↓               ↓
          p_original^(0)       p_t^(0)
                \               /
                 \             /
                  KL divergence
                       ↓
                Distillation loss

This avoids storing two complete copies of the backbone model.

### Implementation mechanism

`MedusaModel.attach_lora(lora_config)` wraps the backbone with `peft.get_peft_model`, which
freezes the shared base weights and leaves only the LoRA adapter parameters trainable. Toggling
between the two states then uses `peft`'s own adapter-disable mechanism rather than swapping in a
second model:

- Student (LoRA ON, the default): `model(...)`.
- Teacher (LoRA OFF): `with model.backbone.disable_adapter(): model(...)`, under `torch.no_grad()`
  and with the resulting teacher logits detached before use in the KL term (§4.12), so gradients
  never flow through the teacher path.

---

# 4.14 Medusa-2 Combined Training Objective

Medusa-2 combines the self-distillation objective with the Medusa-head
training objective.

The Medusa-head loss remains:

L_Medusa-1 =
Σ_(k=1)^K
    -λ_k log p_t^(k)(y_(t+k+1))

The backbone self-distillation loss is:

L_LM-distill =
KL(
    p_original,t^(0)
    ||
    p_t^(0)
)

The combined Medusa-2 objective is:

L_Medusa-2 =
L_LM-distill
+
λ_0 L_Medusa-1

where:

- L_LM-distill preserves the original model's output distribution.
- L_Medusa-1 trains the future-token Medusa heads.
- λ_0 controls the relative contribution of the Medusa-head objective.

The implementation must compute gradients only through the trainable
student path while treating the original-model distribution as the
teacher target.

---

# 4.15 Medusa-2 Self-Distillation Training Flow

The complete training flow is:

Seed prompts
      ↓
Original model
      ↓
Synthetic self-distillation dataset
      ↓
Input sequence
      ↓
Shared base model
      │
      ├─────────────── LoRA OFF ───────────────┐
      │                                         ↓
      │                                  Original / Teacher
      │                                         ↓
      │                                  p_original,t^(0)
      │                                         │
      │                                         │
      └─────────────── LoRA ON ────────────────┐│
                                              ↓↓
                                        Student model
                                              ↓
                                         p_t^(0)
                                              ↓
                                      KL divergence
                                              ↓
                                        L_LM-distill

Student model hidden states
              ↓
        Medusa heads
              ↓
       p_t^(1), ..., p_t^(K)
              ↓
        L_Medusa-1
              ↓
              λ_0
              ↓
L_Medusa-2 = L_LM-distill + λ_0 L_Medusa-1
              ↓
        Backpropagation
              ↓
      Update trainable parameters

The teacher path must not receive gradient updates.

---

# 4.16 Complete Medusa Training Algorithm

The training implementation must support three related training modes.

### Mode 1: Medusa-1 — Frozen Backbone

1. Load the pretrained base LLM.
2. Freeze all backbone parameters.
3. Attach K Medusa heads.
4. Initialize W_2^(k) from the original language-model head.
5. Initialize W_1^(k) to zero.
6. Run the input sequence through the frozen backbone.
7. Obtain hidden states h_t.
8. Generate predictions from all Medusa heads.
9. Compute L_k for every head.
10. Apply head-specific weights λ_k.
11. Compute L_Medusa-1.
12. Backpropagate through the Medusa heads.
13. Update only Medusa-head parameters.

### Mode 2: Medusa-2 — Joint Training with Ground-Truth Data

1. Load the pretrained model and Medusa heads.
2. Enable training of the backbone and Medusa heads.
3. Compute the original model next-token prediction.
4. Compute the original language-model loss if the standard joint
   training objective is being used.
5. Compute predictions from all Medusa heads.
6. Compute L_Medusa-1.
7. Combine the objectives according to the configured Medusa-2 loss.
8. Backpropagate.
9. Update the trainable parameters.

### Mode 3: Medusa-2 — Self-Distillation

1. Select a public seed dataset.
2. Extract prompts from the seed dataset.
3. Use the original model to generate responses.
4. Construct the synthetic self-distillation dataset.
5. Load the shared base model.
6. Attach the LoRA adapter.
7. Run the teacher path with LoRA disabled.
8. Obtain p_original,t^(0).
9. Run the student path with LoRA enabled.
10. Obtain p_t^(0).
11. Compute:

    L_LM-distill =
    KL(p_original,t^(0) || p_t^(0))

12. Compute predictions from the Medusa heads.
13. Compute:

    L_Medusa-1 =
    Σ_(k=1)^K -λ_k log p_t^(k)(y_(t+k+1))

14. Compute:

    L_Medusa-2 =
    L_LM-distill + λ_0 L_Medusa-1

15. Backpropagate only through the trainable student/Medusa paths.
16. Update the trainable parameters.
17. Keep the teacher distribution detached from gradient computation.

---

# 4.17 Complete Medusa Inference Algorithm

One complete inference iteration consists of:

### Step 1: Generate hidden representation

Current context:

x_1, ..., x_n

is passed through the base LLM:

x_1, ..., x_n
      ↓
Base LLM
      ↓
h_n

### Step 2: Generate candidate predictions

The original language-model head and Medusa heads generate probability
distributions.

Top-s_k predictions are selected from each relevant head.

### Step 3: Construct candidate tree

The selected predictions are combined using the Cartesian product.

For K heads:

N_candidates = ∏_(k=1)^K s_k

Each combination corresponds to a candidate branch.

### Step 4: Construct tree attention mask

The parent-child relationships of the candidate tree are converted into
a tree attention mask.

Each candidate token can attend to the original context and its
ancestors, but not to unrelated branches.

### Step 5: Target-model verification

All candidate branches are evaluated simultaneously by the target model.

### Step 6: Calculate acceptance criterion

For each candidate:

1. Obtain target-model probability.
2. Obtain target-model probability distribution.
3. Calculate entropy.
4. Calculate:

   threshold =
   min(ε, δ exp(-H(p_original)))

5. Compare candidate probability with the threshold.

### Step 7: Accept candidates

Candidates satisfying:

p_original(candidate | context) > threshold

are accepted.

### Step 8: Update KV cache

The KV cache is updated using the accepted sequence.

Reusable states are retained for the next decoding iteration.

### Step 9: Continue generation

The updated sequence and KV cache are used for the next Medusa
decoding iteration.

---

# 4.18 Complete Medusa Inference Flow

Input prompt
     ↓
Base LLM
     ↓
Hidden representation h_t
     ↓
┌──────────────────────────────┐
│ Original LM head             │
│ Medusa Head 1                │
│ Medusa Head 2                │
│ ...                          │
│ Medusa Head K                │
└──────────────────────────────┘
     ↓
Top-s_k predictions
     ↓
Cartesian-product candidate construction
     ↓
Candidate tree
     ↓
Tree attention mask
     ↓
Target-model parallel verification
     ↓
Target-model probability distributions
     ↓
Entropy calculation
     ↓
Acceptance threshold
     ↓
Candidate acceptance
     ↓
Accepted token sequence
     ↓
KV-cache update
     ↓
Next decoding iteration

---

# 4.19 Algorithmic Requirements

The implementation must satisfy the following requirements:

### Medusa Heads

- Each Medusa head must receive the base LLM hidden representation.
- Each head must predict a different future token position.
- Each head must have independent trainable parameters.
- W_2^(k) must be initialized from the original language-model head.
- W_1^(k) must be initialized to zero.
- The head must use the specified SiLU activation and residual connection.

### Candidate Generation

- Support configurable s_k for every Medusa head.
- Support different s_k values across heads.
- Construct candidates using the Cartesian product.
- Maintain explicit parent-child relationships.
- Map every candidate node to its corresponding head and tree position.

### Tree Attention

- Construct the attention mask from the candidate tree.
- Allow attention to original context tokens.
- Allow attention to ancestor tokens.
- Prevent cross-branch attention.
- Maintain consistent positional indices.

### Verification and Acceptance

- Verify candidate branches using the target/original model.
- Compute candidate probabilities from the target model.
- Compute target-model entropy.
- Implement the entropy-dependent acceptance threshold.
- Apply the hard threshold ε.
- Accept only candidates satisfying the acceptance criterion.

### Medusa-1 Training

- Support frozen-backbone training.
- Compute a separate cross-entropy loss for every Medusa head.
- Support λ_k head-specific loss weights.
- Update only Medusa-head parameters.

### Self-Distillation

- Support generation of a synthetic training dataset from seed prompts.
- Support multi-turn response generation.
- Use the original model's probability distribution as the teacher.
- Compute KL divergence between teacher and student distributions.
- Support LoRA-based teacher/student separation.
- Disable LoRA for the teacher path.
- Enable LoRA for the student path.
- Prevent gradients from flowing through the teacher distribution.

### Medusa-2 Training

- Support joint training of the backbone and Medusa heads.
- Support the self-distillation objective.
- Support the combined objective:

  L_Medusa-2 =
  L_LM-distill + λ_0 L_Medusa-1

- Keep the teacher model distribution fixed during optimization.

### KV Cache

- Support tree-structured candidate verification.
- Reuse previously computed key-value states.
- Update the cache according to accepted candidates.

### Inference

- Perform candidate generation and target verification efficiently.
- Verify multiple candidate branches in parallel.
- Return accepted tokens.
- Update the KV cache after every decoding iteration.