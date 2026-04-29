# Architecture Ideas

## Amino Acid Interaction Layer for Selection Models

**Problem:** Current DASM models struggle to produce highly differentiated selection factors at the same position. All 20 amino acids at position *i* are predicted via a single linear projection from the same context embedding, creating strong correlations between scores.

**Current architecture:**
```python
context_embedding[i] = transformer(sequence)[i]  # [d_model]
log_selection[i, :] = linear(context_embedding[i])  # [20] - all linearly related
```

**Proposed solution:** Add learned amino acid embeddings that interact with positional context:

### Option A: Hadamard product + MLP
```python
# Learn intrinsic amino acid embeddings
aa_embeds = nn.Embedding(20, d_model)

# For each position:
context = transformer_output[i]  # [d_model]
for each amino acid:
    interaction = context * aa_embeds[aa]  # Element-wise product
    score[i, aa] = mlp(interaction)
```

### Option B: Bilinear interaction
```python
score[i, aa] = context[i]^T @ W @ aa_embeds[aa]
```

### Option C: Concatenation
```python
score[i, aa] = mlp(concat(context[i], aa_embeds[aa]))
```

**Benefits:**
- Breaks linear correlation bottleneck
- Model can learn "context + W = very bad" while "context + F = moderately bad"
- Amino acid embeddings may capture physicochemical properties
- Context-dependent (not absolute position-dependent)

**Trade-offs:**
- Slightly more parameters
- Slightly slower inference
- May need more training data

**Status:** Idea only, not yet implemented

**Date:** 2025-11-03
