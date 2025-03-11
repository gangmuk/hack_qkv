# hack_qkv


Let me walk through the exact computational steps for the "I like apple" example from start to finish:

### 1. Token Embeddings

First, we get token embeddings for each word. Let's assume a tiny model with embedding dimension of 4:

```
"I"     → [0.1, 0.2, 0.3, 0.4]
"like"  → [0.5, 0.6, 0.7, 0.8]
"apple" → [0.9, 1.0, 1.1, 1.2]
```

### 2. Computing Q, K, V Projections (in parallel)

For simplicity, let's assume 2 attention heads with 2 dimensions each. Each token's embedding is projected to Q, K, and V using separate weight matrices:

Q projection (W_Q):
```
"I"     → [[0.11, 0.12], [0.13, 0.14]]
"like"  → [[0.51, 0.52], [0.53, 0.54]]
"apple" → [[0.91, 0.92], [0.93, 0.94]]
```

K projection (W_K):
```
"I"     → [[0.15, 0.16], [0.17, 0.18]]
"like"  → [[0.55, 0.56], [0.57, 0.58]]
"apple" → [[0.95, 0.96], [0.97, 0.98]]
```

V projection (W_V):
```
"I"     → [[0.19, 0.20], [0.21, 0.22]]
"like"  → [[0.59, 0.60], [0.61, 0.62]]
"apple" → [[0.99, 1.00], [1.01, 1.02]]
```

These projections happen in parallel for all tokens.

### 3. Computing Attention Scores (per head)

For each attention head, we compute Q·K^T to get attention scores. Let's focus on the first head:

Q_head1:
```
"I"     → [0.11, 0.12]
"like"  → [0.51, 0.52]
"apple" → [0.91, 0.92]
```

K_head1:
```
"I"     → [0.15, 0.16]
"like"  → [0.55, 0.56]
"apple" → [0.95, 0.96]
```

Attention scores (Q·K^T) for head 1:
```
         | "I"               | "like"            | "apple"           |
---------|-------------------|-------------------|-------------------|
"I"      | 0.11*0.15+0.12*0.16 | 0.11*0.55+0.12*0.56 | 0.11*0.95+0.12*0.96 |
"like"   | 0.51*0.15+0.52*0.16 | 0.51*0.55+0.52*0.56 | 0.51*0.95+0.52*0.96 |
"apple"  | 0.91*0.15+0.92*0.16 | 0.91*0.55+0.92*0.56 | 0.91*0.95+0.92*0.96 |
```

Computing the actual values:
```
         | "I"    | "like"  | "apple" |
---------|--------|---------|---------|
"I"      | 0.0347 | 0.1267  | 0.2187  |
"like"   | 0.1607 | 0.5867  | 1.0127  |
"apple"  | 0.2867 | 1.0467  | 1.8067  |
```

### 4. Applying Causal Mask

The causal mask ensures each token only attends to itself and previous tokens:
```
         | "I"    | "like"  | "apple" |
---------|--------|---------|---------|
"I"      | 0.0347 | -inf    | -inf    |
"like"   | 0.1607 | 0.5867  | -inf    |
"apple"  | 0.2867 | 1.0467  | 1.8067  |
```

### 5. Applying Softmax (row-wise)

After applying softmax to each row:
```
         | "I"    | "like"  | "apple" |
---------|--------|---------|---------|
"I"      | 1.0000 | 0.0000  | 0.0000  |
"like"   | 0.2980 | 0.7020  | 0.0000  |
"apple"  | 0.1270 | 0.3379  | 0.5351  |
```

### 6. Computing Weighted Values

For head 1, we have V values:
```
"I"     → [0.19, 0.20]
"like"  → [0.59, 0.60]
"apple" → [0.99, 1.00]
```

We multiply the attention weights with V values:

For "I":
```
1.0000 * [0.19, 0.20] + 0.0000 * [0.59, 0.60] + 0.0000 * [0.99, 1.00] = [0.19, 0.20]
```

For "like":
```
0.2980 * [0.19, 0.20] + 0.7020 * [0.59, 0.60] + 0.0000 * [0.99, 1.00] = [0.47, 0.48]
```

For "apple":
```
0.1270 * [0.19, 0.20] + 0.3379 * [0.59, 0.60] + 0.5351 * [0.99, 1.00] = [0.79, 0.80]
```

### 7. Final Output (after combining heads)

We would repeat steps 3-6 for the second head, then concatenate and project the results to get the final output for each token.

From a computational perspective, key points:
1. Q, K, V projections happen in parallel for all tokens
2. Attention score computation (Q·K^T) can also happen in parallel 
3. The causal mask is what enforces the dependency on earlier tokens
4. Each token's final representation incorporates information from previous tokens through the weighted sum of values

The computations aren't inherently sequential, but the causal structure of the attention mask creates an effective information flow from earlier tokens to later ones.