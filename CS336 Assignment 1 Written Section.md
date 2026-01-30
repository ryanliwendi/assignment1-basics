## CS336 Assignment 1 Writeup

**Problem: Understanding Unicode**

(a) $chr(0)$ returns '\x00', which is Python's escape notation of the NULL character.

(b) The printed representation shows the character itself, while the string representation shows the escaped form to ensure reproducibility and unambiguity. For example, the printed representation of null is nothing, while the string representation prints '\x00'. 

(c) When concatenated in text, the null character is stored as part of the string. When printed, it is invisible. 


**Problem: Unicode Encodings**

(a) We prefer UTF-8 since its outputs are shorter and more concise than UTF-16 or UTF-32. Furthermore, we don't have to worry about endianness for UTF-8.

(b) The string "你好" yields an incorrect output (raises a decoding error); this is because some UTF-8 characters take multiple bytes, so you cannot decode them byte-by-byte. 

(c) '\x80\x80' is an invalid sequence because they are both continuation bytes (10xxxxxx) in UTF-8, without a leading byte (110xxxxx/1110xxxx/11110xxx) to start the character.


*Note: the general workflow of a BPE tokenizer is as follows: 
Step 1. Initialize the vocabulary, being the set of all 256 possible byte values and the special tokens. 
Step 2. If we choose to parallel pretokenization, we split the corpus into chunks and assign them to different workers. Make sure that chunk boundaries occur at the beginnings of special tokens.
Step 3. For each chunk, we further split them based on special tokens. This is done to ensure that no merging can occur across the text they delimit. We then remove the special tokens.
Step 4. Pre-tokenize each chunk of text and count the pre-tokens.
Step 5. Build initial pair counts. Maintain a dictionary for pair frequencies and a dictionary for a set of tokens where the pair is located in.
Step 6. Merge the max-frequency pair to become the new vocabulary and update the affected pre-tokens accordingly. For each pair in the affected pre-tokens, remove their occurrences in the old pre-token and replace them by occurrences in the new pre-tokens.*

**Problem: BPE Training on TinyStories**

(a) Training took 7.44 minutes, with peak memory usage of ~8GB.  The longest token in the vocabulary translates to " accomplishment", which makes sense because it's a quite common word. 

(b) Using scalene to profile my code, the three steps that take the most time are: reading the corpus, splitting the corpus into chunks, and splitting pretokens into tuples of bytes. 


**Problem: BPE Training on OpenWebText**

(a) Training completed in 5.79 hours. The longest vocabulary is "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ", which is a common placeholder for mis-encoding errors.

(b) The TinyStories tokenizer often contains simple English words, while the OpenWebText tokenizer is a lot broader and messier.


**Problem: Experiments with Tokenizers**

(a) The compression ratio for the TinyStories tokenizer is 4.149 bytes/token. The compression ratio for the OpenWebText tokenizer is 4.284 bytes/token. 

(b) The compression ratio significantly reduces to 3.287 bytes/token if we use the TinyStories tokenizer on the OpenWebText sample.

(c) The throughput of the TinyStories tokenizer is 2.459 MB/s, and the throughput of the OpenWebText tokenizer is 2.028 MB/s. To tokenizer the Pile dataset (825 GB), they would require 95.4 hours and 115.7 hours, respectively. 

(d) uint16 is an appropriate choice because its representable range in 0-65535, which is much larger than the vocab size of 10K and 32K, respectively. Thus uint16 is the most memory-efficient choice.


**Problem: Transformer LM Resource Accounting**

(a) The trainable parameters in this architecture include:

- 80,441,200 parameters for the embedding dimension
- 48 blocks of transformer layers, each containing 10,240,000 parameters for attention, 1,600 parameters for attention norm, 30,720,000 parameters for FFN, 1,600 parameters for FFN norm
- 1,600 parameters for final norm; and 80,411,200 parameters for output layer. 

There are 2,127,057,600 total trainable parameters in this model. Storing them using float32 would require a total of 8,508,230,400 bytes, which is around 8GB to load the model weights.

(b) For each forward pass, each matrix multiply costs the following FLOPs:

- For each transformer block:
  - Attention Projection ($8 * seq * d_{model}^2$) = 20.97B
  - Attention Computation ($4 * seq^2 *d_{model}$) = 6.71B
  - FFN ($6*seq*d_{model}*d_{ff}$) = 62.91B
- Final Layer ($2 * seq*d_{model}*vocab$)  = 164.7B

The total FLOPs count is ~4.51T.

(c) The FFN layers requires the most FLOPs.

(d) For GPT-2 small, the total FLOPs is ~350B, with 16.6% on attention projections, 11.1% on attention computations, 49.8% on FFN, and 22.6% on the final layer. 
For GPT-2 medium, the total FLOPs is ~1.03T, with 20.0% on attention projections, 10.0% on attention computations, 59.9% on FFN, and 10.2% on the final layer.
For GPT-2 large, the total FLOPs is ~2.26T, with 21.4% on attention projections, 8.6% on attention computations, 64.2% on FFN, and 5.2% on the final layer. 
For GPT-2 XL, the total FLOPs is ~4.51T, with 22.3% on attention projections, 7.1% on attention computations, 66.9% on FFN, and 3.6% on the final layer. 
As the model size increases, FFN dominates more and more, and the percentage on attention projection grows modestly. The attention computations and final projection drops. 

(e) By increasing the context length from 1,024 to 16,384, the total FLOPs increases from 4.51B to ~150B. The FLOPs include 55.2% on attention computations, 32.3% on FFN, 10.8% on attention projections, and 1.8% on the final projection. The attention computation, which scales with $seq^2$, explodes.

**Problem: Tuning the Learning Rate**

(a) The loss is 4.01 for lr=1e1, 1.83e-23 for lr=1e2, and 2.49e18 for lr=1e3. The loss for lr=1e1 decays slow, and the loss for lr=1e2 decays very fast. The loss for lr=1e3 diverges.

**Problem: Resource Accounting for Training with AdamW**
