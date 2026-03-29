# EAT: Self-Supervised Pre-Training with Efficient Audio Transformer

[cite_start]**Authors:** Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, Xie Chen (Shanghai Jiao Tong University) [cite: 3]

---

## 1. Abstract
[cite_start]The Efficient Audio Transformer (EAT) addresses the massive computational barriers in audio self-supervised learning (SSL)[cite: 6]. [cite_start]By combining a bootstrap training paradigm with a novel **Utterance-Frame Objective (UFO)** and an **inverse block masking strategy**, EAT achieves state-of-the-art (SOTA) results with a pre-training speedup of approximately **15x** compared to existing models[cite: 7, 8, 11].

---

## 2. Introduction
* [cite_start]Audio SSL leverages large-scale unlabeled data to learn robust representations[cite: 5, 17].
* [cite_start]Existing models like Audio-MAE or BEATs are often slowed down by complex decoders or inefficient tokenization processes[cite: 28, 29, 32].
* [cite_start]EAT introduces a dual-level objective (UFO) that captures both global audio semantics and local frame nuances[cite: 35, 36].

---

## 3. Methodology

### 3.1 Model Architecture
* [cite_start]**CNN Encoder:** Transforms audio spectrograms into non-overlapping patch embeddings $X_{p} \in R^{P \times E}$[cite: 117, 119, 165].
* [cite_start]**Asymmetric Design:** Uses a 12-layer ViT-B Transformer for the student/teacher encoders and a lightweight 6-layer 2D CNN for the student decoder[cite: 166, 167].
* [cite_start]**Bootstrap Framework:** The student model is updated via the UFO loss, while the teacher model is updated through an **Exponential Moving Average (EMA)** of the student's parameters[cite: 38, 110, 169].

### 3.2 Utterance-Frame Objective (UFO)
[cite_start]The loss function is defined as $L_{UFO} = L_{f} + \lambda L_{u}$[cite: 140, 141]:
* [cite_start]**Utterance Loss ($L_u$):** Regresses the student's CLS token feature against the average value of the teacher's multi-layer outputs[cite: 127, 129, 130].
* **Frame Loss ($L_f$):** Uses the MAE method to reconstruct latent representations at masked positions[cite: 136, 137, 138].

### 3.3 Masking Strategy
* **High Masking Ratio:** EAT uses an 80% mask ratio to reduce the data volume processed by the Transformer[cite: 39, 145, 146].
* [cite_start]**Inverse Block Masking:** Instead of random 1D masking, EAT preserves unmasked data in 2D blocks (optimally $5 \times 5$), which increases the challenge of predicting masked features[cite: 41, 151, 315].
* [cite_start]**Multi-Masking:** To optimize teacher-student efficiency, EAT creates 16 clones of masked data per audio clip to amplify parallel data utilization[cite: 43, 44, 198].

### 3.4 Paper-to-Code Mapping

| Paper component | Code mapping | Notes |
| :--- | :--- | :--- |
| Audio spectrogram frontend and patch embedding | [models/images.py](models/images.py#L76), [models/images.py](models/images.py#L89) | Audio is treated as a 1-channel time-frequency map with target length 1024, then embedded with a 16-pixel patch stem. |
| 12-layer ViT student/teacher backbone | [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L85), [models/EAT_pretraining.py](models/EAT_pretraining.py#L67) | The default pretraining config sets depth to 12 and embed dim to 768, matching the paper’s ViT-B-sized encoder. |
| 6-layer CNN decoder | [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L117), [models/images.py](models/images.py#L141) | The default decoder is a 6-layer 2D CNN-style decoder, which matches the paper’s asymmetric design. |
| EMA teacher / bootstrap update | [models/EAT_pretraining.py](models/EAT_pretraining.py#L314), [models/EAT_pretraining.py](models/EAT_pretraining.py#L330) | The teacher is created as an EMA copy and updated in `set_num_updates`, matching the bootstrap setup. |
| Inverse block masking, 80% mask ratio, 5x5 blocks | [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L107), [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L108), [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L110), [models/images.py](models/images.py#L228), [models/base.py](models/base.py#L363), [utils/data_utils.py](utils/data_utils.py#L211) | The config enables inverse masking with 0.8 mask probability and 5x5 blocks; the mask logic is implemented in the encoder and utility helper. |
| Multi-masking with 16 clones | [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L87), [models/EAT_pretraining.py](models/EAT_pretraining.py#L93) | `clone_batch=16` is the repository’s implementation of the paper’s multi-mask strategy. |
| Teacher target averaging across layers | [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L86), [models/EAT_pretraining.py](models/EAT_pretraining.py#L623), [models/EAT_pretraining.py](models/EAT_pretraining.py#L785) | The code averages the last 12 teacher layers before forming targets, which is consistent with the paper’s multi-layer UFO target. |
| Utterance-level UFO loss on CLS token | [config/pretraining_AS2M.yaml](config/pretraining_AS2M.yaml#L100), [models/EAT_pretraining.py](models/EAT_pretraining.py#L651), [models/EAT_pretraining.py](models/EAT_pretraining.py#L659) | The default config enables CLS loss, and the implementation regresses the student CLS token toward the teacher utterance target. |
| Frame-level masked regression loss | [models/EAT_pretraining.py](models/EAT_pretraining.py#L703), [models/EAT_pretraining.py](models/EAT_pretraining.py#L760) | This is the masked latent regression term that corresponds to the paper’s frame loss. |
| CLS-token fine-tuning head | [config/finetuning.yaml](config/finetuning.yaml#L85), [models/EAT_audio_classification.py](models/EAT_audio_classification.py#L75), [models/EAT_audio_classification.py](models/EAT_audio_classification.py#L384) | Fine-tuning defaults to CLS-token prediction, which matches the paper’s ablation finding. |

---

## 4. Experiments and Results

### 4.1 Performance Benchmarks
[cite_start]EAT was pre-trained on AudioSet-2M (AS-2M) for 10 epochs[cite: 182, 197].

| Dataset | Metric | EAT Performance | Comparison |
| :--- | :--- | :--- | :--- |
| **AudioSet (AS-2M)** | mAP | **48.6%** | [cite_start]Outperforms previous SOTA by 0.6% [cite: 206, 240] |
| **AudioSet (AS-20K)** | mAP | **40.2%** | [cite_start]Surpasses previous SOTA by 1.9% [cite: 207, 240] |
| **ESC-50** | Accuracy | **95.9%** | [cite_start]Reduced error rate from 4.4% to 4.1% [cite: 208, 240] |
| **SPC-2** | Accuracy | **98.3%** | [cite_start]Competitive with previous SOTA models [cite: 212, 240] |

### 4.2 Training Efficiency
[cite_start]EAT demonstrates massive efficiency gains in "wall-clock" time[cite: 236]:
* **vs. [cite_start]$BEATs_{iter3}$:** 15.65x speedup[cite: 237, 244].
* **vs. [cite_start]Audio-MAE:** 10.02x speedup[cite: 237, 244].
* [cite_start]EAT matches Audio-MAE's performance in just **2 epochs**[cite: 238].

---

## 5. Key Ablation Findings
* [cite_start]**Utterance Weight:** A balanced ratio ($\lambda=1$) provides the best performance; overemphasizing global features ($\lambda=10$) degrades results[cite: 305, 306].
* **Prediction Head:** Using a **CLS token** for classification in fine-tuning consistently outperforms mean pooling[cite: 180, 307].
* [cite_start]**Block Size:** Increasing block size from $1 \times 1$ to $5 \times 5$ improved mAP from 37.8% to 40.2% by reducing mutual information between patches[cite: 314, 315, 322].

---

## 6. Conclusion
[cite_start]EAT provides a highly efficient pathway for training high-performance audio SSL models[cite: 325, 326]. [cite_start]The authors plan to scale the model further and explore joint audio-speech training in future iterations[cite: 331, 332].