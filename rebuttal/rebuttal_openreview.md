<!--
Each top-level reviewer section below is a standalone OpenReview response.
Copy one section at a time into the corresponding reviewer forum.
LaTeX delimiters are escaped for OpenReview Markdown.
-->

# Response to Reviewer WUoc

We sincerely thank the reviewer for the careful reading and constructive remarks, which prompted us to strengthen the score-level experiments and clarify the scope of the theory.

## Q1: What justifies replacing the real score matrix by its spectral representative?

1. **Empirical side.** We directly compare the actual per-head attention

\\[
A_{{\rm real},h}=\operatorname{softmax}_{\rm row}(S_h+M)
\\]

with the spectral-canonical attention

\\[
A_{\Sigma,h}=\operatorname{softmax}_{\rm row}(\Sigma_h+M),
\qquad
\Sigma_h=\operatorname{diag}(\operatorname{svdvals}(S_h)).
\\]

Every score and attention SVD is computed independently per sample, layer, and head. For the empirical singular-value measures

\\[
\mu_{{\rm real},h}^{(\ell)}
=\ell^{-1}\sum_j\delta_{\sigma_j(A_{{\rm real},h})},
\qquad
\mu_{\Sigma,h}^{(\ell)}
=\ell^{-1}\sum_j\delta_{\sigma_j(A_{\Sigma,h})},
\\]

the one-dimensional distance is

\\[
W_1(\mu_{{\rm real},h}^{(\ell)},\mu_{\Sigma,h}^{(\ell)})
=\ell^{-1}\sum_j
\left|\sigma_j^\downarrow(A_{{\rm real},h})
-\sigma_j^\downarrow(A_{\Sigma,h})\right|.
\\]

We use \\(W_1\\) because it is bin-free and metrizes weak convergence together with first-moment convergence; therefore \\(W_1\to0\\) implies asymptotically equivalent LSDs whenever either limit exists. On 12 WikiText-103 documents, seven lengths, 12 layers, and 12 heads (12,096 comparisons), the all-layer median decreases from 0.3002 to 0.0942 and the last-layer median from 0.3290 to 0.0748 as \\(\ell\\) grows from 64 to 1024 (**Figure R.1**). Representative last-layer rows, rounded to two decimals, are:

| GPT-2 layer | 64 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|
| 8 | 0.39 ± 0.10 | 0.22 ± 0.03 | 0.14 ± 0.02 | 0.11 ± 0.04 |
| 9 | 0.34 ± 0.08 | 0.21 ± 0.02 | 0.14 ± 0.01 | 0.10 ± 0.04 |
| 10 | 0.37 ± 0.08 | 0.22 ± 0.02 | 0.13 ± 0.02 | 0.09 ± 0.04 |
| 11 | 0.36 ± 0.06 | 0.21 ± 0.02 | 0.12 ± 0.01 | 0.08 ± 0.02 |
| 12 | 0.31 ± 0.08 | 0.20 ± 0.02 | 0.13 ± 0.01 | 0.08 ± 0.02 |

For ViT-B/16, all 12 layer medians are below 0.104 over 2,880 image-layer-head comparisons (**Figure R.2**); representative layers, rounded to two decimals, are:

| ViT layer | Median \\(W_1\\) | IQR |
|---:|---:|---:|
| 1 | 0.04 | [0.02, 0.10] |
| 3 | 0.10 | [0.05, 0.45] |
| 6 | 0.07 | [0.06, 0.07] |
| 9 | 0.05 | [0.04, 0.06] |
| 12 | 0.04 | [0.03, 0.04] |

We also test the actual score decomposition \\(S_h=U_{S_h}\Sigma_hV_{S_h}^{\top}\\). For GPT-2, 237,024 score SVDs show decreasing discrepancies from the Haar eigen-angle reference:

| \\(\ell\\) band | \\(U\\): KL | \\(U\\): Kol. | \\(U\\): \\(W_1\\) | \\(V\\): KL | \\(V\\): Kol. | \\(V\\): \\(W_1\\) |
|---:|---:|---:|---:|---:|---:|---:|
| 17--64 | 0.34 | 0.04 | 0.31 | 0.34 | 0.04 | 0.31 |
| 193--256 | 0.01 | 0.01 | 0.13 | 0.01 | 0.01 | 0.13 |
| 385--471 | <0.01 | 0.01 | 0.10 | <0.01 | 0.01 | 0.10 |

For ViT, 565,200 score SVDs give uniformly low discrepancies across layers:

| Side | KL \\((\times10^{-3})\\) | Kolmogorov \\((\times10^{-3})\\) | Angular \\(W_1\\) |
|:---|---:|---:|---:|
| Left \\(U_S\\) | 2.21 ± 0.72 | 4.20 ± 0.62 | 0.08 ± 0.04 |
| Right \\(V_S\\) | 2.21 ± 0.72 | 4.21 ± 0.62 | 0.08 ± 0.04 |

These diagnostics test the orientation component; independence from \\(\Sigma_h\\) remains an explicit hypothesis (**Figures R.3--R.4**).

2. **Mathematical side.** Assumption 1 states \\(S\overset d=\Sigma\\), motivated by the standard invariant-ensemble basis reduction used by Benaych-Georges and Nadakuditi (2011), with Haar left/right orientations independent of \\(\Sigma\\). For the fixed-mask measurable map

\\[
F_M(X)=\operatorname{softmax}_{\rm row}(X+M),
\\]

equality in distribution is preserved:

\\[
S\overset d=\Sigma
\quad\Longrightarrow\quad
F_M(S)\overset d=F_M(\Sigma).
\\]

Thus the replacement is the direct consequence of the stated assumption, and the \\(W_1\\) experiment tests its post-softmax weak-LSD prediction. See also Reviewer CLzG Q1/Q3 and Reviewer XnLy Q1.

**Figure R.1: GPT-2 real-versus-canonical attention LSD distance.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png

**Figure R.2: ViT real-versus-canonical attention LSD distance.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png

**Figure R.3: GPT-2 actual-score Haar diagnostics.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

**Figure R.4: ViT actual-score Haar diagnostics.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

**Conclusion:** The replacement is justified conditionally by \\(S\overset d=\Sigma\\) and is strongly supported by the decreasing real-versus-canonical \\(W_1\\) discrepancy and actual-score orientation diagnostics.

## Q2: Does the spike--bulk separation occur for real score matrices?

1. **Empirical side.** We compute the SVD of every actual per-head score \\(S_h\\) independently and pool singular values only afterward within each layer. **Figures R.5--R.6** show all 12 layers and heads for a 1024-token GPT-2 WikiText input and a 577-token ViT-B/16 Imagenette input. The log-scale histograms display a lower bulk and a separated high-value group throughout the layer panels. The color split is descriptive: it is placed at the largest adjacent gap in the upper half of the layer-median per-head scree curve and is not claimed to be the paper's asymptotic \\(\log\ell\\) cutoff.

Experiment 4.2/Figure 3, reproduced as **Figure R.7**, supplies dataset-scale GPT-2 evidence: for \\(17\le\ell\le471\\), stable rank stays below about 2 and effective rank below about 30, with no substantial growth in length. This supports a finite/moderate effective number of dominant directions over the tested range.

2. **Mathematical side.** For each head,

\\[
k\le\operatorname{rank}(S_h)\le d_h=64,
\\]

so fixed head dimension provides an \\(\ell\\)-independent cap. The histograms support the finite-sample morphology, while the exact super-/sub-logarithmic separation remains the asymptotic assumption. This use of a spiked model is also consistent with Seddik et al., ICML 2020. See also Reviewer CLzG Q2 and Reviewer XnLy Q4.

**Figure R.5: GPT-2 actual score spectra.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.6: ViT actual score spectra.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.7: Score stable/effective ranks versus length.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png

**Conclusion:** The representative GPT-2 and ViT inputs show a practical bulk--spike morphology in every layer panel, and the GPT-2 rank proxies remain finite/moderate over the tested lengths.

## Q3: Can the framework include MHA, LayerNorm, and other Transformer components?

The present paper isolates the attention operator and does not propagate the spectral law through head concatenation, \\(W_O\\), residual addition, and normalization. The analyzed scores are nevertheless extracted from intact pretrained networks, so they already reflect the upstream normalization and residual dynamics that determine the attention input. Explicit propagation through the complete multi-head block is an important and ambitious direction that we are pursuing in follow-up work. See Reviewer XnLy Q3 for the per-head formulation and Reviewer CLzG Q4 for the head-resolved diagnostics.

**Conclusion:** The current theory covers isolated attention; explicit MHA composition, output projection, residual propagation, and normalization remain future extensions.

# Response to Reviewer CLzG

We thank the reviewer for the thoughtful and constructive questions, which motivated direct spectral comparisons and clearer head-resolved evidence.

## Q1: Can the spectra of \\(\operatorname{softmax}(S)\\) and \\(\operatorname{softmax}(\Sigma)\\) be compared directly?

1. **Empirical side.** Yes. For every sample, layer, and head we independently decompose

\\[
A_{{\rm real},h}=\operatorname{softmax}_{\rm row}(S_h+M),
\qquad
A_{\Sigma,h}=\operatorname{softmax}_{\rm row}(\Sigma_h+M),
\\]

and compute their bin-free \\(W_1\\) distance before aggregation. GPT-2 uses 12 WikiText-103 documents, seven lengths, 12 layers, and 12 heads (12,096 comparisons). Each cell below is mean \\(W_1\pm\\) standard deviation over 12 documents \\(\times\\) 12 heads, rounded to two decimals:

| Layer | 64 | 128 | 256 | 384 | 512 | 768 | 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.39 ± 0.10 | 0.31 ± 0.07 | 0.22 ± 0.03 | 0.17 ± 0.02 | 0.14 ± 0.02 | 0.11 ± 0.03 | 0.11 ± 0.04 |
| 9 | 0.34 ± 0.08 | 0.28 ± 0.06 | 0.21 ± 0.02 | 0.16 ± 0.01 | 0.14 ± 0.01 | 0.11 ± 0.03 | 0.10 ± 0.04 |
| 10 | 0.37 ± 0.08 | 0.30 ± 0.05 | 0.22 ± 0.02 | 0.16 ± 0.01 | 0.13 ± 0.02 | 0.10 ± 0.03 | 0.09 ± 0.04 |
| 11 | 0.36 ± 0.06 | 0.28 ± 0.04 | 0.21 ± 0.02 | 0.16 ± 0.01 | 0.12 ± 0.01 | 0.10 ± 0.02 | 0.08 ± 0.02 |
| 12 | 0.31 ± 0.08 | 0.26 ± 0.05 | 0.20 ± 0.02 | 0.16 ± 0.01 | 0.13 ± 0.01 | 0.10 ± 0.01 | 0.08 ± 0.02 |

The all-layer median decreases from 0.3002 to 0.0942 and the last-layer median from 0.3290 to 0.0748 (**Figure R.1**). For ViT-B/16, all 12 layer medians are below 0.104 over 2,880 comparisons (**Figure R.2**):

| ViT layer | Median \\(W_1\\) | IQR |
|---:|---:|---:|
| 1 | 0.04 | [0.02, 0.10] |
| 3 | 0.10 | [0.05, 0.45] |
| 6 | 0.07 | [0.06, 0.07] |
| 9 | 0.05 | [0.04, 0.06] |
| 12 | 0.04 | [0.03, 0.04] |

2. **Mathematical connection.** For the empirical singular-value measures

\\[
\mu_{{\rm real},h}^{(\ell)}
=\ell^{-1}\sum_j\delta_{\sigma_j(A_{{\rm real},h})},
\qquad
\mu_{\Sigma,h}^{(\ell)}
=\ell^{-1}\sum_j\delta_{\sigma_j(A_{\Sigma,h})},
\\]

we have

\\[
W_1
=\ell^{-1}\sum_j
\left|\sigma_j^\downarrow(A_{{\rm real},h})
-\sigma_j^\downarrow(A_{\Sigma,h})\right|.
\\]

Because \\(W_1\\) metrizes weak convergence together with first-moment convergence, \\(W_1\to0\\) implies the same LSD whenever either sequence has a limit. This is an LSD-level comparison of equally weighted measures, not a claim that every leading spike agrees pointwise. See Reviewer WUoc Q1 and Reviewer CLzG Q3.

**Figure R.1: GPT-2 real-versus-canonical attention LSD distance.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png

**Figure R.2: ViT real-versus-canonical attention LSD distance.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png

**Conclusion:** The two spectra are compared directly, with a strong GPT-2 lengthwise decrease and small ViT layerwise distances.

## Q2: How often does spike--bulk separation hold across layers, heads, and model types?

Every score SVD is computed independently per head. **Figures R.5--R.6** pool the resulting singular values only after decomposition and show a lower bulk plus separated high-value group in every layer panel for representative GPT-2 and ViT inputs. The descriptive color split is not identified with the asymptotic \\(\log\ell\\) cutoff. **Figure R.7** adds dataset-scale GPT-2 evidence: stable rank remains below about 2 and effective rank below about 30 for \\(17\le\ell\le471\\), without substantial growth.

Structurally, \\(k\le\operatorname{rank}(S_h)\le d_h=64\\). Thus the head dimension gives an \\(\ell\\)-independent cap, while the rank proxies show a much smaller effective dimension over the tested range. The pooled histograms cover all heads but are not presented as a per-head frequency estimator. See Reviewer WUoc Q2 and Reviewer XnLy Q4.

**Figure R.5: GPT-2 actual score spectra.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.6: ViT actual score spectra.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.7: Score stable/effective ranks versus length.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png

**Conclusion:** The morphology appears throughout the displayed layers and model types, while the dataset-scale GPT-2 proxies remain finite/moderate over the tested lengths.

## Q3: Why should singular values alone control the spectrum after row-wise softmax?

1. **Empirical side.** **Figures R.1--R.2** directly test the post-softmax consequence: GPT-2 \\(W_1\\) decreases from 0.3002 to 0.0942 across all layers and from 0.3290 to 0.0748 in the last layer; every ViT layer median is below 0.104. Separately, actual-score eigen-angle diagnostics show GPT-2 left/right orientations becoming increasingly Haar-like with length:

| \\(\ell\\) band | \\(U\\): KL | \\(U\\): Kol. | \\(U\\): \\(W_1\\) | \\(V\\): KL | \\(V\\): Kol. | \\(V\\): \\(W_1\\) |
|---:|---:|---:|---:|---:|---:|---:|
| 17--64 | 0.34 | 0.04 | 0.31 | 0.34 | 0.04 | 0.31 |
| 193--256 | 0.01 | 0.01 | 0.13 | 0.01 | 0.01 | 0.13 |
| 385--471 | <0.01 | 0.01 | 0.10 | <0.01 | 0.01 | 0.10 |

ViT discrepancies are uniformly low across layers:

| Side | KL \\((\times10^{-3})\\) | Kolmogorov \\((\times10^{-3})\\) | Angular \\(W_1\\) |
|:---|---:|---:|---:|
| Left \\(U_S\\) | 2.21 ± 0.72 | 4.20 ± 0.62 | 0.08 ± 0.04 |
| Right \\(V_S\\) | 2.21 ± 0.72 | 4.21 ± 0.62 | 0.08 ± 0.04 |

These tests support the orientation premise; independence from \\(\Sigma_h\\) remains an explicit hypothesis (**Figures R.3--R.4**).

2. **Mathematical side.** Assumption 1 states \\(S\overset d=\Sigma\\). For the fixed-mask measurable map \\(F_M(X)=\operatorname{softmax}_{\rm row}(X+M)\\),

\\[
S\overset d=\Sigma
\quad\Longrightarrow\quad
F_M(S)\overset d=F_M(\Sigma).
\\]

We will state the Haar-orientation and singular-value/orientation independence hypotheses explicitly and cite Benaych-Georges and Nadakuditi (2011). See Reviewer WUoc Q1 and Reviewer XnLy Q1.

**Figure R.3:** https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

**Figure R.4:** https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

**Conclusion:** Under the stated diagonalization-in-law assumption, row-softmax preserves equality in law, and the orientation plus \\(W_1\\) experiments support this prediction on trained models.

## Q4: Can the authors report actual score and attention matrices per head?

Yes. We now report diagnostics computed from actual head-resolved matrices:

1. For every \\(S_h=Q_hK_h^\top/\sqrt{d_h}\\), we independently compute its SVD and test both score singular-vector matrices against Haar references.
2. For the same head, we independently decompose the real attention \\(\operatorname{softmax}_{\rm row}(S_h+M)\\) and spectral-canonical attention \\(\operatorname{softmax}_{\rm row}(\Sigma_h+M)\\).
3. Score histograms likewise decompose each head before pooling singular values.
4. We aggregate only scalar distances or histogram values; no head-averaged matrix is decomposed.

Primary paths are **Figures R.1--R.6** above. Full GPT-2 per-layer actual-score diagnostics:

- Left/right KL: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_kl_divergence_convergence_log.png and https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_kl_divergence_convergence_log.png
- Left/right Kolmogorov: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_kolmogorov_distance_convergence_log.png and https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_kolmogorov_distance_convergence_log.png
- Left/right angular \\(W_1\\): https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_wasserstein_distance_convergence_log.png and https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_wasserstein_distance_convergence_log.png

See Reviewer CLzG Q1/Q3 for the numerical summaries and Reviewer XnLy Q3 for the per-head theoretical context.

**Conclusion:** The added results are computed from actual per-head score and attention matrices, with aggregation only after each head has been analyzed independently.

# Response to Reviewer XnLy

We sincerely thank the reviewer for the mathematically insightful and constructive comments, which helped us sharpen the assumptions, add new results, and clarify the empirical evidence.

## Q1: When does isotropy of \\(X\\) imply the required isotropy of \\(S\\)?

1. **Mathematical side.** If \\(W_Q,W_K\\) are fixed or independent of the random input \\(X\\), a token-space rotation \\(X\mapsto OX\\) gives

\\[
Q=XW_Q\mapsto OQ,\qquad K=XW_K\mapsto OK,
\\]

and therefore

\\[
S=\frac{QK^\top}{\sqrt{d_{QK}}}\mapsto OSO^\top.
\\]

This proves the inherited simultaneous-conjugation law. We state separately the stronger modeling hypothesis that the left/right score orientations are Haar and independent of \\(\Sigma_h\\).

2. **Empirical side.** We bypass inference from \\(X\\) by directly decomposing every actual per-head score. The GPT-2 server run contains all 1,646 nontrivial WikiText-103 validation sequences, 12 layers, and 12 heads (237,024 score SVDs):

| \\(\ell\\) band | \\(U\\): KL | \\(U\\): Kol. | \\(U\\): \\(W_1\\) | \\(V\\): KL | \\(V\\): Kol. | \\(V\\): \\(W_1\\) |
|---:|---:|---:|---:|---:|---:|---:|
| 17--64 | 0.34 | 0.04 | 0.31 | 0.34 | 0.04 | 0.31 |
| 193--256 | 0.01 | 0.01 | 0.13 | 0.01 | 0.01 | 0.13 |
| 385--471 | <0.01 | 0.01 | 0.10 | <0.01 | 0.01 | 0.10 |

The ViT run contains all 3,925 Imagenette validation images, 12 layers, and 12 heads (565,200 score SVDs):

| Side | KL \\((\times10^{-3})\\) | Kolmogorov \\((\times10^{-3})\\) | Angular \\(W_1\\) |
|:---|---:|---:|---:|
| Left \\(U_S\\) | 2.21 ± 0.72 | 4.20 ± 0.62 | 0.08 ± 0.04 |
| Right \\(V_S\\) | 2.21 ± 0.72 | 4.21 ± 0.62 | 0.08 ± 0.04 |

These diagnostics test the orientation component; independence from \\(\Sigma_h\\) remains explicit. See Reviewer WUoc Q1 and Reviewer CLzG Q3.

**Figure R.3: GPT-2 actual-score Haar diagnostics.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

**Figure R.4: ViT actual-score Haar diagnostics.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

**Conclusion:** Fixed/independent query-key weights transmit token-space isotropy to a simultaneous conjugation law, while direct score experiments support the stronger Haar-orientation premise.

## Q2: Do encoder and decoder spectra degenerate at different rates?

1. **Empirical side.** ViT length cannot be swept asymptotically without changing the setup. Nevertheless, submitted-paper Figures 4--5, reproduced as **Figures R.9--R.10**, compare similar lengths: ViT-B/16 has \\(\ell=577\\), and the longest GPT-2 example has \\(\ell=471\\). The encoder bulk is already very thin and at a smaller scale from early layers, whereas the decoder retains a broader bulk. This cross-model evidence is consistent with faster encoder degeneration.

2. **Mathematical side---new results.** Under the theorem's aggregate-tail condition and \\(k=o(\ell)\\), we derive

\\[
\operatorname{srank}(A)=(k+1)(1+o(1))
\\]

for bidirectional attention. For causal attention with bounded bulk scores, we newly obtain

\\[
\operatorname{srank}(A)
=\Theta_B\!\left(k+\log\frac{\ell+1}{k+1}\right).
\\]

Thus finite \\(k\\) gives constant-order bidirectional stable rank but an additional logarithmic causal term. These theorems and proofs will be added in the revision.

**Figure R.9: ViT encoder attention spectra.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/vit_encoder_attention_spectrum_by_layer.png

**Figure R.10: GPT-2 decoder attention spectra.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/gpt2_decoder_attention_spectrum_by_layer.png

**Conclusion:** The new theorems rigorously establish constant-order bidirectional versus logarithmically larger causal stable rank, and the comparable-length cross-model spectra are consistent with this contrast.

## Q3: Are the assumptions applied per head or after aggregating heads?

The score-isotropy and spike--bulk assumptions are imposed per head. For each head,

\\[
S_h=\frac{Q_hK_h^\top}{\sqrt{d_{QK}}},
\qquad
\operatorname{rank}(S_h)\le d_{QK}\ll\ell,
\\]

so every per-head score is structurally low rank. The current theory does not explicitly propagate through head concatenation, \\(W_O\\), normalization, or residual composition; this is planned follow-up work. In the new score experiments, every \\(S_h\\) is decomposed independently and singular values are pooled only afterward (**Figures R.5--R.6**). See Reviewer WUoc Q3 and Reviewer CLzG Q4.

**Figure R.5:** https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.6:** https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Conclusion:** The score assumptions and new validation are per head; a theory for the composed multi-head block remains outside the present paper.

## Q4: What controls the number of spikes \\(k\\)?

1. **Empirical side.** Experiment 4.2/Figure 3, reproduced as **Figure R.7**, shows that across WikiText-103 lengths 17--471, stable rank stays below about 2 and effective rank below about 30, with neither increasing substantially. Thus the effective number of dominant score directions remains finite/moderate over the tested range and does not appear to scale with length or semantic-token count.

2. **Mathematical side.** Every head obeys

\\[
k\le\operatorname{rank}(S_h)\le d_h=64.
\\]

Fixed head dimension therefore provides an \\(\ell\\)-independent cap; within it, realized \\(k\\) depends on the input, layer/head, and learned query/key projections. Exact spike-number estimation is a longstanding model-selection problem; see Zhidong Bai, Shurong Zheng, and Jianfeng Yao, *Large Sample Covariance Matrices and High-Dimensional Data Analysis* (Cambridge University Press, 2015). Our results require finite/sublinear \\(k\\), not an exact estimator. See Reviewer WUoc Q2 and Reviewer CLzG Q2.

**Figure R.7: Score stable/effective ranks versus length.**

https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png

**Conclusion:** Head dimension gives an \\(\ell\\)-independent cap, and the observed rank proxies support finite/moderate \\(k\\) without claiming to estimate its exact threshold-defined value.
