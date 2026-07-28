# Response to Reviewer WUoc

We sincerely thank the reviewer for the careful reading and constructive remarks, which prompted us to strengthen the score-level experiments and clarify the scope of the theory.

## Q1: What justifies replacing the real score matrix by its spectral representative?

1. **Empirical side.** We added a direct, head-resolved comparison between the real attention spectrum and the spectral-canonical attention spectrum. For every sample, layer, and head, we reconstruct

   \[
   S_h=\frac{Q_hK_h^\top}{\sqrt{d_h}},\qquad
   A_{{\rm real},h}=\operatorname{softmax}_{\rm row}(S_h+M),\qquad
   A_{\Sigma,h}=\operatorname{softmax}_{\rm row}(\Sigma_h+M),
   \]

   where \(\Sigma_h=\operatorname{diag}(\operatorname{svdvals}(S_h))\), with singular values in descending order, and \(M\) is the GPT-2 causal mask (\(M=0\) for ViT). We compute both attention SVDs independently within each head. No head-averaged matrix is decomposed.

   For a length-\(\ell\) head, the two empirical singular-value measures are

   \[
   \mu_{{\rm real},h}^{(\ell)}
   =\frac1\ell\sum_{j=1}^{\ell}
   \delta_{\sigma_j(A_{{\rm real},h})},
   \qquad
   \mu_{\Sigma,h}^{(\ell)}
   =\frac1\ell\sum_{j=1}^{\ell}
   \delta_{\sigma_j(A_{\Sigma,h})}.
   \]

   Here an LSD is precisely a weak limit of these empirical measures; for example, \(\mu_{{\rm real},h}^{(\ell)}\Rightarrow\mu\) means that \(\int\varphi\,d\mu_{{\rm real},h}^{(\ell)}\to\int\varphi\,d\mu\) for every bounded continuous test function \(\varphi\).

   We use \(W_1\) because it is the natural bin-free metric for the weak-LSD statement. Since both empirical measures have \(\ell\) equally weighted atoms on \(\mathbb R_+\), their one-dimensional Wasserstein distance is exactly

   \[
   W_1\!\left(\mu_{{\rm real},h}^{(\ell)},
   \mu_{\Sigma,h}^{(\ell)}\right)
   =\frac1\ell\sum_{j=1}^{\ell}
   \left|
   \sigma_j^\downarrow(A_{{\rm real},h})
   -\sigma_j^\downarrow(A_{\Sigma,h})
   \right|.
   \]

   On \(\mathcal P_1(\mathbb R_+)\), \(W_1\) metrizes weak convergence together with convergence of the first moment. In particular, the Kantorovich--Rubinstein inequality gives, for every Lipschitz test function \(\varphi\),

   \[
   \left|
   \int\varphi\,d\mu_{{\rm real},h}^{(\ell)}
   -\int\varphi\,d\mu_{\Sigma,h}^{(\ell)}
   \right|
   \le
   \operatorname{Lip}(\varphi)\,
   W_1\!\left(\mu_{{\rm real},h}^{(\ell)},
   \mu_{\Sigma,h}^{(\ell)}\right).
   \]

   Therefore, if \(W_1\to0\) as \(\ell\to\infty\), the two empirical spectral measures are asymptotically equivalent in the weak-LSD sense: whenever either sequence has an LSD, the other has the same LSD.

   For GPT-2, we use 12 WikiText-103 validation documents, seven nested sequence lengths, all 12 layers, and all 12 heads, giving 12,096 matched comparisons. **Figure R.1** reports the all-layer and last-layer medians with IQRs. To make the layer-wise convergence explicit, the table below reports mean \(W_1\pm\) standard deviation for the last five layers; every cell summarizes the same 12 documents \(\times\) 12 heads (144 matched per-head distances).

   | GPT-2 layer | \(\ell=64\) | \(128\) | \(256\) | \(384\) | \(512\) | \(768\) | \(1024\) |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 8 | 0.3906 ± 0.0987 | 0.3119 ± 0.0686 | 0.2222 ± 0.0257 | 0.1665 ± 0.0170 | 0.1383 ± 0.0226 | 0.1145 ± 0.0345 | 0.1054 ± 0.0416 |
   | 9 | 0.3352 ± 0.0799 | 0.2756 ± 0.0569 | 0.2093 ± 0.0247 | 0.1630 ± 0.0119 | 0.1363 ± 0.0149 | 0.1123 ± 0.0270 | 0.1026 ± 0.0350 |
   | 10 | 0.3699 ± 0.0753 | 0.3002 ± 0.0514 | 0.2184 ± 0.0213 | 0.1633 ± 0.0135 | 0.1319 ± 0.0190 | 0.1036 ± 0.0294 | 0.0913 ± 0.0357 |
   | 11 | 0.3575 ± 0.0551 | 0.2843 ± 0.0391 | 0.2070 ± 0.0190 | 0.1553 ± 0.0107 | 0.1242 ± 0.0120 | 0.0958 ± 0.0186 | 0.0833 ± 0.0230 |
   | 12 | 0.3088 ± 0.0784 | 0.2556 ± 0.0479 | 0.2006 ± 0.0211 | 0.1588 ± 0.0089 | 0.1282 ± 0.0066 | 0.0972 ± 0.0137 | 0.0829 ± 0.0191 |

   Every one of the last five layers shows a strong decrease with sequence length, consistent with convergence: by \(\ell=1024\), the mean distances lie between \(0.0829\) and \(0.1054\). In **Figure R.1**, the all-layer median decreases from \(0.3002\) to \(0.0942\), and the last-layer median decreases from \(0.3290\) to \(0.0748\). This is the requested quantitative evidence that the spectral discrepancy decreases in the long-sequence regime.

   For ViT-B/16, we use 20 class-balanced Imagenette images, all 12 layers, and all 12 heads, giving 2,880 matched comparisons. Each layer boxplot contains 240 image-head values.

   | ViT layer | Median \(W_1\) | IQR |
   |---:|---:|---:|
   | 1 | 0.0431 | [0.0194, 0.1019] |
   | 2 | 0.0726 | [0.0414, 0.1312] |
   | 3 | 0.1037 | [0.0457, 0.4490] |
   | 4 | 0.0897 | [0.0729, 0.1329] |
   | 5 | 0.0746 | [0.0628, 0.0883] |
   | 6 | 0.0668 | [0.0582, 0.0724] |
   | 7 | 0.0580 | [0.0503, 0.0633] |
   | 8 | 0.0567 | [0.0482, 0.0626] |
   | 9 | 0.0519 | [0.0443, 0.0595] |
   | 10 | 0.0403 | [0.0344, 0.0453] |
   | 11 | 0.0429 | [0.0359, 0.0482] |
   | 12 | 0.0367 | [0.0309, 0.0438] |

   All layer medians are below \(0.104\), and 10 of 12 are below \(0.075\). The IQRs in **Figure R.2** also display the layer/head heterogeneity rather than hiding it through a single average.

   We additionally test both singular-vector matrices of the actual score \(S_h=U_{S_h}\Sigma_hV_{S_h}^{\top}\), rather than only the hidden-state input \(X\). For each matrix, we compare the empirical eigen-angle laws of \(U_{S_h}\) and \(V_{S_h}\) with the circular-uniform Haar reference using KL divergence, Kolmogorov distance, and angular \(W_1\); all three discrepancies are minimized at zero.

   The GPT-2 server run covers all 1,646 nontrivial WikiText-103 validation sequences with \(17\le\ell\le471\), all 12 layers, and all 12 heads, for 237,024 actual-score SVDs. Each entry below is the mean \(\pm\) standard deviation over all head-layer observations in the indicated length band; each sequence contributes 144 such observations.

   | \(\ell\) band | \(n_{\rm seq}\) | \(U_{S}\): KL | \(U_{S}\): Kol. | \(U_{S}\): angular \(W_1\) | \(V_{S}\): KL | \(V_{S}\): Kol. | \(V_{S}\): angular \(W_1\) |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 17--64 | 254 | 0.3393 \(\pm\) 0.2005 | 0.0406 \(\pm\) 0.0145 | 0.3090 \(\pm\) 0.1435 | 0.3387 \(\pm\) 0.2001 | 0.0401 \(\pm\) 0.0136 | 0.3098 \(\pm\) 0.1430 |
   | 65--128 | 512 | 0.0655 \(\pm\) 0.0374 | 0.0202 \(\pm\) 0.0051 | 0.2040 \(\pm\) 0.0901 | 0.0655 \(\pm\) 0.0373 | 0.0203 \(\pm\) 0.0051 | 0.2050 \(\pm\) 0.0907 |
   | 129--192 | 464 | 0.0244 \(\pm\) 0.0096 | 0.0131 \(\pm\) 0.0026 | 0.1590 \(\pm\) 0.0692 | 0.0245 \(\pm\) 0.0096 | 0.0132 \(\pm\) 0.0026 | 0.1585 \(\pm\) 0.0688 |
   | 193--256 | 275 | 0.0131 \(\pm\) 0.0047 | 0.0098 \(\pm\) 0.0018 | 0.1338 \(\pm\) 0.0582 | 0.0131 \(\pm\) 0.0047 | 0.0099 \(\pm\) 0.0018 | 0.1335 \(\pm\) 0.0576 |
   | 257--320 | 100 | 0.0083 \(\pm\) 0.0029 | 0.0079 \(\pm\) 0.0013 | 0.1183 \(\pm\) 0.0512 | 0.0083 \(\pm\) 0.0028 | 0.0080 \(\pm\) 0.0014 | 0.1182 \(\pm\) 0.0513 |
   | 321--384 | 33 | 0.0055 \(\pm\) 0.0019 | 0.0065 \(\pm\) 0.0010 | 0.1054 \(\pm\) 0.0454 | 0.0055 \(\pm\) 0.0019 | 0.0066 \(\pm\) 0.0011 | 0.1061 \(\pm\) 0.0462 |
   | 385--471 | 8 | 0.0039 \(\pm\) 0.0013 | 0.0055 \(\pm\) 0.0009 | 0.0950 \(\pm\) 0.0421 | 0.0039 \(\pm\) 0.0013 | 0.0055 \(\pm\) 0.0009 | 0.0964 \(\pm\) 0.0418 |

   Thus every metric decreases monotonically across these length bands, with nearly identical behavior on the left and right. The ViT-B/16 run covers all 3,925 Imagenette validation images, all 12 layers, and all 12 heads, for 565,200 actual-score SVDs. The next table gives the pooled mean \(\pm\) standard deviation; parentheses give the minimum and maximum of the 12 layer-wise means, each based on 47,100 image-head observations.

   | ViT-B/16 side | KL | Kolmogorov | Angular \(W_1\) |
   |:---|---:|---:|---:|
   | Left \(U_{S}\) | 0.002205 \(\pm\) 0.000715 (0.002199--0.002209) | 0.004203 \(\pm\) 0.000617 (0.004197--0.004209) | 0.082197 \(\pm\) 0.035621 (0.081902--0.082473) |
   | Right \(V_{S}\) | 0.002206 \(\pm\) 0.000717 (0.002202--0.002215) | 0.004205 \(\pm\) 0.000618 (0.004185--0.004214) | 0.082226 \(\pm\) 0.035588 (0.082044--0.082480) |

   The very narrow ranges of the ViT layer means show that these low discrepancies hold throughout the network rather than arising from a favorable subset of layers (see **Figures R.3--R.4**). These eigen-angle diagnostics assess the Haar-orientation component; independence between \(\Sigma_h\) and the singular-vector matrices remains an explicit modeling hypothesis.

2. **Mathematical side.** Assumption 1 is the diagonalization-in-law hypothesis

   \[
   S\overset d=\Sigma,
   \qquad
   \Sigma=\operatorname{diag}(\operatorname{svdvals}(S)).
   \]

   It follows the standard invariant-ensemble basis-reduction technique used, for example, by Benaych-Georges and Nadakuditi, *The eigenvalues and eigenvectors of finite, low rank perturbations of large random matrices*, Adv. Math. 227(1), 2011. The corresponding score-orbit representation is

   \[
   S\mid\Sigma\overset d=U\Sigma V^\top,
   \qquad U,V\ \text{Haar and independent of }\Sigma.
   \]

   We will state the independence hypothesis explicitly. For a fixed mask \(M\), define the deterministic measurable map \(F_M(X)=\operatorname{softmax}_{\rm row}(X+M)\). Equality in distribution is preserved under \(F_M\), so Assumption 1 gives

   \[
   \operatorname{softmax}_{\rm row}(S+M)
   \overset d=
   \operatorname{softmax}_{\rm row}(\Sigma+M).
   \]

   The new \(W_1\) experiment directly tests and supports this predicted post-softmax weak-LSD consequence on pretrained models.

**Cross-reference.** The direct spectral comparison is also summarized in Reviewer CLzG Q1, while Reviewer CLzG Q3 and Reviewer XnLy Q1 discuss the diagonalization-in-law and score-isotropy hypotheses from complementary angles.

**Figure R.1.** GPT-2 \(W_1\) distance between the empirical singular-value measures of real and spectral-canonical attention, aggregated over all layers and for the last layer.

![Figure R.1: GPT-2 real-versus-canonical attention LSD distance](https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png)

Figure R.1 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png

**Figure R.2.** ViT-B/16 per-layer \(W_1\) distance between the empirical singular-value measures of real and spectral-canonical attention.

![Figure R.2: ViT real-versus-canonical attention LSD distance by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png)

Figure R.2 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png

**Figure R.3.** GPT-2 actual-score left/right singular-vector Haar diagnostics versus sequence length.

![Figure R.3: GPT-2 actual-score singular-vector Haar diagnostics](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png)

Figure R.3 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

Figure R.3 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.svg

**Figure R.4.** ViT-B/16 actual-score left/right singular-vector Haar diagnostics across layers.

![Figure R.4: ViT actual-score singular-vector Haar diagnostics](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png)

Figure R.4 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

Figure R.4 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.svg

**Conclusion:** The spectral replacement is justified conditionally by the stated diagonalization-in-law hypothesis and supported in practice by the decreasing real-versus-canonical \(W_1\) discrepancy and direct score-level Haar diagnostics, especially in the long-sequence regime.

## Q2: Does the spike-bulk separation occur for real score matrices?

1. **Empirical side.** We added direct log-scale singular-value histograms of the actual score matrices (see **Figures R.5--R.6**). For every head, we independently compute the SVD of \(S_h\) and pool the nonzero singular values only afterward within each layer. The figures cover all 12 layers and all 12 heads of GPT-2 on a 1024-token WikiText sample and ViT-B/16 on a 577-token Imagenette image. The high-value group is colored after a layer-specific empirical eigengap, making the separation from the lower spectral bulk visible across the full dynamic range.

   This direct evidence is complemented by Experiment 4.2 and Figure 3 of the submitted paper, reproduced here as **Figure R.7**. For sequence lengths from \(17\) to \(471\), the score-matrix stable rank remains below approximately \(2\), while its effective rank remains below approximately \(30\), with neither increasing substantially with \(\ell\). Hence the effective number of dominant score directions stays finite and moderate over the tested range.

2. **Mathematical interpretation.** The log-scale histograms provide direct evidence for the empirical bulk-spike morphology, while the bounded rank proxies support the finite/sublinear-spike regime used in the theory. We deliberately do not impose the paper's asymptotic \(\log\ell\) neighborhood as a finite-sample cutoff, because it does not coincide exactly with the visible gap at these lengths. The dashed threshold is instead descriptive: for each layer, we median-aggregate the ordered per-head spectra, locate the largest adjacent multiplicative gap in the upper half of that scree curve, and use its geometric midpoint. This empirical split makes the finite-sample high-value group visible but is not claimed to equal the asymptotic logarithmic separation. The observed structure is also consistent with the established use of spiked random-matrix models for learned real-data representations; see Seddik et al., *Random Matrix Theory Proves that Deep Learning Representations of GAN-data Behave as Gaussian Mixtures*, ICML 2020.

**Cross-reference.** Reviewer CLzG Q2 asks the same practical question across layers, heads, and model types, while Reviewer XnLy Q4 discusses what controls the finite/moderate number of dominant directions.

**Figure R.5.** GPT-2 actual pre-softmax score singular values by layer, with the descriptive finite-sample high-value group highlighted.

![Figure R.5: GPT-2 actual score singular values by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png)

Figure R.5 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

Figure R.5 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.svg

**Figure R.6.** ViT-B/16 actual pre-softmax score singular values by layer, with the descriptive finite-sample high-value group highlighted.

![Figure R.6: ViT actual score singular values by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png)

Figure R.6 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

Figure R.6 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.svg

**Figure R.7.** Stable and effective ranks of the WikiText-103 score matrices versus sequence length (submitted-paper Figure 3), with medians and IQRs.

![Figure R.7: Score stable and effective ranks versus sequence length](https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png)

Figure R.7 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png

Figure R.7 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.svg

Related spiked-model reference:
https://proceedings.mlr.press/v119/seddik20a/seddik20a.pdf

**Conclusion:** For the representative GPT-2 and ViT inputs, every layer panel exhibits a practical bulk-spike morphology after independent per-head SVDs, while **Figure R.7** shows that the effective spectral dimension remains finite/moderate over the tested GPT-2 lengths; the exact \(\log\ell\) separation remains an asymptotic assumption rather than an imposed finite-sample cutoff.

## Q3: Can the framework include MHA, LayerNorm, and other Transformer components?

1. **Current scope.** The present paper isolates the attention operator and does not explicitly propagate the spectral law through head concatenation, the output projection, residual connections, and normalization. Because the analyzed scores are extracted from intact pretrained networks, they already reflect the upstream normalization and residual dynamics that determine the input to attention; the unmodeled step is propagation through the full multi-head output/residual/normalization block.

2. **Extension.** We agree that this is an important and ambitious direction. We are developing follow-up work that treats multi-head composition and normalization explicitly, and we will state this scope and extension clearly in the revision.

**Cross-reference.** Reviewer XnLy Q3 gives the per-head formulation and structural low-rank bound, while Reviewer CLzG Q4 describes the new head-resolved diagnostics.

**Conclusion:** The current framework rigorously covers the isolated attention operator, whereas MHA composition, LayerNorm, residual connections, and output projections are important extensions beyond the present paper's scope.

# Response to Reviewer CLzG

We thank the reviewer for the thoughtful and constructive questions, which motivated direct spectral comparisons and clearer head-resolved evidence.

## Q1: Can the spectra of \(\operatorname{softmax}(S)\) and \(\operatorname{softmax}(\Sigma)\) be compared directly?

1. **Empirical side.** Yes. We added exactly this experiment on pretrained GPT-2 and ViT-B/16. For every sample, layer, and head, the SVDs of \(A_{{\rm real},h}=\operatorname{softmax}_{\rm row}(S_h+M)\) and \(A_{\Sigma,h}=\operatorname{softmax}_{\rm row}(\Sigma_h+M)\) are computed independently. We then calculate their bin-free \(W_1\) distance before any aggregation.

   GPT-2 uses 12 WikiText-103 documents, seven lengths, 12 layers, and 12 heads, giving 12,096 matched comparisons. **Figure R.1** reports the all-layer and last-layer medians with IQRs; the table makes the decreasing trend in the last five layers explicit. Each entry is mean \(W_1\pm\) standard deviation over 12 documents \(\times\) 12 heads.

   | GPT-2 layer | \(\ell=64\) | \(128\) | \(256\) | \(384\) | \(512\) | \(768\) | \(1024\) |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 8 | 0.3906 ± 0.0987 | 0.3119 ± 0.0686 | 0.2222 ± 0.0257 | 0.1665 ± 0.0170 | 0.1383 ± 0.0226 | 0.1145 ± 0.0345 | 0.1054 ± 0.0416 |
   | 9 | 0.3352 ± 0.0799 | 0.2756 ± 0.0569 | 0.2093 ± 0.0247 | 0.1630 ± 0.0119 | 0.1363 ± 0.0149 | 0.1123 ± 0.0270 | 0.1026 ± 0.0350 |
   | 10 | 0.3699 ± 0.0753 | 0.3002 ± 0.0514 | 0.2184 ± 0.0213 | 0.1633 ± 0.0135 | 0.1319 ± 0.0190 | 0.1036 ± 0.0294 | 0.0913 ± 0.0357 |
   | 11 | 0.3575 ± 0.0551 | 0.2843 ± 0.0391 | 0.2070 ± 0.0190 | 0.1553 ± 0.0107 | 0.1242 ± 0.0120 | 0.0958 ± 0.0186 | 0.0833 ± 0.0230 |
   | 12 | 0.3088 ± 0.0784 | 0.2556 ± 0.0479 | 0.2006 ± 0.0211 | 0.1588 ± 0.0089 | 0.1282 ± 0.0066 | 0.0972 ± 0.0137 | 0.0829 ± 0.0191 |

   All five layers show a strong decrease consistent with convergence; their mean \(W_1\) values are only \(0.0829\)--\(0.1054\) at \(\ell=1024\). In **Figure R.1**, the all-layer median decreases from \(0.3002\) to \(0.0942\), and the last-layer median decreases from \(0.3290\) to \(0.0748\).

   ViT-B/16 uses 20 class-balanced Imagenette images, 12 layers, and 12 heads, giving 2,880 matched comparisons. Each layer contains 240 image-head values.

   | ViT layer | Median \(W_1\) | IQR |
   |---:|---:|---:|
   | 1 | 0.0431 | [0.0194, 0.1019] |
   | 2 | 0.0726 | [0.0414, 0.1312] |
   | 3 | 0.1037 | [0.0457, 0.4490] |
   | 4 | 0.0897 | [0.0729, 0.1329] |
   | 5 | 0.0746 | [0.0628, 0.0883] |
   | 6 | 0.0668 | [0.0582, 0.0724] |
   | 7 | 0.0580 | [0.0503, 0.0633] |
   | 8 | 0.0567 | [0.0482, 0.0626] |
   | 9 | 0.0519 | [0.0443, 0.0595] |
   | 10 | 0.0403 | [0.0344, 0.0453] |
   | 11 | 0.0429 | [0.0359, 0.0482] |
   | 12 | 0.0367 | [0.0309, 0.0438] |

   All ViT layer medians are below \(0.104\), with 10 of 12 below \(0.075\). The table and **Figure R.2** therefore directly show both the agreement and its layer/head heterogeneity.

2. **Mathematical connection.** The experiment compares the two empirical singular-value measures

   \[
   \mu_{{\rm real},h}^{(\ell)}=\frac1\ell\sum_{j=1}^{\ell}\delta_{\sigma_j(A_{{\rm real},h})},
   \qquad
   \mu_{\Sigma,h}^{(\ell)}=\frac1\ell\sum_{j=1}^{\ell}\delta_{\sigma_j(A_{\Sigma,h})}.
   \]

   Their LSDs are the weak limits of these empirical measures as \(\ell\to\infty\).

   With the singular values sorted in descending order, the equal-mass one-dimensional transport formula is

   \[
   W_1(\mu_{{\rm real},h}^{(\ell)},\mu_{\Sigma,h}^{(\ell)})
   =\frac1\ell\sum_{j=1}^{\ell}
   \left|\sigma_j^\downarrow(A_{{\rm real},h})
   -\sigma_j^\downarrow(A_{\Sigma,h})\right|.
   \]

   We use \(W_1\) because it is a bin-free metric for the weak-LSD question. On \(\mathcal P_1(\mathbb R_+)\), it metrizes weak convergence together with convergence of first moments, and the Kantorovich--Rubinstein inequality controls the difference of every Lipschitz spectral statistic by \(W_1\). Consequently,

   \[
   W_1(\mu_{{\rm real},h}^{(\ell)},\mu_{\Sigma,h}^{(\ell)})\to0
   \quad\Longrightarrow\quad
   \mu_{{\rm real},h}^{(\ell)}
   \ \text{and}\
   \mu_{\Sigma,h}^{(\ell)}
   \ \text{are asymptotically equivalent in the weak topology}.
   \]

   Hence, whenever either sequence has an LSD, the other converges weakly to the same LSD. This is an LSD-level comparison of equally weighted empirical spectral measures; it does not assert pointwise equality of every leading spike. The experiment directly tests the asymptotic weak-LSD prediction of the canonical model rather than using hidden-state isotropy or rank proxies as an indirect substitute.

**Cross-reference.** Reviewer WUoc Q1 gives the full diagonalization-in-law justification, and Reviewer CLzG Q3 explains why the same assumption controls the spectrum after row-wise softmax.

**Figure R.1.** GPT-2 \(W_1\) distance between the empirical singular-value measures of real and spectral-canonical attention, aggregated over all layers and for the last layer.

![Figure R.1: GPT-2 real-versus-canonical attention LSD distance](https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png)

Figure R.1 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png

**Figure R.2.** ViT-B/16 per-layer \(W_1\) distance between the empirical singular-value measures of real and spectral-canonical attention.

![Figure R.2: ViT real-versus-canonical attention LSD distance by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png)

Figure R.2 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png

**Conclusion:** Yes—the spectra can be compared directly, and their measured \(W_1\) discrepancy decreases strongly with GPT-2 sequence length and remains small across ViT layers.

## Q2: How often does the spike-bulk separation hold across layers, heads, and model types?

1. **Empirical side.** We added log-scale histograms of the actual score singular values for both model types (see **Figures R.5--R.6**). Each of the 12 subfigures corresponds to one layer; within each layer, the SVD is computed separately for all 12 heads before pooling the nonzero singular values. The GPT-2 figure uses a 1024-token WikiText input, and the ViT-B/16 figure uses a 577-token Imagenette input. The empirical high-value group is colored in each panel, showing the lower bulk and larger upper group while retaining the full dynamic range.

   This is supported at dataset scale by Experiment 4.2 and Figure 3 of the submitted paper, reproduced as **Figure R.7**: over WikiText-103 validation sequences with \(17\le\ell\le471\), stable rank stays below approximately \(2\) and effective rank below approximately \(30\), without substantial growth with \(\ell\). The direct histograms show the morphology, while the dataset-level rank proxies show that the number of dominant directions remains finite/moderate over the tested lengths.

2. **Mathematical interpretation.** The figures do not use the asymptotic \(\log\ell\) neighborhood as a finite-length cutoff. The colored log-scale view uses only a descriptive, layer-specific empirical eigengap: the geometric midpoint across the largest adjacent ratio in the upper half of the median ordered per-head spectrum. It is included to make the observed finite-sample separation readable and is not identified with the exact \(\log\ell\) rate, which remains the asymptotic statement of the assumption. The experiment therefore establishes the practical bulk-spike morphology on actual trained-model scores without claiming that its finite threshold is already the asymptotic one.

**Cross-reference.** Reviewer WUoc Q2 gives the same empirical and mathematical evidence in a complementary formulation, while Reviewer XnLy Q4 explains the structural and empirical controls on \(k\).

**Figure R.5.** GPT-2 actual pre-softmax score singular values by layer, with the descriptive finite-sample high-value group highlighted.

![Figure R.5: GPT-2 actual score singular values by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png)

Figure R.5 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

Figure R.5 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.svg

**Figure R.6.** ViT-B/16 actual pre-softmax score singular values by layer, with the descriptive finite-sample high-value group highlighted.

![Figure R.6: ViT actual score singular values by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png)

Figure R.6 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

Figure R.6 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.svg

**Figure R.7.** Stable and effective ranks of the WikiText-103 score matrices versus sequence length (submitted-paper Figure 3), with medians and IQRs.

![Figure R.7: Score stable and effective ranks versus sequence length](https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png)

Figure R.7 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png

Figure R.7 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.svg

**Conclusion:** For the representative GPT-2 and ViT inputs, every layer panel shows a separated high-value group after independent per-head SVDs, while the dataset-scale GPT-2 rank proxies remain finite/moderate over the tested lengths; the pooled histograms are not presented as a per-head frequency estimator.

## Q3: Why should singular values alone control the spectrum after row-wise softmax?

1. **Empirical side.** We now test this question directly rather than relying only on motivation. In **Figures R.1--R.2**, the per-head spectral distance decreases from \(0.3002\) to \(0.0942\) after pooling all GPT-2 layers and from \(0.3290\) to \(0.0748\) in the last layer as \(\ell\) grows from \(64\) to \(1024\). For ViT, all layer medians are below \(0.104\).

   We also directly decompose every actual score as \(S_h=U_{S_h}\Sigma_hV_{S_h}^{\top}\) and compare the eigen-angle laws of both \(U_{S_h}\) and \(V_{S_h}\) with the circular-uniform Haar reference. The GPT-2 run contains 1,646 WikiText-103 validation sequences, 12 layers, and 12 heads (237,024 per-head score SVDs). The table reports mean \(\pm\) standard deviation over the head-layer observations in each length band; KL, Kolmogorov, and angular \(W_1\) are all distances to the Haar reference, so smaller is closer.

   | \(\ell\) band | \(n_{\rm seq}\) | \(U_{S}\): KL | \(U_{S}\): Kol. | \(U_{S}\): angular \(W_1\) | \(V_{S}\): KL | \(V_{S}\): Kol. | \(V_{S}\): angular \(W_1\) |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 17--64 | 254 | 0.3393 \(\pm\) 0.2005 | 0.0406 \(\pm\) 0.0145 | 0.3090 \(\pm\) 0.1435 | 0.3387 \(\pm\) 0.2001 | 0.0401 \(\pm\) 0.0136 | 0.3098 \(\pm\) 0.1430 |
   | 65--128 | 512 | 0.0655 \(\pm\) 0.0374 | 0.0202 \(\pm\) 0.0051 | 0.2040 \(\pm\) 0.0901 | 0.0655 \(\pm\) 0.0373 | 0.0203 \(\pm\) 0.0051 | 0.2050 \(\pm\) 0.0907 |
   | 129--192 | 464 | 0.0244 \(\pm\) 0.0096 | 0.0131 \(\pm\) 0.0026 | 0.1590 \(\pm\) 0.0692 | 0.0245 \(\pm\) 0.0096 | 0.0132 \(\pm\) 0.0026 | 0.1585 \(\pm\) 0.0688 |
   | 193--256 | 275 | 0.0131 \(\pm\) 0.0047 | 0.0098 \(\pm\) 0.0018 | 0.1338 \(\pm\) 0.0582 | 0.0131 \(\pm\) 0.0047 | 0.0099 \(\pm\) 0.0018 | 0.1335 \(\pm\) 0.0576 |
   | 257--320 | 100 | 0.0083 \(\pm\) 0.0029 | 0.0079 \(\pm\) 0.0013 | 0.1183 \(\pm\) 0.0512 | 0.0083 \(\pm\) 0.0028 | 0.0080 \(\pm\) 0.0014 | 0.1182 \(\pm\) 0.0513 |
   | 321--384 | 33 | 0.0055 \(\pm\) 0.0019 | 0.0065 \(\pm\) 0.0010 | 0.1054 \(\pm\) 0.0454 | 0.0055 \(\pm\) 0.0019 | 0.0066 \(\pm\) 0.0011 | 0.1061 \(\pm\) 0.0462 |
   | 385--471 | 8 | 0.0039 \(\pm\) 0.0013 | 0.0055 \(\pm\) 0.0009 | 0.0950 \(\pm\) 0.0421 | 0.0039 \(\pm\) 0.0013 | 0.0055 \(\pm\) 0.0009 | 0.0964 \(\pm\) 0.0418 |

   All six discrepancies decrease monotonically across the length bands. For ViT-B/16, the run contains all 3,925 Imagenette validation images, 12 layers, and 12 heads (565,200 per-head score SVDs). Entries below are pooled mean \(\pm\) standard deviation, with the min--max range of the 12 layer-wise means in parentheses.

   | ViT-B/16 side | KL | Kolmogorov | Angular \(W_1\) |
   |:---|---:|---:|---:|
   | Left \(U_{S}\) | 0.002205 \(\pm\) 0.000715 (0.002199--0.002209) | 0.004203 \(\pm\) 0.000617 (0.004197--0.004209) | 0.082197 \(\pm\) 0.035621 (0.081902--0.082473) |
   | Right \(V_{S}\) | 0.002206 \(\pm\) 0.000717 (0.002202--0.002215) | 0.004205 \(\pm\) 0.000618 (0.004185--0.004214) | 0.082226 \(\pm\) 0.035588 (0.082044--0.082480) |

   Thus the left and right score singular-vector matrices behave almost identically, GPT-2 becomes increasingly Haar-like with sequence length, and the ViT discrepancies are uniformly low across layers (see **Figures R.3--R.4**). These angle-law diagnostics support the orientation premise; independence from \(\Sigma_h\) remains an explicit hypothesis.

2. **Mathematical side.** The diagonalization-in-law hypothesis is

   \[
   S\overset d=\Sigma.
   \]

   For the deterministic fixed-mask map \(F_M(X)=\operatorname{softmax}_{\rm row}(X+M)\), equality in distribution is preserved. Therefore the stated hypothesis gives

   \[
   \operatorname{softmax}_{\rm row}(S+M)
   \overset d=
   \operatorname{softmax}_{\rm row}(\Sigma+M).
   \]

   We will make this step explicit and cite the standard invariant-ensemble reduction of Benaych-Georges and Nadakuditi (2011), together with the assumed independence of singular values and singular vectors. The high-dimensional motivation is also visible empirically: token-space orientations become increasingly Haar-like with sequence length, while \(\operatorname{rank}(S_h)\le d_{QK}\ll\ell\). The model therefore retains the dominant singular scales while discarding increasingly generic token-space directions.

**Cross-reference.** Reviewer WUoc Q1 gives the complete spectral-replacement argument and experiment, while Reviewer XnLy Q1 explains the condition under which token-space isotropy of \(X\) passes to a rotation law for \(S\).

**Figure R.1.** GPT-2 \(W_1\) distance between the empirical singular-value measures of real and spectral-canonical attention, aggregated over all layers and for the last layer.

![Figure R.1: GPT-2 real-versus-canonical attention LSD distance](https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png)

Figure R.1 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png

**Figure R.2.** ViT-B/16 per-layer \(W_1\) distance between the empirical singular-value measures of real and spectral-canonical attention.

![Figure R.2: ViT real-versus-canonical attention LSD distance by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png)

Figure R.2 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png

**Figure R.3.** GPT-2 actual-score left/right singular-vector Haar diagnostics versus sequence length.

![Figure R.3: GPT-2 actual-score singular-vector Haar diagnostics](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png)

Figure R.3 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

Figure R.3 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.svg

**Figure R.4.** ViT-B/16 actual-score left/right singular-vector Haar diagnostics across layers.

![Figure R.4: ViT actual-score singular-vector Haar diagnostics](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png)

Figure R.4 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

Figure R.4 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.svg

**Conclusion:** Singular values control the post-softmax spectrum under the stated diagonalization-in-law hypothesis—Haar left/right singular vectors independent of \(\Sigma\)—and the direct score-level Haar tests together with the decreasing real-versus-canonical \(W_1\) discrepancy show that this prediction is approximately realized in the tested GPT-2 and ViT models.

## Q4: Can the authors report actual score and attention matrices per head?

1. **Diagnostics on actual score matrices.** For every layer and head, we reconstruct \(S_h=Q_hK_h^\top/\sqrt{d_h}\), compute its SVD independently, and test both \(U_{S_h}\) and \(V_{S_h}\) against the Haar reference. The numerical tables and aggregate figures are reported immediately in Q3 above; **Figure R.8** additionally provides every GPT-2 layer separately for both sides and all three metrics.

2. **Diagnostics on actual attention matrices.** For the same head, we independently construct and decompose the real attention \(\operatorname{softmax}_{\rm row}(S_h+M)\) and its spectral-canonical counterpart \(\operatorname{softmax}_{\rm row}(\Sigma_h+M)\). **Figures R.1--R.2** summarize 12,096 GPT-2 and 2,880 ViT matched per-head comparisons.

3. **Actual score spectra.** **Figures R.5--R.6** likewise compute the score SVD independently within each head and pool singular values only afterward within a layer.

4. **Aggregation.** All these new experiments are head-resolved. We aggregate only per-head scalar distances or histogram values; no head-averaged matrix is decomposed.

**Cross-reference.** Reviewer CLzG Q1 contains the direct real-versus-canonical attention comparison, Reviewer CLzG Q3 contains the score-orientation diagnostics, and Reviewer XnLy Q3 explains why the score assumptions are formulated per head.

**Primary figure paths.** Figure R.1: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/gpt2_w1_all_and_last_layer_vs_seq_len.png

Figure R.2: https://anonymous.4open.science/r/attention-4ECD/rebuttal/softmax_LSD_distance/vit_w1_by_layer.png

Figure R.3: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

Figure R.4: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

Figure R.5: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

Figure R.6: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.8(a).** GPT-2 left singular-vector KL diagnostic by layer.

![Figure R.8(a): GPT-2 left singular-vector KL diagnostic by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_kl_divergence_convergence_log.png)

Figure R.8(a) path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_kl_divergence_convergence_log.png

**Figure R.8(b).** GPT-2 right singular-vector KL diagnostic by layer.

![Figure R.8(b): GPT-2 right singular-vector KL diagnostic by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_kl_divergence_convergence_log.png)

Figure R.8(b) path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_kl_divergence_convergence_log.png

**Figure R.8(c).** GPT-2 left singular-vector Kolmogorov diagnostic by layer.

![Figure R.8(c): GPT-2 left singular-vector Kolmogorov diagnostic by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_kolmogorov_distance_convergence_log.png)

Figure R.8(c) path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_kolmogorov_distance_convergence_log.png

**Figure R.8(d).** GPT-2 right singular-vector Kolmogorov diagnostic by layer.

![Figure R.8(d): GPT-2 right singular-vector Kolmogorov diagnostic by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_kolmogorov_distance_convergence_log.png)

Figure R.8(d) path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_kolmogorov_distance_convergence_log.png

**Figure R.8(e).** GPT-2 left singular-vector angular-Wasserstein diagnostic by layer.

![Figure R.8(e): GPT-2 left singular-vector Wasserstein diagnostic by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_wasserstein_distance_convergence_log.png)

Figure R.8(e) path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_left_wasserstein_distance_convergence_log.png

**Figure R.8(f).** GPT-2 right singular-vector angular-Wasserstein diagnostic by layer.

![Figure R.8(f): GPT-2 right singular-vector Wasserstein diagnostic by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_wasserstein_distance_convergence_log.png)

Figure R.8(f) path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/per_layer/gpt2_right_wasserstein_distance_convergence_log.png

**Conclusion:** Yes—we now report head-resolved diagnostics computed from actual per-head score and attention matrices, aggregating only scalar distances or histogram values after each head has been decomposed independently.

# Response to Reviewer XnLy

We sincerely thank the reviewer for the mathematically insightful and constructive comments, which helped us sharpen the assumptions, add new results, and clarify the empirical evidence.

## Q1: When does isotropy of \(X\) imply the required isotropy of \(S\)?

1. **Mathematical side.** If \(W_Q\) and \(W_K\) are fixed or independent of the random input \(X\), a token-space rotation \(X\mapsto OX\) gives

   \[
   Q=XW_Q\mapsto OQ,\qquad
   K=XW_K\mapsto OK,
   \]

   and hence

   \[
   S=\frac{QK^\top}{\sqrt{d_{QK}}}
   \mapsto OSO^\top.
   \]

   This calculation proves the inherited simultaneous-conjugation law. We will state separately the stronger modeling hypothesis that the left and right score orientations are Haar and independent of \(\Sigma_h\). The direct score experiments below test its orientation component on the actual \(S_h\), rather than relying only on inheritance from \(X\).

2. **Empirical side.** We agree that testing \(X\) alone leaves a gap, so the new experiments bypass the inheritance argument and directly decompose every actual per-head score matrix \(S_h=U_{S_h}\Sigma_hV_{S_h}^\top\). We compare the eigen-angle laws of both \(U_{S_h}\) and \(V_{S_h}\) with the circular-uniform Haar reference using KL divergence, Kolmogorov distance, and angular \(W_1\), all of which are minimized at zero.

   For GPT-2, the server run contains all 1,646 nontrivial WikiText-103 validation sequences with \(17\le\ell\le471\), all 12 layers, and all 12 heads, giving 237,024 per-head score SVDs. The table reports mean \(\pm\) standard deviation over head-layer observations in each length band; each sequence contributes 144 observations.

   | \(\ell\) band | \(n_{\rm seq}\) | \(U_{S}\): KL | \(U_{S}\): Kol. | \(U_{S}\): angular \(W_1\) | \(V_{S}\): KL | \(V_{S}\): Kol. | \(V_{S}\): angular \(W_1\) |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 17--64 | 254 | 0.3393 \(\pm\) 0.2005 | 0.0406 \(\pm\) 0.0145 | 0.3090 \(\pm\) 0.1435 | 0.3387 \(\pm\) 0.2001 | 0.0401 \(\pm\) 0.0136 | 0.3098 \(\pm\) 0.1430 |
   | 65--128 | 512 | 0.0655 \(\pm\) 0.0374 | 0.0202 \(\pm\) 0.0051 | 0.2040 \(\pm\) 0.0901 | 0.0655 \(\pm\) 0.0373 | 0.0203 \(\pm\) 0.0051 | 0.2050 \(\pm\) 0.0907 |
   | 129--192 | 464 | 0.0244 \(\pm\) 0.0096 | 0.0131 \(\pm\) 0.0026 | 0.1590 \(\pm\) 0.0692 | 0.0245 \(\pm\) 0.0096 | 0.0132 \(\pm\) 0.0026 | 0.1585 \(\pm\) 0.0688 |
   | 193--256 | 275 | 0.0131 \(\pm\) 0.0047 | 0.0098 \(\pm\) 0.0018 | 0.1338 \(\pm\) 0.0582 | 0.0131 \(\pm\) 0.0047 | 0.0099 \(\pm\) 0.0018 | 0.1335 \(\pm\) 0.0576 |
   | 257--320 | 100 | 0.0083 \(\pm\) 0.0029 | 0.0079 \(\pm\) 0.0013 | 0.1183 \(\pm\) 0.0512 | 0.0083 \(\pm\) 0.0028 | 0.0080 \(\pm\) 0.0014 | 0.1182 \(\pm\) 0.0513 |
   | 321--384 | 33 | 0.0055 \(\pm\) 0.0019 | 0.0065 \(\pm\) 0.0010 | 0.1054 \(\pm\) 0.0454 | 0.0055 \(\pm\) 0.0019 | 0.0066 \(\pm\) 0.0011 | 0.1061 \(\pm\) 0.0462 |
   | 385--471 | 8 | 0.0039 \(\pm\) 0.0013 | 0.0055 \(\pm\) 0.0009 | 0.0950 \(\pm\) 0.0421 | 0.0039 \(\pm\) 0.0013 | 0.0055 \(\pm\) 0.0009 | 0.0964 \(\pm\) 0.0418 |

   All six discrepancies decrease monotonically with sequence length. For ViT-B/16, we process all 3,925 Imagenette validation images, all 12 layers, and all 12 heads, giving 565,200 per-head score SVDs. Entries below are pooled mean \(\pm\) standard deviation, with the min--max range of the 12 layer-wise means in parentheses; each layer mean contains 47,100 image-head observations.

   | ViT-B/16 side | KL | Kolmogorov | Angular \(W_1\) |
   |:---|---:|---:|---:|
   | Left \(U_{S}\) | 0.002205 \(\pm\) 0.000715 (0.002199--0.002209) | 0.004203 \(\pm\) 0.000617 (0.004197--0.004209) | 0.082197 \(\pm\) 0.035621 (0.081902--0.082473) |
   | Right \(V_{S}\) | 0.002206 \(\pm\) 0.000717 (0.002202--0.002215) | 0.004205 \(\pm\) 0.000618 (0.004185--0.004214) | 0.082226 \(\pm\) 0.035588 (0.082044--0.082480) |

   The left and right results are nearly identical. GPT-2 becomes increasingly Haar-like as \(\ell\) grows, while ViT remains uniformly close to the Haar reference across all layers (see **Figures R.3--R.4**). Thus the score-orientation premise is tested on the actual \(S_h\), not inferred only from \(X\); independence from \(\Sigma_h\) remains the explicitly stated hypothesis.

**Cross-reference.** Reviewer WUoc Q1 and Reviewer CLzG Q3 use the same score-level diagnostics to support the diagonalization-in-law reduction and its post-softmax consequence.

**Figure R.3.** GPT-2 actual-score left/right singular-vector Haar diagnostics versus sequence length.

![Figure R.3: GPT-2 actual-score singular-vector Haar diagnostics](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png)

Figure R.3 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.png

Figure R.3 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/gpt2_svd_distance_metrics_three.svg

**Figure R.4.** ViT-B/16 actual-score left/right singular-vector Haar diagnostics across layers.

![Figure R.4: ViT actual-score singular-vector Haar diagnostics](https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png)

Figure R.4 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.png

Figure R.4 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/invariance_check_S/vit_svd_layer_distance_boxplot.svg

**Conclusion:** Isotropy of \(X\) passes to the simultaneous token-space conjugation law of \(S\) when \(W_Q,W_K\) are fixed or independent of \(X\), while the new score-level experiments directly support the stronger Haar-orientation premise and independence from \(\Sigma_h\) remains explicit.

## Q2: Do encoder and decoder spectra degenerate at different rates?

1. **Empirical side.** ViT sequence length cannot be swept asymptotically without changing the model setup, so we cannot make the same length-convergence plot as for GPT-2. Nevertheless, Figures 4 and 5 of the submitted paper, reproduced here as **Figures R.9--R.10**, compare encoder and decoder attention at similar lengths: ViT-B/16 has \(\ell=577\), and the longest GPT-2 WikiText example has \(\ell\approx470\). The encoder bulk is already very thin and concentrated at a substantially smaller scale from Layer 1, whereas the decoder retains a visibly broader bulk across layers. These cross-model figures provide empirical evidence consistent with faster encoder degeneration.

2. **Mathematical side---new result.** To answer the encoder--decoder comparison theoretically, we derive new stable-rank results. Under the theorem's aggregate-tail condition and finite or sublinear \(k\), bidirectional attention satisfies

   \[
   \operatorname{srank}(A)=(k+1)(1+o(1)).
   \]

   By contrast, for causal attention with bounded bulk scores, we newly obtain

   \[
   \operatorname{srank}(A)
   =\Theta_B\!\left(k+\log\frac{\ell+1}{k+1}\right).
   \]

   Thus, for finite \(k\), the bidirectional stable rank remains asymptotic to \(k+1\), whereas the causal stable rank retains an additional logarithmic term and therefore degenerates more slowly. These stable-rank theorems and their proofs are new results derived for the revision.

**Figure R.9.** ViT-B/16 encoder attention singular-value histograms across layers at \(\ell=577\) (submitted-paper Figure 4).

![Figure R.9: ViT encoder attention spectra across layers](https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/vit_encoder_attention_spectrum_by_layer.png)

Figure R.9 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/vit_encoder_attention_spectrum_by_layer.png

Figure R.9 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/vit_encoder_attention_spectrum_by_layer.svg

**Figure R.10.** GPT-2 decoder attention singular-value histograms across layers at \(\ell=471\) (submitted-paper Figure 5).

![Figure R.10: GPT-2 decoder attention spectra across layers](https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/gpt2_decoder_attention_spectrum_by_layer.png)

Figure R.10 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/gpt2_decoder_attention_spectrum_by_layer.png

Figure R.10 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/gpt2_decoder_attention_spectrum_by_layer.svg

**Conclusion:** Under their stated conditions, our newly derived theorems rigorously establish constant-order bidirectional versus logarithmically larger causal stable rank in the canonical model, while **Figures R.9--R.10** provide comparable-length cross-model evidence consistent with this contrast.

## Q3: Are the assumptions applied per head or after aggregating heads?

1. **Direct answer and mathematical context.** We apply the score-level isotropy and spike--bulk assumptions to each head separately, not to a score matrix obtained by aggregating heads. For every head \(h\),

   \[
   S_h=\frac{Q_hK_h^\top}{\sqrt{d_{QK}}},
   \qquad
   \operatorname{rank}(S_h)\le d_{QK}\ll\ell,
   \]

   so a per-head score matrix is structurally low rank in the practical regime \(d_{QK}\ll\ell\). This is the context in which the paper's spike--bulk assumption should be understood.

2. **Scope.** The present paper does not explicitly analyze the composed multi-head attention block. In particular, the current theory does not propagate the result through head concatenation, the output projection \(W_O\), normalization, or residual composition. An explicit multi-head theory that exploits the low-rank structure of every head is an ambitious direction we plan to pursue in follow-up work.

3. **Brief empirical clarification.** In the new score-spectrum experiment, each \(S_h\) is decomposed independently; singular values are pooled only after the per-head SVDs to form each layer's histogram (see **Figures R.5--R.6**).

**Cross-reference.** Reviewer WUoc Q3 discusses the scope beyond isolated attention, and Reviewer CLzG Q4 lists the new head-resolved score, attention, and singular-vector diagnostics.

**Figure R.5.** GPT-2 actual pre-softmax score singular values by layer, computed per head before pooling.

![Figure R.5: GPT-2 actual score singular values by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png)

Figure R.5 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/gpt2_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Figure R.6.** ViT-B/16 actual pre-softmax score singular values by layer, computed per head before pooling.

![Figure R.6: ViT actual score singular values by layer](https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png)

Figure R.6 path: https://anonymous.4open.science/r/attention-4ECD/rebuttal/score_bulk_spike/vit_score_svd_spike_bulk_by_layer_log_empirical_gap.png

**Conclusion:** The direct answer is that the score-level assumptions are imposed and tested per head, while an explicit theory for the composed multi-head block remains outside the present paper and is planned for follow-up work.

## Q4: What controls the number of spikes \(k\)?

1. **Empirical side.** Experiment 4.2 and Figure 3 of the submitted paper, reproduced here as **Figure R.7**, provide a direct practical answer. Across WikiText-103 validation sequences with lengths from \(17\) to \(471\), the stable rank of the score matrix remains below approximately \(2\), while its effective rank remains below approximately \(30\). Neither proxy increases substantially with \(\ell\). Therefore, over the tested range, the effective number of dominant score directions remains finite and moderate and does not appear to scale with sequence length or with the number of semantically distinct tokens.

2. **Mathematical side.** Every head also obeys the structural bound

   \[
   k\le \operatorname{rank}(S_h)\le d_h,
   \]

   with \(d_h=64\) for both tested checkpoints. Thus a fixed head dimension provides an \(\ell\)-independent upper bound, while **Figure R.7** shows that the observed effective dimension is substantially smaller. Within this structural cap, the realized \(k\) is determined by the input, layer/head, and learned query/key projections. Estimating the exact number of spikes is a longstanding model-selection problem in spiked random-matrix statistics; see Zhidong Bai, Shurong Zheng, and Jianfeng Yao, *Large Sample Covariance Matrices and High-Dimensional Data Analysis* (Cambridge University Press, 2015), for a comprehensive reference. Our degeneration results require \(k\) to remain finite or sublinear but do not require an exact estimator of \(k\).

**Cross-reference.** Reviewer WUoc Q2 and Reviewer CLzG Q2 provide the complementary direct score-spectrum evidence for the practical bulk--spike morphology.

**Figure R.7.** Stable and effective ranks of the WikiText-103 score matrices versus sequence length (submitted-paper Figure 3), with medians and IQRs.

![Figure R.7: Score stable and effective ranks versus sequence length](https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png)

Figure R.7 PNG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.png

Figure R.7 SVG: https://anonymous.4open.science/r/attention-4ECD/rebuttal/manuscript_figures/score_rank_proxies_vs_sequence_length.svg

**Conclusion:** The head dimension gives an \(\ell\)-independent cap, while **Figure R.7** shows a substantially smaller observed effective dimension; together these results support finite/moderate \(k\), although the rank proxies do not estimate its exact threshold-defined value.
