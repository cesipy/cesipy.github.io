# Results
This file contains all kinds of results and observations during my thesis work.

## 21.10 - hypotheses

better hateful memes dataset: https://www.kaggle.com/discussions/general/202833

H1: Alignment is task dependent: some tasks benefit more from one coattention placement than others.

H2: Intra-layer alignment is more important than final alignment!
- do correlation testing here. `corr(cka[-1], accuracy` vs `corr(max(cka), accuracy)`

H3: coattention placements increase metrics.


show correlation, but not absolute values same:
<figure>
<img src="./res/markdown_res/alignment_metrics_comparison.png">
</figure>





## 20.10
currently running more rigorous experiments on the correlation between num_samples in dataset and K and metrics, that I can directly report them to my thesis. also includes time findings.
This analysis is done while the finetuning step of pretrained models is trained for three seeds a three tasks (9) per pretrained.


results are below:
The analysis was done for 49 models (15/15/19: mm_imdb/upmc_food/hateful_memes)

- running time for num_samples= 512: 14.12s ± 5.12 (std)
- running time for num_samples=1536: 42.64s ± 4.82 (std) on uni-gpu, on my gpu way worse as cpu has to be used. over 5 minutes for it.
also


for metrics of interest (cka, procrustes,svcca, mknn ), inter-model correlation was high.
| Metric      | Mean (within-model) | Std Dev    | Mean p-value | Std Dev (p-value) |
|-------------|---------------------|------------|--------------|-------------------|
| mknn        | 0.9077              | 0.1240     | 0.0051       | 0.0169            |
| procrustes  | 0.9957              | 0.0058     | 0.0000       | 0.0000            |
| cka         | 0.9869              | 0.0284     | 0.0000       | 0.0002            |
| svcca       | 0.9264              | 0.0869     | 0.0024       | 0.0142            |


also the new correlation between metrics was similar to the results with fewer metrics:

<figure>
<img src="./res/markdown_res/20252010_metric_analysis/mknn_spearmanr.png">
</figure>










---

## 18.10 & 17.10
**summary**:
to sum my current thesis: Basically this thesis studies representational alignment in two-stream architectures (ViLBERT).

i) How do cross-attention layer between the streams affect the representational alignment?

ii) is there a corerlationbetween performance (acc) and alignment? is it task dependent?

iii) Is there an correlation between representational alignment and coattention placement? (how to measure this??)

iv) optimal alignment for archicture, how is the overall performance?

**past ⁊ current experiments**:
i) best performing architectures for mm_imdb and hateful memes, searched via optuna.

ii) pretraining for the below architectures. (early, mid, late, asymmetric, optuna1, optuna2, [optuna3 is still todo])

iii) correlation analysis of num_dataset and k (kNN mesures) and between measurements.

iv) currently: finetuning on all three tasks (mmimdb, hm, upmcfood) for each three seeds ($6\cdot3\cdot3$)
- correlation analysis for repr measures and performance


**additional things**:
i) is representational alignment really increasing after coattns?

ii) directly optimize for alignment measures (like cka)


---


## 21.10

found imbalances in the hm dataset:
train data: class balance: 0.3551764705882353
Positive samples: 3019, Negative samples: 5481

validation data: class balance: 0.42980769230769234
Positive samples: 447, Negative samples: 593


## 17.10

This comparision is on the two tasks `mm_imdb` and `upmc_food`, each with 15 finetune-only models. this pools the models here for one task and evaluates it for the test and train set. Here different architectures where used. This is a analysis of predictability of performance from alignment measures in general.
<details closed>

	mm_imdb: test loss=0.1897, test acc=0.9233
	mm_imdb: val loss=0.1803, val acc=0.9279
	mm_imdb: test loss=0.1897, test acc=0.9233
	mm_imdb:  val loss=0.1803,  val acc=0.9279
	mm_imdb: test loss=0.1887, test acc=0.9236
	mm_imdb:  val loss=0.1783,  val acc=0.9286
	mm_imdb: test loss=0.1880, test acc=0.9248
	mm_imdb:  val loss=0.1775,  val acc=0.9297
	mm_imdb: test loss=0.1854, test acc=0.9253
	mm_imdb:  val loss=0.1743,  val acc=0.9304
	mm_imdb: test loss=0.1836, test acc=0.9269
	mm_imdb:  val loss=0.1710,  val acc=0.9325
	mm_imdb: test loss=0.1853, test acc=0.9259
	mm_imdb:  val loss=0.1748,  val acc=0.9307
	mm_imdb: test loss=0.1814, test acc=0.9277
	mm_imdb:  val loss=0.1686,  val acc=0.9334
	mm_imdb: test loss=0.1858, test acc=0.9261
	mm_imdb:  val loss=0.1741,  val acc=0.9314
	mm_imdb: test loss=0.1795, test acc=0.9290
	mm_imdb:  val loss=0.1685,  val acc=0.9339
	mm_imdb: test loss=0.1874, test acc=0.9245
	mm_imdb:  val loss=0.1777,  val acc=0.9291
	mm_imdb: test loss=0.1866, test acc=0.9247
	mm_imdb:  val loss=0.1749,  val acc=0.9302
	mm_imdb: test loss=0.1880, test acc=0.9244
	mm_imdb:  val loss=0.1777,  val acc=0.9290
	mm_imdb: test loss=0.1805, test acc=0.9271
	mm_imdb:  val loss=0.1675,  val acc=0.9334
	mm_imdb: test loss=0.1843, test acc=0.9263
	mm_imdb:  val loss=0.1721,  val acc=0.9319
	mm_imdb: test loss=0.1787, test acc=0.9284
	mm_imdb:  val loss=0.1644,  val acc=0.9346
=========================
test dataset
corr. of mknn           with acc : r=+0.739, p=0.002
corr. of mknn           with loss: r=+0.693, p=0.004
corr. of cka            with acc : r=+0.704, p=0.003
corr. of cka            with loss: r=+0.655, p=0.008
corr. of cka_rbf        with acc : r=-0.023, p=0.935
corr. of cka_rbf        with loss: r=-0.235, p=0.400
corr. of unbiased_cka   with acc : r=+0.703, p=0.003
corr. of unbiased_cka   with loss: r=+0.654, p=0.008
corr. of svcca          with acc : r=+0.755, p=0.001
corr. of svcca          with loss: r=+0.729, p=0.002
corr. of cknna          with acc : r=+0.771, p=0.001
corr. of cknna          with loss: r=+0.730, p=0.002
corr. of cycle_knn      with acc : r=+0.052, p=0.855
corr. of cycle_knn      with loss: r=+0.126, p=0.656
corr. of procrustes     with acc : r=-0.263, p=0.344
corr. of procrustes     with loss: r=-0.071, p=0.803
corr. of jaccard        with acc : r=+0.734, p=0.002
corr. of jaccard        with loss: r=+0.682, p=0.005
corr. of rsa            with acc : r=+0.753, p=0.001
corr. of rsa            with loss: r=+0.723, p=0.002
corr. of r2             with acc : r=+0.683, p=0.005
corr. of r2             with loss: r=+0.644, p=0.010
=========================
corr. of mknn           with acc : r=+0.711, p=0.003
corr. of mknn           with loss: r=+0.671, p=0.006
corr. of cka            with acc : r=+0.707, p=0.003
corr. of cka            with loss: r=+0.675, p=0.006
corr. of cka_rbf        with acc : r=+0.186, p=0.508
corr. of cka_rbf        with loss: r=+0.004, p=0.990
corr. of unbiased_cka   with acc : r=+0.707, p=0.003
corr. of unbiased_cka   with loss: r=+0.675, p=0.006
corr. of svcca          with acc : r=+0.689, p=0.004
corr. of svcca          with loss: r=+0.579, p=0.024
corr. of cknna          with acc : r=+0.711, p=0.003
corr. of cknna          with loss: r=+0.664, p=0.007
corr. of cycle_knn      with acc : r=-0.059, p=0.834
corr. of cycle_knn      with loss: r=+0.028, p=0.922
corr. of procrustes     with acc : r=-0.182, p=0.516
corr. of procrustes     with loss: r=+0.014, p=0.960
corr. of jaccard        with acc : r=+0.711, p=0.003
corr. of jaccard        with loss: r=+0.671, p=0.006
corr. of rsa            with acc : r=+0.732, p=0.002
corr. of rsa            with loss: r=+0.689, p=0.004
corr. of r2             with acc : r=+0.704, p=0.003
corr. of r2             with loss: r=+0.632, p=0.011
=========================
validation dataset
corr. of mknn           with acc : r=+0.746, p=0.001
corr. of mknn           with loss: r=+0.696, p=0.004
corr. of cka            with acc : r=+0.711, p=0.003
corr. of cka            with loss: r=+0.658, p=0.008
corr. of cka_rbf        with acc : r=-0.117, p=0.678
corr. of cka_rbf        with loss: r=-0.258, p=0.353
corr. of unbiased_cka   with acc : r=+0.709, p=0.003
corr. of unbiased_cka   with loss: r=+0.656, p=0.008
corr. of svcca          with acc : r=+0.780, p=0.001
corr. of svcca          with loss: r=+0.736, p=0.002
corr. of cknna          with acc : r=+0.783, p=0.001
corr. of cknna          with loss: r=+0.735, p=0.002
corr. of cycle_knn      with acc : r=+0.107, p=0.705
corr. of cycle_knn      with loss: r=+0.156, p=0.579
corr. of procrustes     with acc : r=-0.178, p=0.525
corr. of procrustes     with loss: r=-0.040, p=0.888
corr. of jaccard        with acc : r=+0.739, p=0.002
corr. of jaccard        with loss: r=+0.685, p=0.005
corr. of rsa            with acc : r=+0.774, p=0.001
corr. of rsa            with loss: r=+0.733, p=0.002
corr. of r2             with acc : r=+0.672, p=0.006
corr. of r2             with loss: r=+0.626, p=0.013
=========================
corr. of mknn           with acc : r=+0.714, p=0.003
corr. of mknn           with loss: r=+0.693, p=0.004
corr. of cka            with acc : r=+0.714, p=0.003
corr. of cka            with loss: r=+0.700, p=0.004
corr. of cka_rbf        with acc : r=+0.139, p=0.621
corr. of cka_rbf        with loss: r=+0.046, p=0.869
corr. of unbiased_cka   with acc : r=+0.714, p=0.003
corr. of unbiased_cka   with loss: r=+0.700, p=0.004
corr. of svcca          with acc : r=+0.668, p=0.007
corr. of svcca          with loss: r=+0.639, p=0.010
corr. of cknna          with acc : r=+0.714, p=0.003
corr. of cknna          with loss: r=+0.693, p=0.004
corr. of cycle_knn      with acc : r=+0.008, p=0.978
corr. of cycle_knn      with loss: r=-0.055, p=0.845
corr. of procrustes     with acc : r=-0.125, p=0.657
corr. of procrustes     with loss: r=-0.021, p=0.940
corr. of jaccard        with acc : r=+0.714, p=0.003
corr. of jaccard        with loss: r=+0.693, p=0.004
corr. of rsa            with acc : r=+0.736, p=0.002
corr. of rsa            with loss: r=+0.704, p=0.003
corr. of r2             with acc : r=+0.689, p=0.004
corr. of r2             with loss: r=+0.657, p=0.008


	upmc_food: test loss=0.4185, test acc=0.9040
	upmc_food:  val loss=0.4039,  val acc=0.9065
	upmc_food: test loss=0.4067, test acc=0.9079
	upmc_food:  val loss=0.4000,  val acc=0.9092
	upmc_food: test loss=0.4244, test acc=0.9020
	upmc_food:  val loss=0.4286,  val acc=0.9035
	upmc_food: test loss=0.3638, test acc=0.9203
	upmc_food:  val loss=0.3543,  val acc=0.9194
	upmc_food: test loss=0.3651, test acc=0.9187
	upmc_food:  val loss=0.3651,  val acc=0.9188
	upmc_food: test loss=0.3748, test acc=0.9179
	upmc_food:  val loss=0.3628,  val acc=0.9168
	upmc_food: test loss=0.2812, test acc=0.9367
	upmc_food:  val loss=0.2733,  val acc=0.9388
	upmc_food: test loss=0.2854, test acc=0.9367
	upmc_food:  val loss=0.2777,  val acc=0.9369
	upmc_food: test loss=0.2879, test acc=0.9371
	upmc_food:  val loss=0.2809,  val acc=0.9374
	upmc_food: test loss=0.3408, test acc=0.9261
	upmc_food:  val loss=0.3276,  val acc=0.9245
	upmc_food: test loss=0.3326, test acc=0.9272
	upmc_food:  val loss=0.3309,  val acc=0.9237
	upmc_food: test loss=0.3343, test acc=0.9272
	upmc_food:  val loss=0.3311,  val acc=0.9261
	upmc_food: test loss=0.3362, test acc=0.9249
	upmc_food:  val loss=0.3367,  val acc=0.9259
	upmc_food: test loss=0.3308, test acc=0.9299
	upmc_food:  val loss=0.3276,  val acc=0.9281
=========================
test dataset
corr. of mknn           with acc : r=+0.751, p=0.002
corr. of mknn           with loss: r=+0.798, p=0.001
corr. of cka            with acc : r=+0.724, p=0.003
corr. of cka            with loss: r=+0.780, p=0.001
corr. of cka_rbf        with acc : r=+0.729, p=0.003
corr. of cka_rbf        with loss: r=+0.703, p=0.005
corr. of unbiased_cka   with acc : r=+0.728, p=0.003
corr. of unbiased_cka   with loss: r=+0.784, p=0.001
corr. of svcca          with acc : r=+0.729, p=0.003
corr. of svcca          with loss: r=+0.751, p=0.002
corr. of cknna          with acc : r=+0.743, p=0.002
corr. of cknna          with loss: r=+0.784, p=0.001
corr. of cycle_knn      with acc : r=+0.394, p=0.164
corr. of cycle_knn      with loss: r=+0.424, p=0.131
corr. of procrustes     with acc : r=-0.401, p=0.156
corr. of procrustes     with loss: r=-0.485, p=0.079
corr. of jaccard        with acc : r=+0.753, p=0.002
corr. of jaccard        with loss: r=+0.802, p=0.001
corr. of rsa            with acc : r=+0.718, p=0.004
corr. of rsa            with loss: r=+0.772, p=0.001
corr. of r2             with acc : r=+0.585, p=0.028
corr. of r2             with loss: r=+0.644, p=0.013
=========================
corr. of mknn           with acc : r=+0.654, p=0.011
corr. of mknn           with loss: r=+0.688, p=0.007
corr. of cka            with acc : r=+0.709, p=0.005
corr. of cka            with loss: r=+0.723, p=0.003
corr. of cka_rbf        with acc : r=+0.760, p=0.002
corr. of cka_rbf        with loss: r=+0.728, p=0.003
corr. of unbiased_cka   with acc : r=+0.672, p=0.009
corr. of unbiased_cka   with loss: r=+0.684, p=0.007
corr. of svcca          with acc : r=+0.705, p=0.005
corr. of svcca          with loss: r=+0.701, p=0.005
corr. of cknna          with acc : r=+0.736, p=0.003
corr. of cknna          with loss: r=+0.780, p=0.001
corr. of cycle_knn      with acc : r=+0.580, p=0.030
corr. of cycle_knn      with loss: r=+0.606, p=0.022
corr. of procrustes     with acc : r=-0.295, p=0.306
corr. of procrustes     with loss: r=-0.270, p=0.350
corr. of jaccard        with acc : r=+0.654, p=0.011
corr. of jaccard        with loss: r=+0.688, p=0.007
corr. of rsa            with acc : r=+0.643, p=0.013
corr. of rsa            with loss: r=+0.666, p=0.009
corr. of r2             with acc : r=+0.623, p=0.017
corr. of r2             with loss: r=+0.657, p=0.011
=========================
validation dataset
corr. of mknn           with acc : r=+0.817, p=0.000
corr. of mknn           with loss: r=+0.779, p=0.001
corr. of cka            with acc : r=+0.792, p=0.001
corr. of cka            with loss: r=+0.782, p=0.001
corr. of cka_rbf        with acc : r=+0.698, p=0.006
corr. of cka_rbf        with loss: r=+0.711, p=0.004
corr. of unbiased_cka   with acc : r=+0.797, p=0.001
corr. of unbiased_cka   with loss: r=+0.784, p=0.001
corr. of svcca          with acc : r=+0.777, p=0.001
corr. of svcca          with loss: r=+0.734, p=0.003
corr. of cknna          with acc : r=+0.801, p=0.001
corr. of cknna          with loss: r=+0.764, p=0.001
corr. of cycle_knn      with acc : r=+0.451, p=0.106
corr. of cycle_knn      with loss: r=+0.382, p=0.177
corr. of procrustes     with acc : r=-0.474, p=0.087
corr. of procrustes     with loss: r=-0.530, p=0.051
corr. of jaccard        with acc : r=+0.820, p=0.000
corr. of jaccard        with loss: r=+0.784, p=0.001
corr. of rsa            with acc : r=+0.791, p=0.001
corr. of rsa            with loss: r=+0.767, p=0.001
corr. of r2             with acc : r=+0.673, p=0.008
corr. of r2             with loss: r=+0.641, p=0.014
=========================
corr. of mknn           with acc : r=+0.754, p=0.002
corr. of mknn           with loss: r=+0.556, p=0.039
corr. of cka            with acc : r=+0.789, p=0.001
corr. of cka            with loss: r=+0.697, p=0.006
corr. of cka_rbf        with acc : r=+0.744, p=0.002
corr. of cka_rbf        with loss: r=+0.821, p=0.000
corr. of unbiased_cka   with acc : r=+0.763, p=0.002
corr. of unbiased_cka   with loss: r=+0.662, p=0.010
corr. of svcca          with acc : r=+0.776, p=0.001
corr. of svcca          with loss: r=+0.596, p=0.025
corr. of cknna          with acc : r=+0.780, p=0.001
corr. of cknna          with loss: r=+0.640, p=0.014
corr. of cycle_knn      with acc : r=+0.682, p=0.007
corr. of cycle_knn      with loss: r=+0.455, p=0.102
corr. of procrustes     with acc : r=-0.305, p=0.288
corr. of procrustes     with loss: r=-0.398, p=0.159
corr. of jaccard        with acc : r=+0.754, p=0.002
corr. of jaccard        with loss: r=+0.556, p=0.039
corr. of rsa            with acc : r=+0.754, p=0.002
corr. of rsa            with loss: r=+0.596, p=0.025
corr. of r2             with acc : r=+0.723, p=0.003
corr. of r2             with loss: r=+0.530, p=0.051


</details>
$\Rightarrow$ negative correlation with procrustes, high correlation with cknna.
Interesting really low (highly significant) values for mm_imdb, while upmc_food shows moderate correlations with high p-values (not significant).
Seems to be task dependent.


Here's a concise analysis you can add:



The following metrics show strong, significant correlations (r > 0.7, p < 0.01) across both tasks:

| Metric | MM-IMDB (test) | UPMC-Food (test) |
|--------|----------------|------------------|
| cknna| r=0.771, p=0.001 | r=0.743, p=0.002 |
| mknn | r=0.739, p=0.002 | r=0.751, p=0.002 |
| svcca | r=0.755, p=0.001 | r=0.729, p=0.003 |
| jaccard | r=0.734, p=0.002 | r=0.753, p=0.002 |
| rsa | r=0.753, p=0.001 | r=0.718, p=0.004 |
| cka (linear) | r=0.704, p=0.003 | r=0.724, p=0.003 |

**Conclusion**: These 6 metrics reliably predict performance across diverse architectures and tasks.

**cka_rbf** shows dramatically different performance:
- MM-IMDB: r=-0.023 (p=0.935) - **fails completely**
- UPMC-Food: r=+0.729 (p=0.003) - **strong predictor**


---


is CLS in BERT even the same as CLS in ViT?

implemented test sets for all datasets. now in `experimentTracker.evaluate(model, task)` the test sets are used for computing accuracies.

Problem: hateful memes is too small for meaningful alignment analysis with 512, i have to adjust to 500. `dev.jsonl` contains only 500 samples

implemented sanity check for the measures:
```
SVCCA:           identical=0.8797, random=0.1669
CKA:             identical=1.0000, random=0.0548
CKA normed:      identical=1.0000, random=0.0552
CKA unbiased:    identical=1.0000, random=0.0290
Mutual KNN:      identical=1.0000, random=0.0618
CKNNA:           identical=1.0000, random=0.0194
Cycle KNN:       identical=0.8848, random=0.6484
LCS KNN:         identical=8.0000, random=0.3789
Edit Distance:   identical=1.0000, random=0.7501
Jaccard:         identical=1.0000, random=0.0406
Procrustes:      identical=nan, random=82.1158
RSA:             identical=1.0000, random=0.0286
```
weirdly, svcca has 0.87 even for identical measures, and 0.16 for random. Seems not to be quite in the range of $[0,1]$.

Problem here:
```
0: v-v: 0.9742729933317914, t-t: 0.9047365303899584, c-c: 0.13146975382067233
1: v-v: 0.9538666934372777, t-t: 0.855300764183276, c-c: 0.14771491771396944
2: v-v: 0.9437303334795629, t-t: 0.7867630363653866, c-c: 0.1734919959929184
3: v-v: 0.8232363334222592, t-t: 0.8860326698559919, c-c: 0.5732621813327985
4: v-v: 0.7557494530616364, t-t: 0.8556820596575738, c-c: 0.47956998649240273
5: v-v: 0.7959809675239169, t-t: 0.9044626923547732, c-c: 0.40650191487462395
6: v-v: 0.920212235651318, t-t: 0.8345659617976542, c-c: 0.6030498543079044
7: v-v: 0.8347529077031177, t-t: 0.8586370697752859, c-c: 0.49550097506102564
8: v-v: 0.8238978509428844, t-t: 0.8547694013716522, c-c: 0.4029174003196152
9: v-v: 0.8246623019834024, t-t: 0.8581564826585076, c-c: 0.3787406480808941
10:v-v: 0.8131022050219533, t-t: 0.8672419515501189, c-c: 0.31508022616828846
11:v-v: 0.8049661072054934, t-t: 0.878885846839872, c-c: 0.3102150474189421
```

`v-v`, `t-t` directly compares the embeddings of intra model matrix where `i==j` (both embedings are identical), but still only 0.87.


