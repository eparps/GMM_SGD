# `GMM_SGD`
This is an implementation of the *2D Mixture of Gaussians (MOG)* model based on Toscano &amp; McMurray (2010) which was used in my Master's Thesis (*Differential Cue Weighting in Sibilants: A Case Study of Two Sinitic Languages*). For more detailed information about the model and the simulation processes, check out the original papers (mentioned in the references).

## Create a `GMM_SGD` object to generate the initial distributions
To start off a simulation, first create a `GMM_SGD` object by:

``` python
import GMM_SGD

gmm = GMM_SGD(
    k=20,
    init_mu_means=np.array([0, 0], dtype=float),
    init_mu_covariances=np.array([[1.5**2, 0], [0, 1.5**2]], dtype=float),
    init_covariances=np.array([[0.5**2, 0], [0, 0.5**2]], dtype=float)
    )
```

This will immediately generate the initial distributions based on the parameters specified. The information as well as a figure of the initial Gaussians will be shown on the control panel:

```
There are 20 clusters.
mu_x = [ 1.701 -2.322 -1.876  1.044  0.416 -2.034 -1.759  1.301  1.981  1.65   2.738  1.936  1.089  1.683 -1.146 -0.997  0.188  0.424 -0.534 -1.647]
mu_y = [-1.074 -0.332  1.966  1.607  0.226  1.551  2.022 -0.972 -3.033  0.655 -2.275  2.038 -0.266 -2.438 -0.81   2.201  2.15   2.894  1.218  1.282]
sd_x = [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
sd_y = [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
rho = [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
pi = [0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05]

![output1](https://github.com/eparps/GMM_SGD/blob/main/image/output1.png?raw=true)
```

## Generate the training distributions
Next, we generate the training distributions by inputing a dataframe containing the real world dataset. The training distributions will then be fitted using the provided dataset, which should look something like the following:

``` python
data = pd.read_excel('data_all.xlsx', sheet_name='4.0')
### Data cleaning ###
data = data[~data['Onset'].isin(['ʒ'])]
data = data[~data['Vowel'].isin(['ai', 'iu', 'e', 'o'])]
data = data[~data['IPA'].isin(['ʃɿ'])]
df_TM = data[data['Language'] == 'TM']
df_SH = data[data['Language'] == 'SH']
df_NH = data[data['Language'] == 'NH']
# Remove the outliers (zscore > 3).
df_TM = df_TM[(np.abs(zscore(df_TM.iloc[:, 7:18], ddof=1)) < 3).all(axis=1)]
df_SH = df_SH[(np.abs(zscore(df_SH.iloc[:, 7:18], ddof=1)) < 3).all(axis=1)]
df_NH = df_NH[(np.abs(zscore(df_NH.iloc[:, 7:18], ddof=1)) < 3).all(axis=1)]

df_TM.head() # Here we use the TM data as an example.
```

```
	Language	Speaker	Token	Repetition	IPA	Onset	Vowel	cog	sd	skewness	kurtosis	peak	F1_slope	F1_intercept	F2_slope	F2_intercept	F3_slope	F3_intercept	HF
0	TM	CYT	RND001	1.0	ɕou	ɕ	ou	5228	2138	0.19	-1.24	3086	0.906794	-0.967879	-0.900481	-0.373417	1.119548	-0.560756	N
1	TM	CYT	RND002	1.0	ʂa	ʂ	a	5135	1635	1.31	1.07	3882	1.307575	0.579932	-0.471833	-0.047246	-0.030565	-0.984602	N
2	TM	CYT	RND003	1.0	ɕa	ɕ	a	6073	1146	0.86	2.74	5973	1.575011	0.297072	-0.822768	0.136257	-1.761003	-0.537840	N
3	TM	CYT	RND004	1.0	ɕi	ɕ	i	6146	1434	0.70	1.43	6158	-0.172154	-1.031815	0.462426	2.188492	0.599728	1.267603	Y
4	TM	CYT	RND005	1.0	ʂɿ	ʂ	ɿ	4787	1869	0.78	-0.48	3086	0.035575	-1.136136	-1.987920	0.826618	0.098790	-0.033738	N
```

The input dataframe should only contain the relevant information, where in our case: the category label and the feature values for each observation. For example:

```python
# This corresponds to the third set of simulations for learning TM (refer to page 38 of the thesis).
# Note that the first column should be the labels.
df = df_TM.loc[:, ['Onset', 'cog', 'F2_intercept', 'skewness']]
df.head(10)
```

```
	Onset	cog	F2_intercept	skewness
0	ɕ	5228	-0.373417	0.19
1	ʂ	5135	-0.047246	1.31
2	ɕ	6073	0.136257	0.86
3	ɕ	6146	2.188492	0.70
4	ʂ	4787	0.826618	0.78
5	s	8979	-0.013218	-0.58
7	ʂ	4938	-0.616531	0.62
8	s	9001	0.085482	-0.62
9	ɕ	5353	-0.183715	0.14
10	ʂ	4897	-0.773203	1.01
```

To generate the traning distributions, use the `generate_training_distributions()` method:

```python
gmm.generate_training_distributions(df)
```

Note that if there are more than two input features, principle component analysis (PCA) will first be applied in order to reduce the dimension of the feature vector to two (since our model is 2D). The fiited distribtions for individual categories will be shown on the control panel:

```
There are 3 training categories: ['s', 'ɕ', 'ʂ']
mu_x = [ 1.66  -0.782 -0.9  ]
mu_y = [-0.131  0.667 -0.556]
sd_x = [0.738 0.678 0.769]
sd_y = [0.737 0.979 0.745]
rho = [ 0.56  -0.125  0.066]
pi = [0.333 0.333 0.333]

![output2](https://github.com/eparps/GMM_SGD/blob/main/image/output2.png)
```

## Run the simulation
To start the simulation, use the `sim()` method and specify the values for each argument:

``` python
gmm.sim(n_generation=1,
        n_data=100000,
        lr_pi=0.001,
        lr_mu=0.01,
        lr_sd=0.001,
        lr_rho=0.001,
        threshold=0.01)
```

After the simulation completes, only the Gaussians with a prior probability (pi) greater than the `threshold` value will be counted as stable categories and will thus be shown in the results. The model is considered "success" if it converges to the same number of categories with that of the training distributions (in this case, 3 categories).

```
['s', 'ɕ', 'ʂ']

Generation 1:

100000it [07:57, 209.47it/s]

Success!
training_mu_x = [ 1.66  -0.782 -0.9  ]
training_mu_y = [-0.131  0.667 -0.556]
training_sd_x = [0.738 0.678 0.769]
training_sd_y = [0.737 0.979 0.745]
training_rho = [ 0.56  -0.125  0.066]
training_pi = [0.333 0.333 0.333]

resulting_mu_x = [ 1.494 -0.807 -0.939]
resulting_mu_y = [-0.336  0.736 -0.468]
resulting_sd_x = [0.753 0.651 0.722]
resulting_sd_y = [0.77  0.956 0.764]
resulting_rho = [ 0.547 -0.145  0.089]
resulting_pi = [0.359 0.315 0.327]

Training Bhattacharyya distances: {'s-ɕ': 1.896, 's-ʂ': 1.52, 'ɕ-ʂ': 0.28}
Mean training Bhattacharyya distance: 1.232
Resulting Bhattacharyya distances: {'s-ɕ': 1.868, 's-ʂ': 1.507, 'ɕ-ʂ': 0.271}
Mean resulting Bhattacharyya distance: 1.215

![output3](https://github.com/eparps/GMM_SGD/blob/main/image/output3.png)
![output4](https://github.com/eparps/GMM_SGD/blob/main/image/output4.png)
![output5](https://github.com/eparps/GMM_SGD/blob/main/image/output5.png)
![output6](https://github.com/eparps/GMM_SGD/blob/main/image/output6.png)
![output7](https://github.com/eparps/GMM_SGD/blob/main/image/output7.png)
![output8](https://github.com/eparps/GMM_SGD/blob/main/image/output8.png)
![output9](https://github.com/eparps/GMM_SGD/blob/main/image/output9.png)
![output10](https://github.com/eparps/GMM_SGD/blob/main/image/output10.png)
```

## Manually specify the weight for each traning category (*p.43* of the thesis)
If we were to manually specify the wieght for each training category, simply use the `training_pi` argument, for example:

``` python
training_pi = np.array([0.45, 0.1, 0.45])
gmm.generate_training_distributions(df, training_pi=training_pi)
```

```
There are 3 training categories: ['s', 'ɕ', 'ʂ']
mu_x = [ 1.66  -0.782 -0.9  ]
mu_y = [-0.131  0.667 -0.556]
sd_x = [0.738 0.678 0.769]
sd_y = [0.737 0.979 0.745]
rho = [ 0.56  -0.125  0.066]
pi = [0.45 0.1  0.45]

![output11](https://github.com/eparps/GMM_SGD/blob/main/image/output11.png)
```

## References
[Toscano, J. C. & McMurray, B. (2010). Cue integration with categories: Weighting acoustic cues in speech using unsupervised learning and distributional statistics. *Cognitive science*, 34(3):434-464.](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1551-6709.2009.01077.x)

[McMurray, B., Aslin, R. N., & Toscano, J. C. (2009). Statistical learning of phonetic categories: Insights from a computational approach. *Developmental science*, 12(3):369-378.](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-7687.2009.00822.x?casa_token=rJXx8rbnsdEAAAAA:vBgAm7kaLgaUXA_-Po1QPzt3cQRfeM9bo7z2pN3hJuBkTNFFg9H9J61MZoCfnwFkfbzGgjiIDLaYCULP)

[Tang, C.-h. (2022). Differential cue weighting in sibilants: A case study of two sinitic languages. Master's thesis, National Tsing Hua University.](https://etd.lib.nctu.edu.tw/cgi-bin/gs32/hugsweb.cgi?o=dnthucdr&s=id=%22G021070445010%22.&searchmode=basic)




