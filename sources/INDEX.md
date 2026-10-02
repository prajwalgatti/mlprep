# Source cache index

Plain-text extracts of free, legally available sources for the ML-fundamentals and generative-modeling content rebuild. Built 2026-10-02.

Search with `grep -n` (or `rg`). Every PDF-derived file has page markers like `=== [esl pdf p.81] ===` (books) or `=== [ho2020_ddpm p.3] ===` (papers), so you can cite pages. Original PDFs are in `pdf/` and `pdf/papers/`; raw HTML is in `html/`.

## Extraction quality (read this first)

- **PDF books and papers** (pdftotext): prose reads well. **Equations are mangled.** Subscripts and superscripts end up on separate lines, fractions are flattened, Greek letters usually survive, and summation or integral limits get scattered. Use the text to find the right passage and its wording, then rebuild the maths yourself (or check the PDF in `pdf/`). Ligatures (fi, fl) were normalised.
- **ESL**: the section number and title are often on separate lines (`3.4.1` then `Ridge Regression`). Grep for the title, or use `grep -n -A1 '^3.4.1$' esl.txt`.
- **Deep Learning book (DLB)**: extracted from the official pdf2htmlEX HTML. Inline maths symbols come out as short tokens joined into the line (e.g. `x ∈ R n where each entry x i`). Display equations are fragmented. Section headings sit at the start of a line (`grep -n '^8.5' dlb_ch08_optimization.txt`). `=== [page N] ===` counts HTML pages within the chapter; the printed book page number appears as the last token of each page block.
- **d2l.ai** and the **blogs** (web/): the LaTeX source is preserved, e.g. `\(\mathbf{x}\)` and `\[...\]`. These are the best files for copying the exact form of an equation. Only the PyTorch code tabs were kept.
- `toc/*.txt`: PDF outlines (section title → **pdf** page) for ESL, Bishop, PML1, PML2, Boyd, UML and CS229. Start there to find a section.

## Books

| File | Source | Canonical URL | Printed page = pdf page − | How to find sections |
|---|---|---|---|---|
| `esl.txt` (1767 KB) | Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., 12th printing (2017; Springer 2009) | https://hastie.su.domains/ElemStatLearn/ | 19 | `toc/esl_toc.txt`; `grep -n -A1 '^7.10.2' esl.txt` (number and title on separate lines) |
| `bishop.txt` (1721 KB) | Bishop, *Pattern Recognition and Machine Learning* (Springer 2006; free PDF from Microsoft Research) | https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/ | 20 | `toc/bishop_toc.txt`; `grep -n '^9.1. K-means' bishop.txt`. Equation numbers like (3.28) survive and are greppable |
| `pml1.txt` (1919 KB) | Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press 2022; online version Apr 2025) | https://probml.github.io/pml-book/book1.html | 30 | `toc/pml1_toc.txt`; `grep -n '^6.2.6' pml1.txt` |
| `pml2.txt` (3606 KB) | Murphy, *Probabilistic Machine Learning: Advanced Topics* (MIT Press 2023; online version) | https://probml.github.io/pml-book/book2.html | 34 | `toc/pml2_toc.txt`; ch. 20–26 are generative models; §6.3.5–6.3.8 reparam/Gumbel/STE; §6.4 natural gradient; ch. 19 distribution shift |
| `uml.txt` (915 KB) | Shalev-Shwartz & Ben-David, *Understanding Machine Learning: From Theory to Algorithms* (CUP 2014) | https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/ | 0 | `toc/uml_toc.txt` (chapters only); `grep -n '^5.1 The No-Free-Lunch' uml.txt` |
| `boyd.txt` (1411 KB) | Boyd & Vandenberghe, *Convex Optimization* (CUP 2004) | https://web.stanford.edu/~boyd/cvxbook/ | 14 | `toc/boyd_toc.txt` has titles but no numbers. Ch. 3 starts at pdf p.81 and ch. 9 at pdf p.471. Section numbers sit alone on a line: `grep -n '^9.5.3$' boyd.txt` |
| `mackay.txt` (1762 KB) | MacKay, *Information Theory, Inference, and Learning Algorithms* (CUP 2003; on-screen viewing copy) | https://www.inference.org.uk/itprnn/book.html | 12 | No outline. Chapter start pdf pages: ch2 Probability/Entropy/Inference 34, ch4 Source Coding 79, ch8 Dependent RVs (mutual info) 150, ch20 Clustering (k-means) 296, ch22 ML & Clustering 312, ch36 Decision Theory 463, ch37 Bayesian Inference & Sampling Theory 469, ch39 Single Neuron 483 |
| `cs229.txt` (487 KB) | Ng, Ma et al., *CS229 Lecture Notes* (Stanford, current main_notes.pdf) | https://cs229.stanford.edu/main_notes.pdf | 1 | `toc/cs229_toc.txt`; chapter headings are `Chapter N` then the title on the next line (`grep -n -A1 '^Chapter 8$' cs229.txt`). Ch1 lin reg, 2 logistic, 3 GLM, 4 generative, 5 kernels, 6 SVM, 7 deep learning/backprop, 8 generalization & bias-variance/double descent, 9 regularization & model selection, 10 k-means, 11 EM/VAE, 12 PCA, 13 ICA, 14 diffusion, 15–16 foundation models/representation learning |

## Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press 2016): per chapter
Canonical: https://www.deeplearningbook.org/ (chapter URL = `https://www.deeplearningbook.org/contents/<name>.html`).

| File | Chapter | URL | Printed start page |
|---|---|---|---|
| `dlb_ch02_linear_algebra.txt` (40 KB) | Ch. 2 Linear Algebra (2.12 PCA derivation) | https://www.deeplearningbook.org/contents/linear_algebra.html | 29 |
| `dlb_ch03_prob.txt` (53 KB) | Ch. 3 Probability & Information Theory (3.13 info theory) | https://www.deeplearningbook.org/contents/prob.html | 51 |
| `dlb_ch04_numerical.txt` (36 KB) | Ch. 4 Numerical Computation (4.3 gradient/Hessian, 4.5 least squares) | https://www.deeplearningbook.org/contents/numerical.html | 78 |
| `dlb_ch05_ml.txt` (150 KB) | Ch. 5 Machine Learning Basics (capacity, NFL, CV, estimators, MLE, Bayes, SGD, curse of dim.) | https://www.deeplearningbook.org/contents/ml.html | 96 |
| `dlb_ch06_mlp.txt` (137 KB) | Ch. 6 Deep Feedforward Networks (losses, output units, activations, backprop) | https://www.deeplearningbook.org/contents/mlp.html | 164 |
| `dlb_ch07_regularization.txt` (104 KB) | Ch. 7 Regularization (L2/L1, early stopping, bagging, dropout) | https://www.deeplearningbook.org/contents/regularization.html | 224 |
| `dlb_ch08_optimization.txt` (133 KB) | Ch. 8 Optimization (SGD, momentum, init 8.4, AdaGrad/RMSProp/Adam 8.5, Newton/CG/BFGS 8.6, BatchNorm 8.7.1) | https://www.deeplearningbook.org/contents/optimization.html | 271 |
| `dlb_ch09_convnets.txt` (90 KB) | Ch. 9 Convolutional Networks | https://www.deeplearningbook.org/contents/convnets.html | 326 |
| `dlb_ch10_rnn.txt` (102 KB) | Ch. 10 Sequence Modeling: RNNs, LSTM 10.10, clipping 10.11 | https://www.deeplearningbook.org/contents/rnn.html | 367 |
| `dlb_ch11_guidelines.txt` (54 KB) | Ch. 11 Practical Methodology (metrics 11.1, debugging 11.5) | https://www.deeplearningbook.org/contents/guidelines.html | 416 |
| `dlb_ch14_autoencoders.txt` (50 KB) | Ch. 14 Autoencoders | https://www.deeplearningbook.org/contents/autoencoders.html | 499 |
| `dlb_ch15_representation.txt` (77 KB) | Ch. 15 Representation Learning (15.2 transfer/domain adaptation, zero/one-shot) | https://www.deeplearningbook.org/contents/representation.html | 524 |
| `dlb_ch20_generative_models.txt` (154 KB) | Ch. 20 Deep Generative Models (20.9 backprop through random ops, 20.10.3 VAE, 20.10.4 GAN, 20.10.7 autoregressive) | https://www.deeplearningbook.org/contents/generative_models.html | see last token of page 1 |

## d2l.ai (Zhang, Lipton, Li & Smola, *Dive into Deep Learning*, CC BY-SA): `d2l/`
Each file's first line is `SOURCE: <url>`. The LaTeX is preserved. Headings look like `## 12.10.1. ...`.

| File | URL |
|---|---|
| `d2l/d2l_adagrad.txt` | https://d2l.ai/chapter_optimization/adagrad.html |
| `d2l/d2l_adam.txt` | https://d2l.ai/chapter_optimization/adam.html |
| `d2l/d2l_backprop.txt` | https://d2l.ai/chapter_multilayer-perceptrons/backprop.html |
| `d2l/d2l_batch_norm.txt` | https://d2l.ai/chapter_convolutional-modern/batch-norm.html |
| `d2l/d2l_bptt.txt` | https://d2l.ai/chapter_recurrent-neural-networks/bptt.html |
| `d2l/d2l_channels.txt` | https://d2l.ai/chapter_convolutional-neural-networks/channels.html |
| `d2l/d2l_conv_layer.txt` | https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html |
| `d2l/d2l_convexity.txt` | https://d2l.ai/chapter_optimization/convexity.html |
| `d2l/d2l_distributions.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html |
| `d2l/d2l_dropout.txt` | https://d2l.ai/chapter_multilayer-perceptrons/dropout.html |
| `d2l/d2l_eigendecomposition.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html |
| `d2l/d2l_environment_and_distribution_shift.txt` | https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html |
| `d2l/d2l_fine_tuning.txt` | https://d2l.ai/chapter_computer-vision/fine-tuning.html |
| `d2l/d2l_gd.txt` | https://d2l.ai/chapter_optimization/gd.html |
| `d2l/d2l_generalization.txt` | https://d2l.ai/chapter_linear-regression/generalization.html |
| `d2l/d2l_generalization_deep.txt` | https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html |
| `d2l/d2l_gru.txt` | https://d2l.ai/chapter_recurrent-modern/gru.html |
| `d2l/d2l_information_theory.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html |
| `d2l/d2l_linear_regression.txt` | https://d2l.ai/chapter_linear-regression/linear-regression.html |
| `d2l/d2l_linear_regression_scratch.txt` | https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html |
| `d2l/d2l_lr_scheduler.txt` | https://d2l.ai/chapter_optimization/lr-scheduler.html |
| `d2l/d2l_lstm.txt` | https://d2l.ai/chapter_recurrent-modern/lstm.html |
| `d2l/d2l_maximum_likelihood.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html |
| `d2l/d2l_minibatch_sgd.txt` | https://d2l.ai/chapter_optimization/minibatch-sgd.html |
| `d2l/d2l_mlp.txt` | https://d2l.ai/chapter_multilayer-perceptrons/mlp.html |
| `d2l/d2l_momentum.txt` | https://d2l.ai/chapter_optimization/momentum.html |
| `d2l/d2l_naive_bayes.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/naive-bayes.html |
| `d2l/d2l_numerical_stability_and_init.txt` | https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html |
| `d2l/d2l_padding_and_strides.txt` | https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html |
| `d2l/d2l_pooling.txt` | https://d2l.ai/chapter_convolutional-neural-networks/pooling.html |
| `d2l/d2l_random_variables.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/random-variables.html |
| `d2l/d2l_resnet.txt` | https://d2l.ai/chapter_convolutional-modern/resnet.html |
| `d2l/d2l_rmsprop.txt` | https://d2l.ai/chapter_optimization/rmsprop.html |
| `d2l/d2l_rnn.txt` | https://d2l.ai/chapter_recurrent-neural-networks/rnn.html |
| `d2l/d2l_sgd.txt` | https://d2l.ai/chapter_optimization/sgd.html |
| `d2l/d2l_softmax_regression.txt` | https://d2l.ai/chapter_linear-classification/softmax-regression.html |
| `d2l/d2l_statistics.txt` | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/statistics.html |
| `d2l/d2l_weight_decay.txt` | https://d2l.ai/chapter_linear-regression/weight-decay.html |

## Web articles / blogs: `web/` (secondary sources; LaTeX preserved where the page used MathJax)

| File | Source |
|---|---|
| `web/dieleman2022_guidance.txt` | S. Dieleman, "Guidance: a cheat code for diffusion models" (2022): https://sander.ai/2022/05/26/guidance.html |
| `web/dieleman2023_perspectives.txt` | S. Dieleman, "Perspectives on diffusion" (2023): https://sander.ai/2023/07/20/perspectives.html |
| `web/distill2016_tsne.txt` | Wattenberg, Viégas & Johnson, "How to Use t-SNE Effectively", Distill (2016): https://distill.pub/2016/misread-tsne/ |
| `web/gao2024_diffusion_meets_flow_matching.txt` | Gao, Hoogeboom, Heek, De Bortoli, Murphy, Salimans, "Diffusion Meets Flow Matching: Two Sides of the Same Coin" (2024): https://diffusionflow.github.io/ |
| `web/karpathy2019_training_recipe.txt` | A. Karpathy, "A Recipe for Training Neural Networks" (2019): https://karpathy.github.io/2019/04/25/recipe/ |
| `web/olah2015_lstm.txt` | C. Olah, "Understanding LSTM Networks" (2015): https://colah.github.io/posts/2015-08-Understanding-LSTMs/ |
| `web/sapora_ml_interviews.txt` | S. Sapora, "ML interviews" (topic list origin): https://silviasapora.github.io/blog/ml-interviews.html |
| `web/song2021_score_blog.txt` | Y. Song, "Generative Modeling by Estimating Gradients of the Data Distribution" (2021): https://yang-song.net/blog/2021/score/ |
| `web/weng2017_gan.txt` | L. Weng, "From GAN to WGAN" (2017): https://lilianweng.github.io/posts/2017-08-20-gan/ |
| `web/weng2018_flows.txt` | L. Weng, "Flow-based Deep Generative Models" (2018): https://lilianweng.github.io/posts/2018-10-13-flow-models/ |
| `web/weng2018_vae.txt` | L. Weng, "From Autoencoder to Beta-VAE" (2018): https://lilianweng.github.io/posts/2018-08-12-vae/ |
| `web/weng2021_diffusion.txt` | L. Weng, "What are Diffusion Models?" (2021, updated): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ |

## Papers: `papers/` (page markers `=== [stem p.N] ===`)

| File | Title | Authors | Year | Canonical URL | Used by |
|---|---|---|---|---|---|
| `papers/albergo2023_stochastic_interpolants.txt` | Stochastic Interpolants: A Unifying Framework for Flows and Diffusions | Albergo, Boffi, Vanden-Eijnden | 2023 | https://arxiv.org/abs/2303.08797 | gen.flow-matching |
| `papers/arjovsky2017_principled_gan.txt` | Towards Principled Methods for Training GANs | Arjovsky, Bottou | 2017 | https://arxiv.org/abs/1701.04862 | gen.gans, fund.information-theory |
| `papers/arjovsky2017_wgan.txt` | Wasserstein GAN | Arjovsky, Chintala, Bottou | 2017 | https://arxiv.org/abs/1701.07875 | gen.gans, fund.information-theory |
| `papers/arthur2007_kmeanspp.txt` | k-means++: The Advantages of Careful Seeding | Arthur, Vassilvitskii | 2007 | https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf | fund.clustering |
| `papers/ba2016_layernorm.txt` | Layer Normalization | Ba, Kiros, Hinton | 2016 | https://arxiv.org/abs/1607.06450 | fund.normalization |
| `papers/baydin2018_autodiff.txt` | Automatic Differentiation in Machine Learning: a Survey | Baydin, Pearlmutter, Radul, Siskind | 2018 | https://arxiv.org/abs/1502.05767 | fund.backprop-mlp |
| `papers/belkin2019_double_descent.txt` | Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-off | Belkin, Hsu, Ma, Mandal | 2019 | https://arxiv.org/abs/1812.11118 | fund.bias-variance |
| `papers/bendavid2010_da_theory.txt` | A Theory of Learning from Different Domains | Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan | 2010 | https://www.alexkulesza.com/pubs/adapt_mlj10.pdf | fund.domain-adaptation |
| `papers/bengio2004_cv_variance.txt` | No Unbiased Estimator of the Variance of K-Fold Cross-Validation | Bengio, Grandvalet | 2004 | https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf | fund.model-selection |
| `papers/bengio2013_straight_through.txt` | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation | Bengio, Leonard, Courville | 2013 | https://arxiv.org/abs/1308.3432 | fund.gumbel-softmax |
| `papers/bottou2018_large_scale_opt.txt` | Optimization Methods for Large-Scale Machine Learning | Bottou, Curtis, Nocedal | 2018 | https://arxiv.org/abs/1606.04838 | fund.gradient-descent, fund.second-order |
| `papers/bowman2015_posterior_collapse.txt` | Generating Sentences from a Continuous Space | Bowman et al. | 2016 | https://arxiv.org/abs/1511.06349 | gen.vae |
| `papers/breiman1996_bagging.txt` | Bagging Predictors | Breiman | 1996 | https://www.stat.berkeley.edu/~breiman/bagging.pdf | fund.ensembles |
| `papers/breiman2001_random_forests.txt` | Random Forests | Breiman | 2001 | https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf | fund.ensembles |
| `papers/brown2020_gpt3.txt` | Language Models are Few-Shot Learners | Brown et al. | 2020 | https://arxiv.org/abs/2005.14165 | fund.transfer-learning |
| `papers/burda2015_iwae.txt` | Importance Weighted Autoencoders | Burda, Grosse, Salakhutdinov | 2016 | https://arxiv.org/abs/1509.00519 | gen.vae |
| `papers/burges1998_svm_tutorial.txt` | A Tutorial on Support Vector Machines for Pattern Recognition | Burges | 1998 | https://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf | fund.svm |
| `papers/carlini2023_diffusion_memorization.txt` | Extracting Training Data from Diffusion Models | Carlini et al. | 2023 | https://arxiv.org/abs/2301.13188 | gen.evaluation |
| `papers/cawley2010_model_selection_overfit.txt` | On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation | Cawley, Talbot | 2010 | https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf | fund.model-selection |
| `papers/chen2016_xgboost.txt` | XGBoost: A Scalable Tree Boosting System | Chen, Guestrin | 2016 | https://arxiv.org/abs/1603.02754 | fund.boosting |
| `papers/chen2018_neural_ode.txt` | Neural Ordinary Differential Equations | Chen, Rubanova, Bettencourt, Duvenaud | 2018 | https://arxiv.org/abs/1806.07366 | gen.flows |
| `papers/chong2020_unbiased_fid.txt` | Effectively Unbiased FID and Inception Score and Where to Find Them | Chong, Forsyth | 2020 | https://arxiv.org/abs/1911.07023 | gen.evaluation |
| `papers/chung2014_gru.txt` | Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | Chung, Gulcehre, Cho, Bengio | 2014 | https://arxiv.org/abs/1412.3555 | fund.rnn-lstm |
| `papers/cremer2018_amortization_gap.txt` | Inference Suboptimality in Variational Autoencoders | Cremer, Li, Duvenaud | 2018 | https://arxiv.org/abs/1801.03558 | gen.vae |
| `papers/dauphin2014_saddle_free.txt` | Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization | Dauphin et al. | 2014 | https://arxiv.org/abs/1406.2572 | fund.second-order |
| `papers/davis2006_pr_roc.txt` | The Relationship Between Precision-Recall and ROC Curves | Davis, Goadrich | 2006 | https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf | fund.classification-metrics |
| `papers/dhariwal2021_guided_diffusion.txt` | Diffusion Models Beat GANs on Image Synthesis | Dhariwal, Nichol | 2021 | https://arxiv.org/abs/2105.05233 | gen.guidance |
| `papers/dinh2016_realnvp.txt` | Density Estimation using Real NVP | Dinh, Sohl-Dickstein, Bengio | 2017 | https://arxiv.org/abs/1605.08803 | gen.flows |
| `papers/duchi2011_adagrad.txt` | Adaptive Subgradient Methods for Online Learning and Stochastic Optimization | Duchi, Hazan, Singer | 2011 | https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf | fund.adam |
| `papers/dumoulin2016_conv_arithmetic.txt` | A Guide to Convolution Arithmetic for Deep Learning | Dumoulin, Visin | 2016 | https://arxiv.org/abs/1603.07285 | fund.cnn |
| `papers/efron2011_tweedie.txt` | Tweedie's Formula and Selection Bias | Efron | 2011 | https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf | gen.score-based |
| `papers/esser2024_sd3_rectified_flow.txt` | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | Esser et al. | 2024 | https://arxiv.org/abs/2403.03206 | gen.flow-matching, gen.latent-diffusion |
| `papers/fawcett2006_roc_intro.txt` | An Introduction to ROC Analysis | Fawcett | 2006 | https://people.inf.elte.hu/kiss/11dwhdm/roc.pdf | fund.classification-metrics |
| `papers/finn2017_maml.txt` | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks | Finn, Abbeel, Levine | 2017 | https://arxiv.org/abs/1703.03400 | fund.transfer-learning |
| `papers/flach2015_prg.txt` | Precision-Recall-Gain Curves: PR Analysis Done Right | Flach, Kull | 2015 | https://papers.nips.cc/paper_files/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html | fund.classification-metrics |
| `papers/freund1999_boosting_intro.txt` | A Short Introduction to Boosting | Freund, Schapire | 1999 | https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf | fund.boosting |
| `papers/friedman2001_gbm.txt` | Greedy Function Approximation: A Gradient Boosting Machine | Friedman | 2001 | https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full | fund.boosting |
| `papers/ganin2016_dann.txt` | Domain-Adversarial Training of Neural Networks | Ganin et al. | 2016 | https://arxiv.org/abs/1505.07818 | fund.domain-adaptation |
| `papers/gers2002_lstm_peephole_timing.txt` | Learning Precise Timing with LSTM Recurrent Networks (peepholes; recaps forget gate) | Gers, Schraudolph, Schmidhuber | 2002 | https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf | fund.rnn-lstm |
| `papers/glorot2010_init.txt` | Understanding the Difficulty of Training Deep Feedforward Neural Networks | Glorot, Bengio | 2010 | https://proceedings.mlr.press/v9/glorot10a.html | fund.initialization |
| `papers/glorot2011_relu.txt` | Deep Sparse Rectifier Neural Networks | Glorot, Bordes, Bengio | 2011 | https://proceedings.mlr.press/v15/glorot11a.html | fund.activations |
| `papers/goodfellow2014_gan.txt` | Generative Adversarial Nets | Goodfellow et al. | 2014 | https://arxiv.org/abs/1406.2661 | gen.gans, fund.information-theory |
| `papers/goodfellow2016_gan_tutorial.txt` | NIPS 2016 Tutorial: Generative Adversarial Networks | Goodfellow | 2016 | https://arxiv.org/abs/1701.00160 | gen.gans, gen.overview |
| `papers/goyal2017_large_minibatch.txt` | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | Goyal et al. | 2017 | https://arxiv.org/abs/1706.02677 | fund.gradient-descent, fund.backprop-mlp |
| `papers/grathwohl2018_ffjord.txt` | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models | Grathwohl et al. | 2019 | https://arxiv.org/abs/1810.01367 | gen.flows |
| `papers/greff2015_lstm_odyssey.txt` | LSTM: A Search Space Odyssey | Greff et al. | 2017 | https://arxiv.org/abs/1503.04069 | fund.rnn-lstm |
| `papers/gulrajani2017_wgan_gp.txt` | Improved Training of Wasserstein GANs | Gulrajani et al. | 2017 | https://arxiv.org/abs/1704.00028 | gen.gans |
| `papers/guo2017_calibration.txt` | On Calibration of Modern Neural Networks | Guo, Pleiss, Sun, Weinberger | 2017 | https://arxiv.org/abs/1706.04599 | fund.loss-functions, fund.classification-metrics |
| `papers/he2015_prelu_init.txt` | Delving Deep into Rectifiers (He/Kaiming init, PReLU) | He, Zhang, Ren, Sun | 2015 | https://arxiv.org/abs/1502.01852 | fund.initialization, fund.activations |
| `papers/he2016_resnet.txt` | Deep Residual Learning for Image Recognition | He, Zhang, Ren, Sun | 2016 | https://arxiv.org/abs/1512.03385 | fund.cnn |
| `papers/hendrycks2016_gelu.txt` | Gaussian Error Linear Units (GELUs) | Hendrycks, Gimpel | 2016 | https://arxiv.org/abs/1606.08415 | fund.activations |
| `papers/heusel2017_fid.txt` | GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID) | Heusel et al. | 2017 | https://arxiv.org/abs/1706.08500 | gen.evaluation |
| `papers/higgins2017_beta_vae.txt` | beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework | Higgins et al. | 2017 | https://openreview.net/forum?id=Sy2fzU9gl | gen.vae |
| `papers/ho2020_ddpm.txt` | Denoising Diffusion Probabilistic Models | Ho, Jain, Abbeel | 2020 | https://arxiv.org/abs/2006.11239 | gen.ddpm |
| `papers/ho2022_cfg.txt` | Classifier-Free Diffusion Guidance | Ho, Salimans | 2022 | https://arxiv.org/abs/2207.12598 | gen.guidance |
| `papers/hochreiter1997_lstm.txt` | Long Short-Term Memory | Hochreiter, Schmidhuber | 1997 | https://www.bioinf.jku.at/publications/older/2604.pdf | fund.rnn-lstm |
| `papers/huijben2021_gumbel_review.txt` | A Review of the Gumbel-max Trick and its Extensions for Discrete Stochasticity in Machine Learning | Huijben, Kool, Paulus, van Sloun | 2022 | https://arxiv.org/abs/2110.01515 | fund.gumbel-softmax |
| `papers/hyvarinen2005_score_matching.txt` | Estimation of Non-Normalized Statistical Models by Score Matching | Hyvarinen | 2005 | https://jmlr.org/papers/v6/hyvarinen05a.html | gen.score-based |
| `papers/ioffe2015_batchnorm.txt` | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | Ioffe, Szegedy | 2015 | https://arxiv.org/abs/1502.03167 | fund.normalization |
| `papers/jang2016_gumbel_softmax.txt` | Categorical Reparameterization with Gumbel-Softmax | Jang, Gu, Poole | 2017 | https://arxiv.org/abs/1611.01144 | fund.gumbel-softmax |
| `papers/jozefowicz2015_rnn_empirical.txt` | An Empirical Exploration of Recurrent Network Architectures | Jozefowicz, Zaremba, Sutskever | 2015 | https://proceedings.mlr.press/v37/jozefowicz15.html | fund.rnn-lstm |
| `papers/karras2022_edm.txt` | Elucidating the Design Space of Diffusion-Based Generative Models (EDM) | Karras, Aittala, Aila, Laine | 2022 | https://arxiv.org/abs/2206.00364 | gen.fast-sampling, gen.ddpm |
| `papers/kessy2018_whitening.txt` | Optimal Whitening and Decorrelation | Kessy, Lewin, Strimmer | 2018 | https://arxiv.org/abs/1512.00809 | fund.dim-reduction |
| `papers/kingma2013_vae.txt` | Auto-Encoding Variational Bayes | Kingma, Welling | 2014 | https://arxiv.org/abs/1312.6114 | gen.vae, fund.autoencoders, fund.gumbel-softmax |
| `papers/kingma2014_adam.txt` | Adam: A Method for Stochastic Optimization | Kingma, Ba | 2015 | https://arxiv.org/abs/1412.6980 | fund.adam |
| `papers/kingma2016_iaf.txt` | Improving Variational Inference with Inverse Autoregressive Flow | Kingma et al. | 2016 | https://arxiv.org/abs/1606.04934 | gen.flows |
| `papers/kingma2018_glow.txt` | Glow: Generative Flow with Invertible 1x1 Convolutions | Kingma, Dhariwal | 2018 | https://arxiv.org/abs/1807.03039 | gen.flows |
| `papers/kingma2019_vae_intro.txt` | An Introduction to Variational Autoencoders | Kingma, Welling | 2019 | https://arxiv.org/abs/1906.02691 | gen.vae, fund.autoencoders |
| `papers/kingma2021_vdm.txt` | Variational Diffusion Models | Kingma, Salimans, Poole, Ho | 2021 | https://arxiv.org/abs/2107.00630 | gen.ddpm |
| `papers/kingma2023_diffusion_objectives.txt` | Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation | Kingma, Gao | 2023 | https://arxiv.org/abs/2303.00848 | gen.ddpm, gen.flow-matching |
| `papers/kumar2022_lpft.txt` | Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution | Kumar et al. | 2022 | https://arxiv.org/abs/2202.10054 | fund.transfer-learning |
| `papers/kunstner2019_empirical_fisher.txt` | Limitations of the Empirical Fisher Approximation for Natural Gradient Descent | Kunstner, Balles, Hennig | 2019 | https://arxiv.org/abs/1905.12558 | fund.second-order |
| `papers/kynkaanniemi2019_gen_precision_recall.txt` | Improved Precision and Recall Metric for Assessing Generative Models | Kynkaanniemi et al. | 2019 | https://arxiv.org/abs/1904.06991 | gen.evaluation |
| `papers/kynkaanniemi2024_guidance_interval.txt` | Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models | Kynkäänniemi, Aittala, Karras, Laine, Aila, Lehtinen | 2024 | https://arxiv.org/abs/2404.07724 | gen.guidance |
| `papers/lakshminarayanan2017_deep_ensembles.txt` | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | Lakshminarayanan, Pritzel, Blundell | 2017 | https://arxiv.org/abs/1612.01474 | fund.ensembles |
| `papers/lin2017_focal_loss.txt` | Focal Loss for Dense Object Detection | Lin, Goyal, Girshick, He, Dollar | 2017 | https://arxiv.org/abs/1708.02002 | fund.loss-functions |
| `papers/lin2024_noise_schedules_flawed.txt` | Common Diffusion Noise Schedules and Sample Steps are Flawed (zero terminal SNR, CFG rescale) | Lin, Liu, Li, Yang | 2024 | https://arxiv.org/abs/2305.08891 | gen.diffusion-parameterizations, gen.guidance |
| `papers/lipman2023_flow_matching.txt` | Flow Matching for Generative Modeling | Lipman et al. | 2023 | https://arxiv.org/abs/2210.02747 | gen.flow-matching |
| `papers/lipton2018_label_shift.txt` | Detecting and Correcting for Label Shift with Black Box Predictors | Lipton, Wang, Smola | 2018 | https://arxiv.org/abs/1802.03916 | fund.domain-adaptation |
| `papers/liu2022_rectified_flow.txt` | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | Liu, Gong, Liu | 2023 | https://arxiv.org/abs/2209.03003 | gen.flow-matching |
| `papers/loshchilov2017_adamw.txt` | Decoupled Weight Decay Regularization (AdamW) | Loshchilov, Hutter | 2019 | https://arxiv.org/abs/1711.05101 | fund.adam, fund.regularization |
| `papers/lu2022_dpm_solver.txt` | DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling | Lu et al. | 2022 | https://arxiv.org/abs/2206.00927 | gen.fast-sampling |
| `papers/luo2022_diffusion_unified.txt` | Understanding Diffusion Models: A Unified Perspective | Luo | 2022 | https://arxiv.org/abs/2208.11970 | gen.ddpm, gen.score-based, gen.vae |
| `papers/maddison2016_concrete.txt` | The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables | Maddison, Mnih, Teh | 2017 | https://arxiv.org/abs/1611.00712 | fund.gumbel-softmax |
| `papers/martens2014_natural_gradient.txt` | New Insights and Perspectives on the Natural Gradient Method | Martens | 2020 | https://arxiv.org/abs/1412.1193 | fund.second-order |
| `papers/martens2015_kfac.txt` | Optimizing Neural Networks with Kronecker-factored Approximate Curvature | Martens, Grosse | 2015 | https://arxiv.org/abs/1503.05671 | fund.second-order |
| `papers/mcinnes2018_umap.txt` | UMAP: Uniform Manifold Approximation and Projection | McInnes, Healy, Melville | 2018 | https://arxiv.org/abs/1802.03426 | fund.dim-reduction |
| `papers/mescheder2018_gan_convergence_r1.txt` | Which Training Methods for GANs do actually Converge? (R1/R2 penalties, Dirac-GAN) | Mescheder, Geiger, Nowozin | 2018 | https://arxiv.org/abs/1801.04406 | gen.gans, gen.wgan |
| `papers/mirza2014_cgan.txt` | Conditional Generative Adversarial Nets | Mirza, Osindero | 2014 | https://arxiv.org/abs/1411.1784 | gen.gans |
| `papers/miyato2018_spectral_norm.txt` | Spectral Normalization for Generative Adversarial Networks | Miyato et al. | 2018 | https://arxiv.org/abs/1802.05957 | gen.gans |
| `papers/muller2019_label_smoothing.txt` | When Does Label Smoothing Help? | Muller, Kornblith, Hinton | 2019 | https://arxiv.org/abs/1906.02629 | fund.loss-functions |
| `papers/nakkiran2019_deep_double_descent.txt` | Deep Double Descent: Where Bigger Models and More Data Hurt | Nakkiran et al. | 2019 | https://arxiv.org/abs/1912.02292 | fund.bias-variance |
| `papers/neal2018_modern_bias_variance.txt` | A Modern Take on the Bias-Variance Tradeoff in Neural Networks | Neal et al. | 2018 | https://arxiv.org/abs/1810.08591 | fund.bias-variance |
| `papers/nichol2021_improved_ddpm.txt` | Improved Denoising Diffusion Probabilistic Models | Nichol, Dhariwal | 2021 | https://arxiv.org/abs/2102.09672 | gen.ddpm |
| `papers/nowozin2016_fgan.txt` | f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization | Nowozin, Cseke, Tomioka | 2016 | https://arxiv.org/abs/1606.00709 | gen.gans, fund.information-theory |
| `papers/papamakarios2017_maf.txt` | Masked Autoregressive Flow for Density Estimation | Papamakarios, Pavlakou, Murray | 2017 | https://arxiv.org/abs/1705.07057 | gen.flows |
| `papers/papamakarios2021_flows_review.txt` | Normalizing Flows for Probabilistic Modeling and Inference | Papamakarios et al. | 2021 | https://arxiv.org/abs/1912.02762 | gen.flows |
| `papers/pascanu2013_rnn_difficulty.txt` | On the Difficulty of Training Recurrent Neural Networks | Pascanu, Mikolov, Bengio | 2013 | https://arxiv.org/abs/1211.5063 | fund.rnn-lstm |
| `papers/peebles2023_dit.txt` | Scalable Diffusion Models with Transformers (DiT) | Peebles, Xie | 2023 | https://arxiv.org/abs/2212.09748 | gen.diffusion-transformers |
| `papers/radford2021_clip.txt` | Learning Transferable Visual Models From Natural Language Supervision (CLIP) | Radford et al. | 2021 | https://arxiv.org/abs/2103.00020 | fund.transfer-learning |
| `papers/ramachandran2017_swish.txt` | Searching for Activation Functions (Swish) | Ramachandran, Zoph, Le | 2017 | https://arxiv.org/abs/1710.05941 | fund.activations |
| `papers/reddi2018_amsgrad.txt` | On the Convergence of Adam and Beyond (AMSGrad) | Reddi, Kale, Kumar | 2018 | https://arxiv.org/abs/1904.09237 | fund.adam |
| `papers/rombach2022_ldm.txt` | High-Resolution Image Synthesis with Latent Diffusion Models | Rombach et al. | 2022 | https://arxiv.org/abs/2112.10752 | gen.latent-diffusion |
| `papers/ruder2016_gd_overview.txt` | An Overview of Gradient Descent Optimization Algorithms | Ruder | 2016 | https://arxiv.org/abs/1609.04747 | fund.gradient-descent, fund.adam |
| `papers/saharia2022_imagen.txt` | Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen; dynamic thresholding) | Saharia et al. | 2022 | https://arxiv.org/abs/2205.11487 | gen.guidance |
| `papers/salimans2016_improved_gan_is.txt` | Improved Techniques for Training GANs (Inception Score) | Salimans et al. | 2016 | https://arxiv.org/abs/1606.03498 | gen.evaluation, gen.gans |
| `papers/salimans2022_progressive_distillation.txt` | Progressive Distillation for Fast Sampling of Diffusion Models (v-prediction) | Salimans, Ho | 2022 | https://arxiv.org/abs/2202.00512 | gen.fast-sampling, gen.ddpm |
| `papers/santurkar2018_bn_help.txt` | How Does Batch Normalization Help Optimization? | Santurkar, Tsipras, Ilyas, Madry | 2018 | https://arxiv.org/abs/1805.11604 | fund.normalization |
| `papers/saxe2013_orthogonal.txt` | Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks | Saxe, McClelland, Ganguli | 2014 | https://arxiv.org/abs/1312.6120 | fund.initialization |
| `papers/shazeer2020_glu.txt` | GLU Variants Improve Transformer | Shazeer | 2020 | https://arxiv.org/abs/2002.05202 | fund.activations |
| `papers/shlens2014_pca_tutorial.txt` | A Tutorial on Principal Component Analysis | Shlens | 2014 | https://arxiv.org/abs/1404.1100 | fund.dim-reduction |
| `papers/snell2017_protonets.txt` | Prototypical Networks for Few-shot Learning | Snell, Swersky, Zemel | 2017 | https://arxiv.org/abs/1703.05175 | fund.transfer-learning |
| `papers/sohldickstein2015_diffusion.txt` | Deep Unsupervised Learning using Nonequilibrium Thermodynamics | Sohl-Dickstein et al. | 2015 | https://arxiv.org/abs/1503.03585 | gen.ddpm |
| `papers/song2019_ncsn.txt` | Generative Modeling by Estimating Gradients of the Data Distribution (NCSN) | Song, Ermon | 2019 | https://arxiv.org/abs/1907.05600 | gen.score-based |
| `papers/song2020_ddim.txt` | Denoising Diffusion Implicit Models | Song, Meng, Ermon | 2021 | https://arxiv.org/abs/2010.02502 | gen.fast-sampling |
| `papers/song2021_sde.txt` | Score-Based Generative Modeling through Stochastic Differential Equations | Song et al. | 2021 | https://arxiv.org/abs/2011.13456 | gen.score-based |
| `papers/song2021_train_ebm.txt` | How to Train Your Energy-Based Models | Song, Kingma | 2021 | https://arxiv.org/abs/2101.03288 | gen.score-based, gen.overview |
| `papers/song2023_consistency.txt` | Consistency Models | Song, Dhariwal, Chen, Sutskever | 2023 | https://arxiv.org/abs/2303.01469 | gen.fast-sampling |
| `papers/srivastava2014_dropout.txt` | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Srivastava et al. | 2014 | https://jmlr.org/papers/v15/srivastava14a.html | fund.regularization |
| `papers/sutskever2013_momentum.txt` | On the Importance of Initialization and Momentum in Deep Learning | Sutskever, Martens, Dahl, Hinton | 2013 | https://proceedings.mlr.press/v28/sutskever13.html | fund.gradient-descent |
| `papers/theis2016_note_on_evaluation.txt` | A Note on the Evaluation of Generative Models | Theis, van den Oord, Bethge | 2016 | https://arxiv.org/abs/1511.01844 | gen.evaluation |
| `papers/tian2020_good_embedding_fewshot.txt` | Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need? | Tian et al. | 2020 | https://arxiv.org/abs/2003.11539 | fund.transfer-learning |
| `papers/tong2023_ot_cfm.txt` | Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport | Tong et al. | 2024 | https://arxiv.org/abs/2302.00482 | gen.flow-matching |
| `papers/vandenoord2016_pixelcnn_decoders.txt` | Conditional Image Generation with PixelCNN Decoders | van den Oord et al. | 2016 | https://arxiv.org/abs/1606.05328 | gen.autoregressive |
| `papers/vandenoord2016_pixelrnn.txt` | Pixel Recurrent Neural Networks | van den Oord, Kalchbrenner, Kavukcuoglu | 2016 | https://arxiv.org/abs/1601.06759 | gen.autoregressive |
| `papers/vandenoord2016_wavenet.txt` | WaveNet: A Generative Model for Raw Audio | van den Oord et al. | 2016 | https://arxiv.org/abs/1609.03499 | gen.autoregressive |
| `papers/vandenoord2017_vqvae.txt` | Neural Discrete Representation Learning (VQ-VAE) | van den Oord, Vinyals, Kavukcuoglu | 2017 | https://arxiv.org/abs/1711.00937 | gen.vae |
| `papers/vandermaaten2008_tsne.txt` | Visualizing Data using t-SNE | van der Maaten, Hinton | 2008 | https://www.jmlr.org/papers/v9/vandermaaten08a.html | fund.dim-reduction |
| `papers/vincent2010_denoising_ae.txt` | Stacked Denoising Autoencoders | Vincent et al. | 2010 | https://jmlr.org/papers/v11/vincent10a.html | fund.autoencoders |
| `papers/vincent2011_dsm.txt` | A Connection Between Score Matching and Denoising Autoencoders | Vincent | 2011 | https://web.archive.org/web/2020/http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf | gen.score-based, fund.autoencoders |
| `papers/wilson2017_adaptive_marginal.txt` | The Marginal Value of Adaptive Gradient Methods in Machine Learning | Wilson et al. | 2017 | https://arxiv.org/abs/1705.08292 | fund.adam |
| `papers/wu2018_groupnorm.txt` | Group Normalization | Wu, He | 2018 | https://arxiv.org/abs/1803.08494 | fund.normalization |
| `papers/xiong2020_preln.txt` | On Layer Normalization in the Transformer Architecture (Pre-LN) | Xiong et al. | 2020 | https://arxiv.org/abs/2002.04745 | fund.normalization |
| `papers/yosinski2014_transferable.txt` | How Transferable are Features in Deep Neural Networks? | Yosinski, Clune, Bengio, Lipson | 2014 | https://arxiv.org/abs/1411.1792 | fund.transfer-learning |
| `papers/zhang2017_rethinking_generalization.txt` | Understanding Deep Learning Requires Rethinking Generalization | Zhang, Bengio, Hardt, Recht, Vinyals | 2017 | https://arxiv.org/abs/1611.03530 | fund.bias-variance, fund.regularization |
| `papers/zhang2019_rmsnorm.txt` | Root Mean Square Layer Normalization | Zhang, Sennrich | 2019 | https://arxiv.org/abs/1910.07467 | fund.normalization |

## Not cached (tried and failed, or deliberately skipped)

- Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024): bishopbook.com only offers an in-browser viewer, with no extractable text or PDF. Skipped.
- Wolpert (1996) NFL paper, Cover & Hart (1967) kNN, Amari (1998) natural gradient, Shimodaira (2000) covariate shift, Lin (1991) JS divergence, Hinton & Salakhutdinov (2006): paywalled. Their results are covered in UML ch.5, ESL 13.3, Martens 2014, PML2 19.5.2, PML2 2.7 / Goodfellow 2014, and DLB 14.
- Freund & Schapire (1997) JCSS: paywalled. Use `freund1999_boosting_intro`, ESL ch.10 and UML ch.10.
- Cover & Thomas, Wasserman *All of Statistics*: not free. Use MacKay, PML1 ch.4/6, d2l statistics/information-theory appendices.

## Grep recipes

```
grep -n '^8.5' dlb_ch08_optimization.txt              # DLB section start
grep -n -A1 '^7.10.2' esl.txt                           # ESL: number then title
sed -n '/=== \[bishop pdf p.444\]/,/=== \[bishop pdf p.446\]/p' bishop.txt   # print pages
grep -n -i 'gumbel' pml2.txt | head                      # concept search
grep -rn -i 'v-prediction\|v prediction' papers/         # across all papers
```

