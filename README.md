# A Survey on Biomedical Text Summarisation with Pre-trained Language Model(PLM)s

![](https://img.shields.io/badge/Status-building-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) 


# Resource
This repository contains a list of papers, codes, and datasets in Biomedical Text Summarisation based on PLM. If you found any errors, please don't hesitate to open an issue or pull a request.

If you find this repository helpful for your work,  please consider citing our survey paper. The Bibtex are listed below:
<pre>

</pre>

## Contributor


Resource Contributed by [Qianqian Xie](), [Zheheng Luo](),  [Jiamin Huang](), [Hua Wang](),[Sophia Ananiadou](https://www.research.manchester.ac.uk/portal/sophia.ananiadou.html).

## Introduction

Biomedical text summarization has long been a fundamental task in biomedical natural language processing (BioNLP),
aiming at generating concise summaries that distill key information from one or multiple biomedical documents. In recent years,
pre-trained language models (PLMs) have been the de facto standard of various natural language processing tasks in the general
domain. Most recently, PLMs have been further investigated in the biomedical domain and brought new insights into the biomedical text
summarization task. 

To help researchers quickly grasp the development in this task and inspire further research, we line up available datasets, recent approaches and evaluation methods in this project.

At present, the project has been completely open source, including:

1. **BioTS dataset table:** we listed the datasets in the BioTS field, You can find the category, size, content, and access of them in the table.
2. **PLM Based BioTS Methods:** we classified and arranged papers based on the type of output summary, numbers and type of input documents. the current mainstream frontiers. Each line of the table contains the category, the strategy of applying PLM, the backbone model, the training type, and used datasets.

The organization and our survey and the detailed background of biomedical text summarization are illustrated in the pictures below.


![joint-compared-framework](./pics/OverviewOfBiomedicalTextSummarizationWithPLM.png)


![SLUs-taxonomy](./pics/TaxonomyOfMethods.png)


## Quick path
- [Resources](#resources)
  * [survey paper links](#survey-paper-links)
  * [recent open-sourced code](#recent-open-sourced-code)
  * [Single Model](#single-model)
  * [Joint Model](#joint-model)
  * [Complex SLU Model](#complex-slu-model)
- [Dataset](#dataset)
- [Frontiers](#frontiers)
  * [Single Slot Filling](#single-slot-filling)
  * [Single Intent Detection](#single-intent-detection)
  * [Joint Model](#joint-model-1)
    + [Implicit joint modeling](#implicit-joint-modeling)
    + [Explicit joint modeling](#explicit-joint-modeling)
  * [Contextual SLU](#contextual-slu)
  * [Multi-intent SLU](#multi-intent-slu)
  * [Chinese SLU](#chinese-slu)
  * [Cross-domain SLU](#cross-domain-slu)
  * [Cross-lingual SLU](#cross-lingual-slu)
  * [Low-resource SLU](#low-resource-slu)
    + [Few-shot SLU](#few-shot-slu)
    + [Zero-shot SLU](#zero-shot-slu)
    + [Unsupervised SLU](#unsupervised-slu)
- [LeaderBoard](#leaderboard)
  * [ATIS](#atis)
    + [Non-pretrained model](#non-pretrained-model)
    + [+ Pretrained model](#--pretrained-model)
  * [SNIPS](#snips)
    + [Non-pretrained model](#non-pretrained-model-1)
    + [+ Pretrained model](#--pretrained-model-1)

## Resources
### survey paper links

1. **A Survey on Spoken Language Understanding: Recent Advances and New Frontiers** `arxiv` [[pdf]](https://arxiv.org/pdf/2103.03095.pdf)
2. **Spoken language understanding: Systems for extracting semantic information from speech** `book` [[pdf]](https://ieeexplore.ieee.org/book/8042134)
3. **Recent Neural Methods on Slot Filling and Intent Classification**  `COLING 2020` [[pdf]](https://www.aclweb.org/anthology/2020.coling-main.42.pdf) 
4. **A survey of joint intent detection and slot-filling models in natural language understanding**  `arxiv 2021` [[pdf]](https://arxiv.org/pdf/2101.08091.pdf) 

### recent open-sourced code

### Single Model

1. **Few-shot Slot Tagging with  Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection  Network** (SNIPS) `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging) 
2. **Sequence-to-Sequence Data  Augmentation for Dialogue Language Understanding** (ATIS/Stanford Dialogue Dataset) `COLING 2018` [[pdf]](https://arxiv.org/pdf/1807.01554.pdf) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) 

### Joint Model

1. **A Co-Interactive Transformer for Joint Slot Filling and Intent Detection**(ATIS/SNIPS) `ICASSP 2021` [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) [[code]](https://github.com/kangbrilliant/DCA-Net)
2. **SlotRefine: A Fast Non-Autoregressive Model for Joint Intent Detection and Slot Filling** (ATIS/SNIPS) `EMNLP 2020` [[pdf]](https://www.aclweb.org/anthology/2020.emnlp-main.152.pdf) [[code]](https://github.com/moore3930/SlotRefine)
3. **Joint Slot Filling and Intent  Detection via Capsule Neural Networks** (ATIS/SNIPS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) [[code]](https://github.com/czhang99/Capsule-NLU) 
4. **BERT for Joint Intent  Classification and Slot Filling** (ATIS/SNIPS/Stanford Dialogue Dataset) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) [[code]](https://github.com/monologg/JointBERT) 
5. **A Novel Bi-directional  Interrelated Model for Joint Intent Detection and Slot Filling** (ATIS/Stanford Dialogue Dataset/SNIPS) `ACL 2019` [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) [[code]](https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU) 
6. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 
7. **Slot-Gated Modeling for Joint  Slot Filling and Intent Prediction** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) [[code]](https://github.com/MiuLab/SlotGated-SLU) 
8. **Joint Online Spoken Language  Understanding and Language Modeling with Recurrent Neural Networks** (ATIS) `SIGDIAL 2016` [[pdf]](https://www.aclweb.org/anthology/W16-3603.pdf) [[code]](https://github.com/HadoopIt/joint-slu-lm)

### Complex SLU Model

1. **How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues** (DSTC4) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-1194.pdf) [[code]](https://github.com/MiuLab/Time-Decay-SLU) 
2. **Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning** (DSTC4) `IJCNLP 2017` [[pdf]](https://www.aclweb.org/anthology/I17-2028.pdf) [[code]](https://github.com/MiuLab/Spk-Dialogue) 
3. **Dynamic time-aware attention to speaker roles and contexts for spoken language understanding** (DSTC4) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1710.00165.pdf) [[code]](https://github.com/MiuLab/Time-SLU) 
4. **Injecting Word Information with Multi-Level Word Adapter for Chinese Spoken Language Understanding** (CAIS/ECDT-NLU) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.03903.pdf) [[code]](https://github.com/AaronTengDeChuan/MLWA-Chinese-SLU) 
5. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 
6. **Coach: A Coarse-to-Fine  Approach for Cross-domain Slot Filling** (SNIPS) `ACL 2020` [[pdf]](https://arxiv.org/pdf/2004.11727.pdf) [[code]](https://github.com/zliucr/coach)
7. **CoSDA-ML: Multi-Lingual  Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP** (SC2/4/MLDoc/Multi WOZ/Facebook Multilingual SLU Dataset) `IJCAI 2020` [[pdf]](https://arxiv.org/pdf/2006.06402.pdf) [[code]](https://github.com/kodenii/CoSDA-ML) 
8. **Cross-lingual Spoken Language  Understanding with Regularized Representation Alignment** (Multilingual spoken language understanding (SLU) dataset) `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.14510.pdf) [[code]](https://github.com/zliucr/crosslingual-slu.)
9. **Attention-Informed  Mixed-Language Training for Zero-shot Cross-lingual Task-oriented Dialogue  Systems** (Facebook Multilingual SLU Dataset/(DST)MultiWOZ) `AAAI 2020` [[pdf]](https://arxiv.org/pdf/1911.09273.pdf) [[code]](https://github.com/zliucr/mixedlanguage-training) 
10. **MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark** (MTOP/Multilingual ATIS) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2008.09335.pdf) [[code]]() 
11. **Neural Architectures for  Multilingual Semantic Parsing** (GEO/ATIS) `ACL 2017` [[pdf]](https://www.aclweb.org/anthology/P17-2007.pdf) [[code]](http://statnlp.org/research/sp/) 
12. **Few-shot Learning for Multi-label Intent Detection** (TourSG/StandfordLU) `AAAI 2021` [[pdf]](https://arxiv.org/abs/2010.05256.pdf) [[code]](https://github.com/AtmaHou/FewShotMultiLabel) 
13. **Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network** (SNIPS and further construct) `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging)


## Dataset
<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Name</th>
    <th>Category</th>
    <th>Size</th>
    <th>Content</th>
    <th>Multi/Single Sum(M/S)</th>
    <th>Access</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> PubMed </td></code>
		<td> Biomedical literature </td>
		<td> 133,215 </td>
		<td> Full contents of articles</td>
		<td> Single </td>
		<td> <a href="https://github.com/armancohan/long-summarization">https://github.com/armancohan/long-summarization</a></td>

<tr>
	<td><code> RCT                              </td></code>
		<td> Biomedical literature </td>
		<td> 4,528 </td>
		<td> Titles and abstracts of articles</td>
		<td> Multiple </td>
		<td> <a href="https://github.com/bwallace/RCT-summarization-data">https://github.com/bwallace/RCT-summarization-data</a></td>
<tr>
	<td><code> MSˆ2                               </td></code>
		<td> Biomedical literature </td>
		<td> 470,402 </td>
		<td> Abstracts of articles</td>
		<td> Multiple </td>
		<td> <a href="https://github.com/allenai/ms2/">https://github.com/allenai/ms2/</a></td>
<tr>
	<td><code> CDSR                               </td></code>
		<td> Biomedical literature </td>
		<td> 7,805 </td>
		<td> Abstracts of articles</td>
		<td> Single </td>
		<td> <a href="https://github.com/qiuweipku/Plain language summarization">https://github.com/qiuweipku/Plain language summarization</a></td>
<tr>
	<td><code> SumPubMed                               </td></code>
		<td> Biomedical literature </td>
		<td> 33,772 </td>
		<td> Full contents of articles</td>
		<td> Single </td>
		<td> <a href="https://github.com/vgupta123/sumpubmed<">https://github.com/vgupta123/sumpubmed</a></td>
<tr>
	<td><code>S2ORC                              </td></code>
		<td> Biomedical literature </td>
		<td> 63,709 </td>
		<td> Full contents of articles </td>
		<td> Single </td>
		<td> <a href="https://github.com/jbshp/GenCompareSum<">https://github.com/jbshp/GenCompareSum</a></td>
<tr>
	<td><code> CORD-19                               </td></code>
		<td> Biomedical literature </td>
		<td> - (constantly increasing)</td>
		<td> Full contents of articles</td>
		<td> Single </td>
		<td> <a href="https://github.com/allenai/cord19<">https://github.com/allenai/cord19</a></td>
<tr>
	<td><code> MIMIC-CXR                              </td></code>
		<td> EHR</td>
		<td> 124577</td>
		<td> Full contents of reports</td>
		<td> Single </td>
		<td> <a href="https://physionet.org/content/mimic-cxr/2.0.0/<">https://physionet.org/content/mimic-cxr/2.0.0/</a></td>
<tr>
	<td><code> OpenI                              </td></code>
		<td> EHR</td>
		<td> 3599</td>
		<td> Full contents of reports</td>
		<td> Single </td>
		<td> <a href="https://openi.nlm.nih.gov/faq#collection<">https://openi.nlm.nih.gov/faq#collection</a></td>
<tr>
	<td><code> MeQSum                              </td></code>
		<td> meidical question summarization</td>
		<td> 1000</td>
		<td> Full contents of question</td>
		<td> Single </td>
		<td> <a href="https://github.com/abachaa/MeQSum<">https://github.com/abachaa/MeQSum/</a></td>
<tr>
	<td><code> CHQ-Summ                               </td></code>
		<td> meidical question summarization</td>
		<td> 1507</td>
		<td> Full contents of question</td>
		<td> Single </td>
		<td> <a href="https://github.com/shwetanlp/Yahoo-CHQ-Summ<">https://github.com/shwetanlp/Yahoo-CHQ-Summ</a></td>
<tr>
</tbody >
</table>
</div>


## Frontiers

### Single Slot Filling

1. **Few-shot Slot Tagging with  Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection  Network** (SNIPS) `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging) 
2. **A Hierarchical Decoding Model  For Spoken Language Understanding From Unaligned Data** (DSTC2) `ICASSP 2019` [[pdf]](https://arxiv.org/pdf/1904.04498.pdf) 
3. **Utterance Generation With  Variational Auto-Encoder for Slot Filling in Spoken Language Understanding** (ATIS/SNIPS/MIT Corpus) `IEEE Signal Processing Letters 2019` [[pdf]]([https://ieeexplore.ieee.org/document/8625384](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8625384)) 
4. **Data Augmentation with Atomic  Templates for Spoken Language Understanding** (ATIS) `EMNLP 2019` [[pdf]](https://arxiv.org/pdf/1908.10770.pdf) 
5. **A New Concept of Deep  Reinforcement Learning based Augmented General Sequence Tagging System** (ATIS/CNLL-2003) `COLING 2018` [[pdf]](https://www.aclweb.org/anthology/C18-1143.pdf) 
6. **Improving Slot Filling in  Spoken Language Understanding with Joint Pointer and Attention** (DSTC2) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/P18-2068.pdf) 
7. **Sequence-to-Sequence Data  Augmentation for Dialogue Language Understanding** (ATIS/Stanford Dialogue Dataset) `COLING 2018` [[pdf]](https://arxiv.org/pdf/1807.01554.pdf) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) 
8. **Encoder-Decoder with  Focus-Mechanism for Sequence Labelling Based Spoken Language Understanding** (ATIS) `ICASSP 2017` [[pdf]]([https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=79532433](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7953243)) 
9. **Neural Models for Sequence  Chunking** (ATIS/LARGE) `AAAI 2017` [[pdf]](https://arxiv.org/pdf/1701.04027.pdf) 
10. **Bi-directional recurrent  neural network with ranking loss for spoken language understanding** (ATIS) `IEEE 2016` [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) 
11. **Labeled Data Generation with  Encoder-decoder LSTM for Semantic Slot Filling** (ATIS) `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/7ffe/83d7dd3a474e15ccc2aef412009f100a5802.pdf) 
12. **Syntax or Semantics?  Knowledge-Guided Joint Semantic Frame Parsing** (ATIS/Cortana) `IEEE Workshop on Spoken Language Technology 2016` [[pdf]](https://www.csie.ntu.edu.tw/~yvchen/doc/SLT16_SyntaxSemantics.pdf) 
13. **Bi-Directional Recurrent  Neural Network with Ranking Loss for Spoken Language Understanding** (ATIS) `ICASSP 2016` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7472841) 
14. **Leveraging Sentence-level  Information with Encoder LSTM for Semantic Slot Filling** (ATIS) `EMNLP 2016` [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) 
15. **Labeled Data Generation with  Encoder-decoder LSTM for Semantic Slot Filling** (ATIS) `INTERSPEECH 2016` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0727.PDF) 
16. **Using Recurrent Neural  Networks for Slot Filling in Spoken Language Understanding** (ATIS) `IEEE/ACM TASLP 2015` [[pdf]](https://ieeexplore.ieee.org/document/6998838) 
17. **Using Recurrent Neural  Networks for Slot Filling in Spoken Language Understanding** (ATIS) `IEEE/ACM Transactions on Audio, Speech, and Language  Processing 2015` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6998838) 
18. **Recurrent  Neural Network Structured Output Prediction for Spoken Language Understanding** (ATIS) `- 2015` [[pdf]](http://speech.sv.cmu.edu/publications/liu-nipsslu-2015.pdf) 
19. **Spoken Language Understanding  Using Long Short-Term Memory Neural Networks** (ATIS) `IEEE 2014` [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) 
20. **Recurrent conditional random  field for language understanding** (ATIS) `IEEE 2014` [[pdf]](https://ieeexplore.ieee.org/document/6854368) 
21. **Recurrent Neural Networks for  Language Understanding** (ATIS) `INTERSPEECH 2013` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kaisheny-338_file_paper.pdf) 
22. **Investigation of  Recurrent-Neural-Network Architectures and Learning Methods for Spoken  Language Understanding** (ATIS) `ISCA 2013` [[pdf]](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_3771.pdf) 
23. **Large-scale personal assistant  technology deployment: the siri experience** (-) `INTERSPEECH 2013` [[pdf]](https://isca-speech.org/archive/archive_papers/interspeech_2013/i13_2029.pdf) 

### Single Intent Detection

1. **Zero-shot User Intent  Detection via Capsule Neural Networks** (SNIPS/CVA) `EMNLP 2018` [[pdf]](https://arxiv.org/pdf/1809.00385.pdf) 
2. **Intention Detection Based on Siamese Neural Network With Triplet Loss** (SNIPS/ATIS/Facebook multilingual datasets/ Daily Dialogue/MRDA) `IEEE Acess 2020` [[pdf]](https://ieeexplore.ieee.org/document/9082602) 
3. **Multi-Layer Ensembling Techniques for Multilingual Intent Classification** (ATIS) `arXiv 2018` [[pdf]](https://arxiv.org/pdf/1806.07914.pdf) 
4. **Deep Unknown Intent Detection with Margin Loss** (SNIPS/ATIS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1906.00434.pdf) 
5. **Subword Semantic Hashing for Intent Classification on Small Datasets** (The Chatbot Corpus/The AskUbuntu Corpus) `IJCNN 2019` [[pdf]](https://arxiv.org/pdf/1810.07150.pdf) 
6. **Dialogue intent classification with character-CNN-BGRU networks** (the Chinese Wikipedia dataset) `Multimedia Tools and Applications 2018` [[pdf]](https://link.springer.com/article/10.1007/s11042-019-7678-1)  
7. **Joint Learning of Domain Classification and Out-of-Domain Detection with Dynamic Class Weighting for Satisficing False Acceptance Rates** (Alexa) `InterSpeech 2018` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1581.pdf)  
8. **Recurrent neural network and  LSTM models for lexical utterance classification** (ATIS/CB) `INTERSPEECH 2015` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) 
9. **Adversarial Training for Multi-task and Multi-lingual Joint Modeling of Utterance Intent Classification** (collected by the author) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/D18-1064.pdf) 
10. **Exploiting Shared Information for Multi-Intent Natural Language Sentence Classification** (collected by the author) `ISCA 2013` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/08/double_intent.pdf)  

### Joint Model

#### Implicit joint modeling

1.  **Leveraging Non-Conversational  Tasks for Low Resource Slot Filling: Does it help?** (ATIS/MIT Restaurant, and Movie/OntoNotes 5.0/OPUS   News Commentary) `SIGDIAL 2019` [[pdf]](https://www.aclweb.org/anthology/W19-5911.pdf) 
2.  **Simple, Fast, Accurate Intent Classification and Slot Labeling for Goal-Oriented Dialogue Systems** (ATIS/SNIPS) `SIGDIAL 2019` [[pdf]](https://www.aclweb.org/anthology/W19-5906.pdf)
3.  **Multi-task learning for Joint  Language Understanding and Dialogue State Tracking** (M2M/DSTC2) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5045.pdf) 
4.  **A Joint Model of Intent  Determination and Slot Filling for Spoken Language Understanding** (ATIS/CQUD) `IJCAI 2016` [[pdf]](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) 
5.  **Joint Online Spoken Language  Understanding and Language Modeling with Recurrent Neural Networks** (ATIS) `SIGDIAL 2016` [[pdf]](https://www.aclweb.org/anthology/W16-3603.pdf) [[code]](https://github.com/HadoopIt/joint-slu-lm)
6.  **Multi-Domain Joint Semantic  Frame Parsing using Bi-directional RNN-LSTM** (ATIS) `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/d644/ae996755c803e067899bdd5ea52498d7091d.pdf) 
7.  **Attention-Based Recurrent  Neural Network Models for Joint Intent Detection and Slot Filling** (ATIS) `INTERSPEECH 2016` [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) 
8.  **Multi-domain joint semantic  frame parsing using bi-directional RNN-LSTM** (ATIS) `INTERSPEECH 2016` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/IS16_MultiJoint.pdf) 
9.  **JOINT SEMANTIC UTTERANCE  CLASSIFICATION AND SLOT FILLING WITH RECURSIVE NEURAL NETWORKS** (ATIS/Stanford Dialogue Dataset,Microsoft Cortana  conversational understanding task(-)) `IEEE SLT 2014` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7078634) 
10.  **CONVOLUTIONAL NEURAL NETWORK  BASED TRIANGULAR CRF FOR JOINT INTENT DETECTION AND SLOT FILLING** (ATIS) `IEEE Workshop on Automatic Speech Recognition and  Understanding 2013` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6707709) 

#### Explicit joint modeling

1.	**A Result based Portable Framework for Spoken Language Understanding**(KVRET) `ICME 2021` [[pdf]](https://arxiv.org/pdf/2103.06010.pdf) 
2.  **A Co-Interactive Transformer for Joint Slot Filling and Intent Detection**(ATIS/SNIPS) `ICASSP 2021` [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) [[code]](https://github.com/kangbrilliant/DCA-Net)
3.  **SlotRefine: A Fast Non-Autoregressive Model for Joint Intent Detection and Slot Filling** (ATIS/SNIPS) `EMNLP 2020` [[pdf]](https://www.aclweb.org/anthology/2020.emnlp-main.152.pdf) [[code]](https://github.com/moore3930/SlotRefine)
4.  **Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding**(ATIS/SNIPS) `AAAI 2020` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6499/6355) 
5.  **Joint Slot Filling and Intent  Detection via Capsule Neural Networks** (ATIS/SNIPS) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) [[code]](https://github.com/czhang99/Capsule-NLU) 
6.  **A Stack-Propagation Framework  with Token-Level Intent Detection for Spoken Language Understanding** (ATIS/SNIPS) `EMNLP 2019` [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) [[code]](https://github.com/LeePleased/StackPropagation-SLU) 
7.  **A Joint Learning Framework  With BERT for Spoken Language Understanding** (ATIS/SNIPS/Facebook's Multilingual dataset) `IEEE 2019` [[pdf]](https://ieeexplore.ieee.org/document/8907842) 
8.  **BERT for Joint Intent  Classification and Slot Filling** (ATIS/SNIPS/Stanford Dialogue Dataset) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) [[code]](https://github.com/monologg/JointBERT) 
9.  **A Novel Bi-directional  Interrelated Model for Joint Intent Detection and Slot Filling** (ATIS/Stanford Dialogue Dataset,SNIPS) `ACL 2019` [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) [[code]](https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU) 
10.  **Joint Multiple Intent  Detection and Slot Labeling for Goal-Oriented Dialog** (ATIS/Stanford Dialogue Dataset/SNIPS) `NAACL 2019` [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf) 
11.  **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 
12.  **A Bi-model based RNN Semantic  Frame Parsing Model for Intent Detection and Slot Filling** (ATIS) `NAACL 2018` [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) 
13.  **Slot-Gated Modeling for Joint  Slot Filling and Intent Prediction** (ATIS/Stanford Dialogue Dataset,SNIPS) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) [[code]](https://github.com/MiuLab/SlotGated-SLU) 
14.  **A Self-Attentive Model with  Gate Mechanism for Spoken Language Understanding** (ATIS) `EMNLP 2018` [[pdf]](https://www.aclweb.org/anthology/D18-1417.pdf) 

### Contextual SLU

2. **Knowing Where to Leverage: Context-Aware Graph Convolutional Network with An Adaptive Fusion Layer for Contextual Spoken Language Understanding** (Simulated Dialogues dataset) `IEEE 2021` [[pdf]](https://ieeexplore.ieee.org/document/9330801) 
2. **Dynamically Context-sensitive Time-decay Attention for Dialogue Modeling** (DSTC4) `IEEE 2019` [[pdf]](https://arxiv.org/pdf/1809.01557.pdf) 
3. **Multi-turn Intent Determination for Goal-oriented Dialogue systems** (Frames/Key-Value Retrieval) `IJCNN 2019` [[pdf]](https://ieeexplore.ieee.org/document/8852246) 
4. **Transfer Learning for Context-Aware Spoken Language Understanding** (single-turn: ATIS/SNIPS multi-turn: Simulated Dialogues dataset) `IEEE 2019` [[pdf]](https://ieeexplore.ieee.org/document/9003902) 
5. **How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues** (DSTC4) `NAACL 2018` [[pdf]](https://www.aclweb.org/anthology/N18-1194.pdf) [[code]](https://github.com/MiuLab/Time-Decay-SLU) 
6. **An Efficient Approach to Encoding Context for Spoken Language Understanding** (Simulated Dialogues dataset) `InterSpeech 2018` [[pdf]](https://arxiv.org/pdf/1807.00267.pdf) 
7. **Speaker-sensitive dual memory networks for multi-turn slot tagging** (Microsoft Cortana) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1711.10705.pdf) 
8. **Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning** (DSTC4) `IJCNLP 2017` [[pdf]](https://www.aclweb.org/anthology/I17-2028.pdf) [[code]](https://github.com/MiuLab/Spk-Dialogue) 
9. **Sequential dialogue context modeling for spoken language understanding** (collected by the author) `SIGDIAL 2017` [[pdf]](https://arxiv.org/pdf/1705.03455.pdf) 
10. **End-to-end joint learning of natural language understanding and dialogue manager** (DSTC4) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1612.00913.pdf) [[code]](https://github.com/XuesongYang/end2end_dialog.git) 
11. **Dynamic time-aware attention to speaker roles and contexts for spoken language understanding** (DSTC4) `IEEE 2017` [[pdf]](https://arxiv.org/pdf/1710.00165.pdf) [[code]](https://github.com/MiuLab/Time-SLU) 
12. **An Intelligent Assistant for High-Level Task Understanding** (collected by the author) `IUI 2016` [[pdf]](http://www.cs.cmu.edu/~mings/papers/IUI16IntelligentAssistant.pdf) 
13. **End-to-End Memory Networks with Knowledge Carryover for Multi-Turn Spoken Language Understanding** (Collected from Microsoft Cortana) `INTEERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/df07/45ce821007cb3122f00509cc18f2885fa8bd.pdf) 
14. **Leveraging behavioral patterns of mobile applications for personalized spoken language understanding** (collected by the author) `ICMI 2015` [[pdf]](https://www.csie.ntu.edu.tw/~yvchen/doc/ICMI15_MultiModel.pdf) 
15. **Contextual spoken language understanding using recurrent neural networks** (single-turn: ATIS multi-turn: Microsoft Cortana) ` 2015` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/0005271.pdf) 
16. **Contextual domain classification in spoken language understanding systems using recurrent neural network** (collected by the author) `IEEE 2014` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/05/rnn_dom.pdf) 
17. **Easy contextual intent prediction and slot detection** (collected by the author) `IEEE 2013` [[pdf]](https://ieeexplore.ieee.org/document/6639291) 

### Multi-intent SLU

1. **AGIF: An Adaptive Graph-Interactive Framework for Joint Multiple Intent Detection and Slot Filling** (MixATIS/MixSNIPS) `EMNLP 2020` [[pdf]](https://www.aclweb.org/anthology/2020.findings-emnlp.163.pdf) [[code]](https://github.com/LooperXX/AGIF)
2. **Joint Multiple Intent Detection and Slot Labeling for Goal-Oriented Dialog** (ATIS/SNIPS/internal dataset) `NACCL 2019` [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf)
3. **Two-stage multi-intent detection for spoken language understanding** (Korean-language corpus for the TV guide domain colleted by author) `Multimed Tools Appl 2017` [[pdf]](https://link.springer.com/article/10.1007/s11042-016-3724-4)
4. **Exploiting Shared Information for Multi-intent Natural Language Sentence Classification** (inhouse corpus from Microsoft) `Interspeech 2013` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/08/double_intent.pdf)

### Chinese SLU

1. **Injecting Word Information with Multi-Level Word Adapter for Chinese Spoken Language Understanding** (CAIS/ECDT-NLU) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.03903.pdf) [[code]](https://github.com/AaronTengDeChuan/MLWA-Chinese-SLU) 
2. **CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding** (ATIS/SNIPS/CAIS) `EMNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf) [[code]](https://github.com/Adaxry/CM-Net) 

### Cross-domain SLU

1. **Coach: A Coarse-to-Fine  Approach for Cross-domain Slot Filling** (SNIPS) `ACL 2020` [[pdf]](https://arxiv.org/pdf/2004.11727.pdf) [[code]](https://github.com/zliucr/coach)
2. **Towards  Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue  Dataset** (SGD) `AAAI 2020` [[pdf]](https://arxiv.org/pdf/1909.05855.pdf) 
3. **Unsupervised Transfer Learning  for Spoken Language Understanding in Intelligent Agents** (ATIS/SINPS) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) 
4. **Zero-Shot Adaptive Transfer  for Conversational Language Understanding** (collected by author) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1808.10059.pdf) 
5. **Robust Zero-Shot Cross-Domain  Slot Filling with Example Values** (SNIPS/XSchema) `ACL 2019` [[pdf]](https://arxiv.org/pdf/1906.06870.pdf) 
6. **Concept Transfer Learning for  Adaptive Language Understanding** (ATIS/DSTC2&3) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5047.pdf) 
7. **Fast and Scalable Expansion of  Natural Language Understanding Functionality for Intelligent Agents** (generated by the author) `NAACL 2018` [[pdf]](https://arxiv.org/pdf/1805.01542.pdf) 
8. **Bag of Experts Architectures  for Model Reuse in Conversational Language Understanding** (generated by the author) `NAACL-HLT 2018` [[pdf]](https://www.aclweb.org/anthology/N18-3019.pdf) 
9. **Domain Attention with an  Ensemble of Experts** (corpus 7 Microsoft Cortana domains) `ACL 2017` [[pdf]](https://www.aclweb.org/anthology/P17-1060.pdf) 
10. **Towards Zero-Shot Frame  Semantic Parsing for Domain Scaling** `INTERSPEECH 2017` (collected by the author) [[pdf]](https://arxiv.org/pdf/1707.02363.pdf) 
11. **Zero-Shot Learning across  Heterogeneous Overlapping Domains** `INTERSPEECH 2017` (inhouse data from Amazon) [[pdf]](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0516.PDFF) 
12. **Domainless Adaptation by  Constrained Decoding on a Schema Lattice** (Cortana) `COLING 2016` [[pdf]](https://www.aclweb.org/anthology/C16-1193.pdf) 
13. **Domain Adaptation of Recurrent  Neural Networks for Natural Language Understanding** (United Airlines/Airbnb/Grey-hound bus service/OpenTable (Data  obtained from App)) `INTERSPEECH 2016` [[pdf]](https://arxiv.org/pdf/1604.00117.pdf) 
14. **Natural Language Model  Re-usability for Scaling to Different Domains** (ATIS/MultiATIS) `EMNLP 2016` [[pdf]](https://www.aclweb.org/anthology/D16-1222.pdf) 
15. **Frustratingly Easy Neural  Domain Adaptation** (Cortana) `COLING 2016` [[pdf]](https://www.aclweb.org/anthology/C16-1038.pdf) 
16. **Multi-Domain Joint Semantic  Frame Parsing using Bi-directional RNN-LSTM** (ATIS) `INTERSPEECH 2016` [[pdf]](https://pdfs.semanticscholar.org/d644/ae996755c803e067899bdd5ea52498d7091d.pdf) 
17. **A Model of Zero-Shot Learning  of Spoken Language Understanding** (generated by the author) `EMNLP 2015` [[pdf]](https://www.aclweb.org/anthology/D15-1027.pdf) 
18. **Online adaptative zero-shot learning spoken language understanding using word-embedding** (DSTC2) `IEEE 2015` [[pdf]](https://ieeexplore.ieee.org/document/7178987) 
19. **Multi-Task Learning for Spoken  Language Understanding with Shared Slots** (collected by the author) `INTERSPEECH 2011` [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/08/Xiao-IS11.pdf) 

### Cross-lingual SLU

1. **CoSDA-ML: Multi-Lingual  Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP** (SC2/4/MLDoc/Multi WOZ/Facebook Multilingual SLU Dataset) `IJCAI 2020` [[pdf]](https://arxiv.org/pdf/2006.06402.pdf) [[code]](https://github.com/kodenii/CoSDA-ML) 
2. **Cross-lingual Spoken Language  Understanding with Regularized Representation Alignment** (Multilingual spoken language understanding (SLU) dataset) `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.14510.pdf) [[code]](https://github.com/zliucr/crosslingual-slu.) 
3. **End-to-End Slot Alignment and  Recognition for Cross-Lingual NLU** (ATIS/MultiATIS) `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2004.14353.pdf) 
4. **Multi-Level Cross-Lingual  Transfer Learning With Language Shared and Specific Knowledge for Spoken  Language Understanding** (Facebook Multilingual SLU Dataset) `IEEE Access 2020` [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8990095) 
5. **Attention-Informed  Mixed-Language Training for Zero-shot Cross-lingual Task-oriented Dialogue  Systems** (Facebook Multilingual SLU Dataset/(DST)MultiWOZ) `AAAI 2020` [[pdf]](https://arxiv.org/pdf/1911.09273.pdf) [[code]](https://github.com/zliucr/mixedlanguage-training) 
6. **MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark** (MTOP /Multilingual ATIS) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2008.09335.pdf) [[code]]() 
7. **Cross-lingual Transfer  Learning with Data Selection for Large-Scale Spoken Language Understanding** (ATIS) `EMNLP-IJCNLP 2019` [[pdf]](https://www.aclweb.org/anthology/D19-1153.pdf) 
8. **Zero-shot Cross-lingual  Dialogue Systems with Transferable Latent Variables** (Facebook Multilingual SLU Dataset) `EMNLP-IJCNLP 2019` [[pdf]](https://arxiv.org/pdf/1911.04081.pdf) 
9. **Cross-Lingual Transfer  Learning for Multilingual Task Oriented Dialog** (Facebook Multilingual SLU Dataset) `NAACL 2019` [[pdf]](https://arxiv.org/pdf/1810.13327.pdf) 
10. **Almawave-SLU: A new dataset  for SLU in Italian** (Valentina.Bellomaria@almawave.it) `CEUR Workshop 2019` [[pdf]](https://arxiv.org/pdf/1907.07526.pdf) 
11. **Multi-lingual Intent Detection  and Slot Filling in a Joint BERT-based Model** (ATIS/SNIPS) `arXiv 2019` [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) 
12. **(Almost) Zero-Shot  Cross-Lingual Spoken Language Understanding** (ATIS manually translated into Hindi and Turkish) `IEEE/ICASSP 2018` [[pdf]](http://shyamupa.com/papers/UFTHH18.pdf) 
14. **Neural Architectures for  Multilingual Semantic Parsing** (GEO/ATIS) `ACL 2017` [[pdf]](https://www.aclweb.org/anthology/P17-2007.pdf) [[code]](http://statnlp.org/research/sp/) 
15. **Multi-style adaptive training  for robust cross-lingual spoken language understanding** (English-Chinese ATIS) `IEEE 2013` [[pdf]](https://ieeexplore.ieee.org/abstract/document/6639292) 
16. **ASGARD: A PORTABLE  ARCHITECTURE FOR MULTILINGUAL DIALOGUE SYSTEMS** (collected from crowd-sourcing platform) `ICASSP 2013` [[pdf]](https://groups.csail.mit.edu/sls/publications/2013/Liu_ICASSP-2013.pdf) 
17. **Combining multiple translation  systems for Spoken Language Understanding portability** (MEDIA) `IEEE 2012` [[pdf]](https://ieeexplore.ieee.org/document/6424221) 

### Low-resource SLU

#### Few-shot SLU

1. **Few-shot Learning for Multi-label Intent Detection** (TourSG/StandfordLU) `AAAI 2021` [[pdf]](https://arxiv.org/abs/2010.05256) [[code]](https://github.com/AtmaHou/FewShotMultiLabel) 
2. **Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network** (SNIPS and further construct) `ACL 2020` [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.128.pdf) [[code]](https://github.com/AtmaHou/FewShotTagging)
3. **Data Augmentation for Spoken  Language Understanding via Pretrained Models** (ATIS/SNIPS) `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2004.13952.pdf) 
4. **Data augmentation by data  noising for open vocabulary slots in spoken language understanding** (ATIS/Snips/MIT-Restaurant.) `NAACL-HLT 2019` [[pdf]](https://www.aclweb.org/anthology/N19-3014.pdf) 
5. **Data Augmentation for Spoken  Language Understanding via Joint Variational Generation** (ATIS) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) 
6. **Marrying Up Regular  Expressions with Neural Networks: A Case Study for Spoken Language  Understanding** (ATIS) `ACL 2018` [[pdf]](https://www.aclweb.org/anthology/P18-1194.pdf) 
7. **Concept Transfer Learning for  Adaptive Language Understanding** (ATIS/DSTC2&3) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5047.pdf) 

#### Zero-shot SLU
1. **Coach: A Coarse-to-Fine  Approach for Cross-domain Slot Filling** (SNIPS) `ACL 2020` [[pdf]](https://arxiv.org/pdf/2004.11727.pdf) [[code]](https://github.com/zliucr/coach)
2. **Zero-Shot Adaptive Transfer  for Conversational Language Understanding** (collected by the author) `AAAI 2019` [[pdf]](https://arxiv.org/pdf/1808.10059.pdf) 
3. **Toward zero-shot Entity  Recognition in Task-oriented Conversational Agents** (Entity gazetteers/Synthetic Gazetteers/Synthetic Utterances) `SIGDIAL 2018` [[pdf]](https://www.aclweb.org/anthology/W18-5036.pdf) 
4. **Zero-shot User Intent  Detection via Capsule Neural Networks** (SNIPS/CVA) `EMNLP 2018` [[pdf]](https://arxiv.org/pdf/1809.00385.pdf) 
5. **Towards Zero-Shot Frame  Semantic Parsing for Domain Scaling** `INTERSPEECH 2017` [[pdf]](https://arxiv.org/pdf/1707.02363.pdf) 
6. **Zero-Shot Learning across  Heterogeneous Overlapping Domains** `INTERSPEECH 2017` [[pdf]](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0516.PDFF) 
7. **A Model of Zero-Shot Learning  of Spoken Language Understanding** (generated by the author) `EMNLP 2015` [[pdf]](https://www.aclweb.org/anthology/D15-1027.pdf) 
8. **Zero-shot semantic parser for  spoken language understanding** (DSTC2&3) `INTERSPEECH 2015` [[pdf]](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1403.pdf) 

#### Unsupervised SLU

1. **Deep Open Intent Classification with Adaptive Decision Boundary** (Banking-77 / CLINC150) `AAAI 2021`  [[pdf]](https://arxiv.org/pdf/2012.10209.pdf) [[code]](https://github.com/thuiar/Adaptive-Decision-Boundary)
2. **Discovering New Intents with Deep Aligned Clustering** (Banking-77 / CLINC150) `AAAI 2021`  [[pdf]](https://arxiv.org/pdf/2012.08987.pdf) [[code]](https://github.com/thuiar/DeepAligned-Clustering)
3. **Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement** (SNIPS) `AAAI 2020`  [[pdf]](https://arxiv.org/pdf/1911.08891.pdf) [[code]](https://github.com/thuiar/CDAC-plus)
4. **Dialogue State Induction Using Neural Latent Variable Models** (MultiWOZ 2.1/SGD) `IJCAI 2020`  [[pdf]](https://www.ijcai.org/proceedings/2020/0532.pdf)

## LeaderBoard
### ATIS

#### Non-pretrained model

<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th> Model</th>
    <th>Intent Acc</th>
    <th>Slot F1</th>
    <th>Paper / Source</th>
    <th>Code link</th>
    <th>Conference</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> Co-Interactive(Qin et al., 2021)                         </td></code>
		<td> 97.7       </td>
		<td> 95.9    </td>
		<td> A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) </td>
		<td> https://github.com/kangbrilliant/DCA-Net      </td>
		<td> ICASSP  </td>
		<td></td>
</tr>
<tr>
	<td><code> Graph LSTM(Zhang et al., 2021)                         </td></code>
		<td> 97.20      </td>
		<td> 95.91    </td>
		<td> Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding  [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6499/6355) </td>
		<td> -      </td>
		<td> AAAI  </td>
		<td></td>
</tr>
<tr>
	<td><code> Stack  Propagation(Qin et al., 2019)                         </td></code>
		<td> 96.9       </td>
		<td> 95.9    </td>
		<td> A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding  [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) </td>
		<td> https://github.com/LeePleased/StackPropagation-SLU      </td>
		<td> EMNLP  </td>
		<td></td>
</tr>
<tr>
	<td><code> SF-ID+CRF(SF first)(E et al., 2019)         </td></code>
		<td> 97.76      </td>
		<td> 95.75   </td>
		<td> A Novel   Bi-directional Interrelated Model for Joint Intent Detection and Slot   Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) </td>
		<td>                                                       </td>
		<td> ACL        </td>
		<td></td>
</tr>
<tr>
	<td><code> SF-ID+CRF(ID first)(E et al., 2019)         </td></code>
		<td> 97.09      </td>
		<td> 95.8    </td>
		<td> A Novel   Bi-directional Interrelated Model for Joint Intent Detection and Slot   Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) </td>
		<td> https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU </td>
		<td> ACL        </td>
		<td></td>
</tr>
<tr>
	<td><code> Capsule-NLU(Zhang  et al. 2019)                              </td></code>
		<td> 95         </td>
		<td> 95.2    </td>
		<td> Joint Slot   Filling and Intent Detection via Capsule Neural Networks [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) </td>
		<td> https://github.com/czhang99/Capsule-NLU                 </td>
		<td> ACL                                         </td>
		<td></td>
</tr>
<tr>
	<td><code> Utterance  Generation With Variational Auto-Encoder(Guo et al., 2019) </td></code>
		<td> -          </td>
		<td> 95.04   </td>
		<td> Utterance  Generation With Variational Auto-Encoder for Slot Filling in Spoken Language  Understanding [[pdf]](https://ieeexplore.ieee.org/document/8625384) </td>
		<td> -                                                       </td>
		<td> IEEE Signal Processing Letters              </td>
		<td></td>
</tr>
<tr>
	<td><code> JULVA(full)(Yoo  et al., 2019)                               </td></code>
		<td> 97.24      </td>
		<td> 95.51   </td>
		<td> Data Augmentation   for Spoken Language Understanding via Joint Variational Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) </td>
		<td> -                                                       </td>
		<td> AAAI                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> CM-Net(Liu  et al., 2019)                               </td></code>
		<td> 99.1      </td>
		<td> 96.20   </td>
		<td> CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding[[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf)</td>
		<td> https://github.com/Adaxry/CM-Net    </td>
		<td> EMNLP                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> Data  noising method(Kim et al., 2019)                       </td></code>
		<td> 98.43      </td>
		<td> 96.20    </td>
		<td> Data  augmentation by data noising for open vocabulary slots in spoken language  understanding [[pdf]](https://www.aclweb.org/anthology/N19-3014.pdf) </td>
		<td> -                                                       </td>
		<td> NAACL-HLT                                   </td>
		<td></td>
</tr>
<tr>
	<td><code> ACD(Zhu  et al., 2018)                                       </td></code>
		<td> -          </td>
		<td> 96.08   </td>
		<td> Concept   Transfer Learning for Adaptive Language Understanding [[pdf]](https://www.aclweb.org/anthology/W18-5047.pdf) </td>
		<td> -                                                       </td>
		<td> SIGDIAL                                     </td>
		<td></td>
</tr>
<tr>
	<td><code> A Self-Attentive Model with Gate Mechanism(Li et al., 2018)  </td></code>
		<td> 98.77      </td>
		<td> 96.52   </td>
		<td> A   Self-Attentive Model with Gate Mechanism for Spoken Language   Understanding [[pdf]](https://www.aclweb.org/anthology/D18-1417.pdf) </td>
		<td> -                                                       </td>
		<td> EMNLP                                       </td>
		<td></td>
</tr>
<tr>
	<td><code> Slot-Gated(Goo  et al., 2018)                                </td></code>
		<td> 94.1       </td>
		<td> 95.2    </td>
		<td> Slot-Gated   Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) </td>
		<td> https://github.com/MiuLab/SlotGated-SLU                 </td>
		<td> NAACL                                       </td>
		<td></td>
</tr>
<tr>
	<td><code> DRL based Augmented Tagging System(Wang et al., 2018)        </td></code>
		<td> -          </td>
		<td> 97.86   </td>
		<td> A  New Concept of Deep Reinforcement Learning based Augmented General Sequence  Tagging System [[pdf]](https://www.aclweb.org/anthology/C18-1143.pdf) </td>
		<td> -                                                       </td>
		<td> COLING      </td>
		<td></td>
</tr>
<tr>
	<td><code> Bi-model(Wang  et al., 2018)                                 </td></code>
		<td> 98.76      </td>
		<td> 96.65   </td>
		<td> A Bi-model based   RNN Semantic Frame Parsing Model for Intent Detection and Slot Filling [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) </td>
		<td> -                                                       </td>
		<td> NAACL                                       </td>
		<td></td>
</tr>
<tr>
	<td><code> Bi-model+decoder(Wang  et al., 2018)        </td></code>
		<td> 98.99      </td>
		<td> 96.89   </td>
		<td> A Bi-model based   RNN Semantic Frame Parsing Model for Intent Detection and Slot Filling [[pdf]](https://arxiv.org/pdf/1812.10235.pdf) </td>
		<td> -                                                     </td>
		<td> NAACL      </td>
		<td></td>
</tr>
<tr>
	<td><code> Seq2Seq DA for LU(Hou et al., 2018)                          </td></code>
		<td> -          </td>
		<td> 94.82   </td>
		<td> Sequence-to-Sequence  Data Augmentation for Dialogue Language Understanding [[pdf]](https://arxiv.org/pdf/1807.01554.pdf) </td>
		<td> https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU </td>
		<td> COLING                                      </td>
		<td></td>
</tr>
<tr>
	<td><code> BLSTM-LSTM(Zhu  et al., 2017)                                </td></code>
		<td> -          </td>
		<td> 95.79   </td>
		<td> ENCODER-DECODER  WITH FOCUS-MECHANISM FOR SEQUENCE LABELLING BASED SPOKEN LANGUAGE  UNDERSTANDING  [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7953243) </td>
		<td> -                                                       </td>
		<td> ICASSP                                      </td>
		<td></td>
</tr>
<tr>
	<td><code> neural  sequence chunking model(Zhai et al., 2017)           </td></code>
		<td> -          </td>
		<td> 95.86   </td>
		<td> Neural  Models for Sequence Chunking [[pdf]](https://arxiv.org/pdf/1701.04027.pdf) </td>
		<td> -                                                       </td>
		<td> AAAI                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  Model of ID and SF(Zhang et al., 2016)                </td></code>
		<td> 98.32      </td>
		<td> 96.89   </td>
		<td> A   Joint Model of Intent Determination and Slot Filling for Spoken Language   Understanding [[pdf]](https://www.ijcai.org/Proceedings/16/Papers/425.pdf) </td>
		<td> -                                                       </td>
		<td> IJCAI                                       </td>
		<td></td>
</tr>
<tr>
	<td><code> Attention Encoder-Decoder NN (with aligned inputs)           </td></code>
		<td> 98.43      </td>
		<td> 95.87   </td>
		<td> Attention-Based   Recurrent Neural Network Models for Joint Intent Detectionand Slot   Filling      [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> Attention  BiRNN(Liu et al., 2016)                           </td></code>
		<td> 98.21      </td>
		<td> 95.98   </td>
		<td> Attention-Based   Recurrent Neural Network Models for Joint Intent Detectionand Slot   Filling      [[pdf]](https://arxiv.org/pdf/1609.01454.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  SLU-LM model(Liu ei al., 2016)                        </td></code>
		<td> 98.43      </td>
		<td> 94.64   </td>
		<td> Joint Online   Spoken Language Understanding and Language Modeling with Recurrent Neural   Networks [[pdf]](https://arxiv.org/pdf/1609.01462.pdf) </td>
		<td> http://speech.sv.cmu.edu/software.html                  </td>
		<td> SIGDIAL                                     </td>
		<td></td>
</tr>
<tr>
	<td><code> RNN-LSTM(Hakkani-Tur  et al., 2016)                          </td></code>
		<td> 94.3       </td>
		<td> 92.6    </td>
		<td> Multi-Domain Joint Semantic Frame Parsing using   Bi-directional RNN-LSTM [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/IS16_MultiJoint.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> R-biRNN(Vu  et al., 2016)                                    </td></code>
		<td> -          </td>
		<td> 95.47   </td>
		<td> Bi-directional   recurrent neural network with ranking loss for spoken language   understanding      [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) </td>
		<td> -                                                       </td>
		<td> IEEE                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> Encoder-labeler  LSTM(Kurata et al., 2016)                   </td></code>
		<td> -          </td>
		<td> 95.4    </td>
		<td> Leveraging Sentence-level Information with  Encoder LSTM for Semantic Slot Filling [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) </td>
		<td> -                                                       </td>
		<td> EMNLP                                       </td>
		<td></td>
</tr>
<tr>
	<td><code> Encoder-labeler  Deep LSTM(Kurata et al., 2016)              </td></code>
		<td> -          </td>
		<td> 95.66   </td>
		<td> Leveraging Sentence-level Information with  Encoder LSTM for Semantic Slot Filling [[pdf]](https://www.aclweb.org/anthology/D16-1223.pdf) </td>
		<td>                                                         </td>
		<td> EMNLP                                       </td>
		<td></td>
</tr>
<tr>
	<td><code> 5xR-biRNN(Vu  et al., 2016)                 </td></code>
		<td> -          </td>
		<td> 95.56   </td>
		<td> Bi-directional  recurrent neural network with ranking loss for spoken language  understanding [[pdf]](https://ieeexplore.ieee.org/abstract/document/7472841/) </td>
		<td> -                                                     </td>
		<td> IEEE       </td>
		<td></td>
</tr>
<tr>
	<td><code> Data  Generation for SF(Kurata et al., 2016)                 </td></code>
		<td> -          </td>
		<td> 95.32   </td>
		<td> Labeled  Data Generation with Encoder-decoder LSTM for Semantic Slot Filling [[pdf]](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0727.PDF) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> RNN-EM(Peng  et al., 2015)                                   </td></code>
		<td> -          </td>
		<td> 95.25   </td>
		<td> Recurrent Neural   Networks with External Memory for Language Understanding [[pdf]](https://arxiv.org/pdf/1506.00195.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> RNN  trained with sampled label(Liu et al., 2015)            </td></code>
		<td> -          </td>
		<td> 94.89   </td>
		<td> Recurrent Neural Network Structured Output Prediction for   Spoken Language Understanding      [[pdf]](http://speech.sv.cmu.edu/publications/liu-nipsslu-2015.pdf) </td>
		<td> -                                                       </td>
		<td> -                                           </td>
		<td></td>
</tr>
<tr>
	<td><code> RNN(Ravuri  et al., 2015)                                    </td></code>
		<td> 97.55      </td>
		<td> -       </td>
		<td> Recurrent neural network and LSTM models for  lexical utterance classification [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> LSTM(Ravuri  et al., 2015)                                   </td></code>
		<td> 98.06      </td>
		<td> -       </td>
		<td> Recurrent neural network and LSTM models for  lexical utterance classification [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RNNLM_addressee.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> Hybrid  RNN(Mesnil et al., 2015)                             </td></code>
		<td> -          </td>
		<td> 95.06   </td>
		<td> Using  Recurrent Neural Networks for Slot Filling in Spoken Language  Understanding [[pdf]](https://ieeexplore.ieee.org/document/6998838) </td>
		<td> -                                                       </td>
		<td> IEEE/ACM-TASLP                              </td>
		<td></td>
</tr>
<tr>
	<td><code> RecNN(Guo  et al., 2014)                                     </td></code>
		<td> 95.4       </td>
		<td> 93.22   </td>
		<td> Joint semantic utterance classification and slot filling with   recursive neural networks [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/12/RecNNSLU.pdf) </td>
		<td> -                                                       </td>
		<td> IEEE-SLT                                    </td>
		<td></td>
</tr>
<tr>
	<td><code> LSTM(Yao  et al., 2014)                                      </td></code>
		<td> -          </td>
		<td> 94.85   </td>
		<td> Spoken Language Understading Using Long  Short-Term Memory Neural Networks [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) </td>
		<td> -                                                       </td>
		<td> IEEE                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> Deep  LSTM(Yao et al., 2014)                                 </td></code>
		<td> -          </td>
		<td> 95.08   </td>
		<td> Spoken Language Understading Using Long  Short-Term Memory Neural Networks [[pdf]](https://groups.csail.mit.edu/sls/publications/2014/Zhang_SLT_2014.pdf) </td>
		<td> -                                                       </td>
		<td> IEEE                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> R-CRF(Yao  et al., 2014)                                     </td></code>
		<td> -          </td>
		<td> 96.65   </td>
		<td> Recurrent  conditional random field for language understanding [[pdf]](https://ieeexplore.ieee.org/document/6854368) </td>
		<td> -                                                       </td>
		<td> IEEE                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> RecNN+Viterbi(Guo  et al., 2014)            </td></code>
		<td> 95.4       </td>
		<td> 93.96   </td>
		<td> Joint semantic utterance classification and slot filling with   recursive neural networks [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/12/RecNNSLU.pdf) </td>
		<td> -                                                     </td>
		<td> IEEE-SLT   </td>
		<td></td>
</tr>
<tr>
	<td><code> CNN  CRF(Xu et al., 2013)                                    </td></code>
		<td> 94.09      </td>
		<td> 5.42   </td>
		<td> Convolutional neural network based triangular crf for joint   intent detection and slot filling [[pdf]]((http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.7548&rep=rep1&type=pdf)) </td>
		<td> -                                                       </td>
		<td> IEEE                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> RNN(Yao  et al., 2013)                                       </td></code>
		<td> -          </td>
		<td> 94.11   </td>
		<td> Recurrent  Neural Networks for Language Understanding [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kaisheny-338_file_paper.pdf) </td>
		<td> -                                                       </td>
		<td> InterSpeech                                 </td>
		<td></td>
</tr>
<tr>
	<td><code> Bi-dir.  Jordan-RNN(2013)                                    </td></code>
		<td> -          </td>
		<td> 93.98   </td>
		<td> Investigation  of Recurrent-Neural-Network Architectures and Learning Methods for Spoken  Language Understanding [[pdf]](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_3771.pdf) </td>
		<td> -                                                       </td>
		<td> ISCA                                        </td>
		<td></td>
</tr>
</tbody>
</table>
</div>


#### + Pretrained model
<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th> Model</th>
    <th>Intent Acc</th>
    <th>Slot F1</th>
    <th>Paper / Source</th>
    <th>Code link</th>
    <th>Conference</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> Co-Interactive(Qin et al., 2021)                         </td></code>
		<td> 98.0       </td>
		<td> 96.1    </td>
		<td> A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) </td>
		<td> https://github.com/kangbrilliant/DCA-Net      </td>
		<td> ICASSP  </td>
		<td></td>
</tr>
<tr>
	<td><code> Stack  Propagation+BERT(Qin et al., 2019)   </td></code>
		<td> 97.5       </td>
		<td> 96.1    </td>
		<td> A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) </td>
		<td> https://github.com/LeePleased/StackPropagation-SLU    </td>
		<td> EMNLP      </td>
		<td></td>
</tr>
<tr>
	<td><code> Bert-Joint(Castellucci  et al., 2019)       </td></code>
		<td> 97.8       </td>
		<td> 95.7    </td>
		<td> Multi-lingual  Intent Detection and Slot Filling in a Joint BERT-based Model [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) </td>
		<td> -                                                     </td>
		<td> arXiv      </td>
		<td></td>
</tr>
<tr>
	<td><code> BERT-SLU(Zhang  et al., 2019)               </td></code>
		<td> 99.76      </td>
		<td> 98.75   </td>
		<td> A Joint   Learning Framework With BERT for Spoken Language Understanding [[pdf]](https://ieeexplore.ieee.org/document/8907842) </td>
		<td> -                                                     </td>
		<td> IEEE       </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  BERT(Chen et al., 2019)              </td></code>
		<td> 97.5       </td>
		<td> 96.1    </td>
		<td> BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) </td>
		<td> https://github.com/monologg/JointBERT                 </td>
		<td> arXiv      </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  BERT+CRF(Chen et al., 2019)          </td></code>
		<td> 97.9       </td>
		<td> 96      </td>
		<td> BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) </td>
		<td> https://github.com/monologg/JointBERT                 </td>
		<td> arXiv      </td>
		<td></td>
</tr>
<tr>
	<td><code> ELMo-Light  (ELMoL) (Siddhant et al., 2019) </td></code>
		<td> 97.3       </td>
		<td> 95.42   </td>
		<td> Unsupervised   Transfer Learning for Spoken Language Understanding in Intelligent Agents [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) </td>
		<td> -                                                     </td>
		<td> AAAI       </td>
		<td></td>
</tr>
</tbody >
</table>
</div>


### SNIPS

#### Non-pretrained model
<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th> Model</th>
    <th>Intent Acc</th>
    <th>Slot F1</th>
    <th>Paper / Source</th>
    <th>Code link</th>
    <th>Conference</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> Co-Interactive(Qin et al., 2021)                         </td></code>
		<td> 98.8       </td>
		<td> 95.9    </td>
		<td> A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) </td>
		<td> https://github.com/kangbrilliant/DCA-Net      </td>
		<td> ICASSP  </td>
		<td></td>
</tr>
<tr>
	<td><code> Graph LSTM(Zhang et al., 2021)                         </td></code>
		<td> 98.29      </td>
		<td> 95.30    </td>
		<td> Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding  [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6499/6355) </td>
		<td> -      </td>
		<td> AAAI  </td>
		<td></td>
</tr>
<tr>
	<td><code> SF-ID  Network(E et al, 2019)                                </td></code>
		<td> 97.43      </td>
		<td> 91.43   </td>
		<td> A  Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot  Filling [[pdf]](https://www.aclweb.org/anthology/P19-1544.pdf) </td>
		<td> https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU        </td>
		<td> ACL                            </td>
		<td></td>
</tr>
<tr>
	<td><code> CAPSULE-NLU(Zhang  et al, 2019)                              </td></code>
		<td> 97.3       </td>
		<td> 91.8    </td>
		<td> Joint  Slot Filling and Intent Detection via Capsule Neural Networks [[pdf]](https://arxiv.org/pdf/1812.09471.pdf) </td>
		<td> https://github.com/czhang99/Capsule-NLU                      </td>
		<td> ACL                            </td>
		<td></td>
</tr>
<tr>
	<td><code> StackPropagation(Qin  et al, 2019)                           </td></code>
		<td> 98         </td>
		<td> 94.2    </td>
		<td> A  Stack-Propagation Framework with Token-Level Intent Detection for Spoken  Language Understanding     [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) </td>
		<td> [https://github.com/LeePleased/StackPropagation-SLU. ](https://github.com/LeePleased/StackPropagation-SLU.) </td>
		<td> EMNLP                          </td>
		<td></td>
</tr>
<tr>
	<td><code> CM-Net(Liu  et al., 2019)                               </td></code>
		<td> 99.29      </td>
		<td> 97.15   </td>
		<td> CM-Net: A Novel Collaborative Memory Network for Spoken Language Understanding[[pdf]](https://www.aclweb.org/anthology/D19-1097.pdf)</td>
		<td> https://github.com/Adaxry/CM-Net    </td>
		<td> EMNLP                                        </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  Multiple(Gangadharaiah et al, 2019)                   </td></code>
		<td> 97.23      </td>
		<td> 88.03   </td>
		<td> Joint  Multiple Intent Detection and Slot Labeling for Goal-Oriented Dialog [[pdf]](https://www.aclweb.org/anthology/N19-1055.pdf) </td>
		<td> -                                                            </td>
		<td> NAACL                          </td>
		<td></td>
</tr>
<tr>
	<td><code> Utterance  Generation With Variational Auto-Encoder(Guo et al., 2019) </td></code>
		<td> -          </td>
		<td> 93.18   </td>
		<td> Utterance  Generation With Variational Auto-Encoder for Slot Filling in Spoken Language  Understanding        [[pdf]](https://ieeexplore.ieee.org/document/8625384) </td>
		<td> -                                                            </td>
		<td> IEEE Signal Processing Letters </td>
		<td></td>
</tr>
<tr>
	<td><code> Slot  Gated Intent Atten.(Goo et al, 2018)                   </td></code>
		<td> 96.8       </td>
		<td> 88.3    </td>
		<td> Slot-Gated   Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) </td>
		<td> https://github.com/MiuLab/SlotGated-SLU                      </td>
		<td> NAACL                          </td>
		<td></td>
</tr>
<tr>
	<td><code> Slot  Gated Fulled Atten.(Goo et al, 2018)                   </td></code>
		<td> 97         </td>
		<td> 88.8    </td>
		<td> Slot-Gated  Modeling for Joint Slot Filling and Intent Prediction [[pdf]](https://www.aclweb.org/anthology/N18-2118.pdf) </td>
		<td> https://github.com/MiuLab/SlotGated-SLU                      </td>
		<td> NAACL                          </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  Variational Generation + Slot Gated Intent Atten(Yoo et al., 2018) </td></code>
		<td> 96.7       </td>
		<td> 88.3    </td>
		<td> Data  Augmentation for Spoken Language Understanding via Joint Variational  Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) </td>
		<td> -                                                            </td>
		<td> AAAI                           </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  Variational Generation + Slot Gated Full Atten(Yoo et al., 2018) </td></code>
		<td> 97.3       </td>
		<td> 89.3    </td>
		<td> Data Augmentation  for Spoken Language Understanding via Joint Variational Generation [[pdf]](https://arxiv.org/pdf/1809.02305.pdf) </td>
		<td> -                                                            </td>
		<td> AAAI                           </td>
		<td></td>
</tr>
</tbody >
</table>
</div>



#### + Pretrained model
<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th> Model</th>
    <th>Intent Acc</th>
    <th>Slot F1</th>
    <th>Paper / Source</th>
    <th>Code link</th>
    <th>Conference</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> Co-Interactive(Qin et al., 2021)                         </td></code>
		<td> 98.8       </td>
		<td> 97.1    </td>
		<td> A Co-Interactive Transformer for Joint Slot Filling and Intent Detection  [[pdf]](https://arxiv.org/pdf/2010.03880.pdf) </td>
		<td> https://github.com/kangbrilliant/DCA-Net      </td>
		<td> ICASSP  </td>
		<td></td>
</tr>
<tr>
	<td><code> StackPropagation  + Bert(Qin et al, 2019)       </td></code>
		<td> 99         </td>
		<td> 97      </td>
		<td> A   Stack-Propagation Framework with Token-Level Intent Detection for Spoken   Language Understanding [[pdf]](https://arxiv.org/pdf/1909.02188.pdf) </td>
		<td> [https://github.com/LeePleased/StackPropagation-SLU. ](https://github.com/LeePleased/StackPropagation-SLU.) </td>
		<td> EMNLP      </td>
		<td></td>
</tr>
<tr>
	<td><code> Bert-Joint(Castellucci  et al, 2019)            </td></code>
		<td> 99         </td>
		<td> 96.2    </td>
		<td> Multi-lingual  Intent Detection and Slot Filling in a Joint BERT-based Mode [[pdf]](https://arxiv.org/pdf/1907.02884.pdf) </td>
		<td> -                                                            </td>
		<td> arXiv      </td>
		<td></td>
</tr>
<tr>
	<td><code> Bert-SLU(Zhang  et al, 2019)                    </td></code>
		<td> 98.96      </td>
		<td> 98.78   </td>
		<td> A Joint Learning  Framework With BERT for Spoken Language Understanding [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8907842) </td>
		<td> -                                                            </td>
		<td> IEEE       </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  BERT(Chen et al, 2019)                   </td></code>
		<td> 98.6       </td>
		<td> 97      </td>
		<td> BERT for Joint   Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) </td>
		<td> https://github.com/monologg/JointBERT                        </td>
		<td> arXiv      </td>
		<td></td>
</tr>
<tr>
	<td><code> Joint  BERT + CRF(Chen et al, 2019)             </td></code>
		<td> 98.4       </td>
		<td> 96.7    </td>
		<td> BERT  for Joint Intent Classification and Slot Filling [[pdf]](https://arxiv.org/pdf/1902.10909.pdf) </td>
		<td> https://github.com/monologg/JointBERT                        </td>
		<td> arXiv      </td>
		<td></td>
</tr>
<tr>
	<td><code> ELMo-Light(Siddhant  et al, 2019)               </td></code>
		<td> 98.38      </td>
		<td> 93.29   </td>
		<td> Unsupervised   Transfer Learning for Spoken Language Understanding in Intelligent Agents         [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) </td>
		<td> -                                                            </td>
		<td> AAAI       </td>
		<td></td>
</tr>
<tr>
	<td><code> ELMo(Peters  et al, 2018;Siddhant et al, 2019 ) </td></code>
		<td> 99.29      </td>
		<td> 93.9    </td>
		<td> Deep   contextualized word representations      [[pdf]](https://arxiv.org/pdf/1802.05365.pdf)Unsupervised Transfer Learning for Spoken Language Understanding in   Intelligent Agents [[pdf]](https://arxiv.org/pdf/1811.05370.pdf) </td>
		<td> -                                                            </td>
		<td> NAACL/AAAI </td>
		<td></td>
</tr>
</tbody>
</table>
</div>


