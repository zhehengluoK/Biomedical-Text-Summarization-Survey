# A Survey on Biomedical Text Summarisation with Pre-trained Language Model(PLM)s

![](https://img.shields.io/badge/Status-building-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) 


# Resource
This repository contains a list of papers, codes, and datasets in Biomedical Text Summarisation based on PLM. If you found any errors, please don't hesitate to open an issue or pull a request.

<!--
If you find this repository helpful for your work,  please consider citing our survey paper. The Bibtex are listed below:
<pre>

</pre>
-->


## Contributor


Resource Contributed by [Qianqian Xie](), [Zheheng Luo](),  [Benyou Wang](),[Sophia Ananiadou](https://www.research.manchester.ac.uk/portal/sophia.ananiadou.html).

## Introduction

Biomedical text summarization has long been a fundamental task in biomedical natural language processing (BioNLP),
aiming at generating concise summaries that distil key information from one or multiple biomedical documents. In recent years,
pre-trained language models (PLMs) have been the de facto standard of various natural language processing tasks in the general
domain. Most recently, PLMs have been further investigated in the biomedical domain and brought new insights into the biomedical text
summarization task. 

To help researchers quickly grasp the development in this task and inspire further research, we line up available datasets, recent approaches and evaluation methods in this project.

At present, the project has been completely open source, including:

1. **BioTS dataset table:** we listed the datasets in the BioTS field, You can find the category, size, content, and access of them in the table.
2. **PLM Based BioTS Methods:** we classified and arranged papers based on the type of output summary, numbers and type of input documents. the current mainstream frontiers. Each line of the table contains the category, the strategy of applying PLM, the backbone model, the training type, and used datasets.
3. **BioTS Evaluation:** we listed metrics that cover three essential aspects in the evaluation of biomedical text summarization: 1) relevancy 2) fluency 3) factuality.

The organization and our survey and the detailed background of biomedical text summarization are illustrated in the pictures below.


![survey-overview](./pics/OverviewOfBiomedicalTextSummarizationWithPLM.png)


![BTSwPLMs-taxonomy](./pics/TaxonomyOfMethods.png)


## Quick path
- [Survey Paper](#survey-paper)
- [Dataset](#dataset)
- [Methods](#methods)
- [Evaluation](#evaluation)
- [Leader Board](#leader-board)

## Survey Paper
1. **Text summarization in the biomedical domain: A systematic review of recent research** `J. biomedical informatics 2014` [[html]](https://www.sciencedirect.com/science/article/pii/S1532046414001476)
1. **Summarization from medical documents: a survey** `Artif. intelligence medicine 2005` [[html]](https://www.sciencedirect.com/science/article/pii/S0933365704001320)
1. **Automated methods for the summarization of electronic health records** `J Am Med Inform Assoc. 2015` [[html]](https://pubmed.ncbi.nlm.nih.gov/25882031/)
1. **A systematic review of automatic text summarization for biomedical literature and ehrs** `J Am Med Inform Assoc. 2021` [[html]](https://pubmed.ncbi.nlm.nih.gov/34338801/)

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
		<td> <a href="https://github.com/vgupta123/sumpubmed">https://github.com/vgupta123/sumpubmed</a></td>
<tr>
	<td><code>S2ORC                              </td></code>
		<td> Biomedical literature </td>
		<td> 63,709 </td>
		<td> Full contents of articles </td>
		<td> Single </td>
		<td> <a href="https://github.com/jbshp/GenCompareSum">https://github.com/jbshp/GenCompareSum</a></td>
<tr>
	<td><code> CORD-19                               </td></code>
		<td> Biomedical literature </td>
		<td> - (constantly increasing)</td>
		<td> Full contents of articles</td>
		<td> Single </td>
		<td> <a href="https://github.com/allenai/cord19">https://github.com/allenai/cord19</a></td>
<tr>
	<td><code> MIMIC-CXR                              </td></code>
		<td> EHR</td>
		<td> 124577</td>
		<td> Full contents of reports</td>
		<td> Single </td>
		<td> <a href="https://physionet.org/content/mimic-cxr/2.0.0/">https://physionet.org/content/mimic-cxr/2.0.0/</a></td>
<tr>
	<td><code> OpenI                              </td></code>
		<td> EHR</td>
		<td> 3599</td>
		<td> Full contents of reports</td>
		<td> Single </td>
		<td> <a href="https://openi.nlm.nih.gov/faq#collection">https://openi.nlm.nih.gov/faq#collection</a></td>
<tr>
	<td><code> MeQSum                              </td></code>
		<td> meidical question summarization</td>
		<td> 1000</td>
		<td> Full contents of question</td>
		<td> Single </td>
		<td> <a href="https://github.com/abachaa/MeQSum">https://github.com/abachaa/MeQSum/</a></td>
<tr>
	<td><code> CHQ-Summ                               </td></code>
		<td> meidical question summarization</td>
		<td> 1507</td>
		<td> Full contents of question</td>
		<td> Single </td>
		<td> <a href="https://github.com/shwetanlp/Yahoo-CHQ-Summ">https://github.com/shwetanlp/Yahoo-CHQ-Summ</a></td>
<tr>
</tbody >
</table>
</div>


## Methods
<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Paper</th>
    <th>Category</th>
    <th>Strategy</th>
    <th>Model</th>
    <th>Training</th>
    <th>Dataset</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code><a href="https://arxiv.org/pdf/2007.03405.pdf"> ContinualBERT </a></code><a href="https://github.com/jdubpark/continual-bert"> [code] </a></td>
		<td>extractive</td>
		<td> fine-tuning</td>
		<td> BERT</td>
		<td> supervised </td>
		<td> PubMed, CORD-19</td>

<tr>
	<td><code><a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705120302859"> BioBERTSum </a> </td></code>
		<td> extractive </td>
		<td>fine-tuning </td>
		<td> BioBERT</td>
		<td> supervised </td>
		<td> PubMed</td>
<tr>
	<td><code> <a href="https://www.sciencedirect.com/science/article/pii/S0950705122007328"> KeBioSum </a> </code> <a href="https://github.com/xashely/KeBioSum"> [code] </a> </td>
		<td> extractive </td>
		<td> adaption+fine-tuning </td>
		<td> PubMedBERT</td>
		<td> supervised </td>
		<td> PubMed, CORD-19, S2ORC</td>
<tr>
	<td><code>    <a href="	https://arxiv.org/pdf/2104.08942.pdf"> N. Kanwal and G. Rizzo</a></code> <a href="https://github.com/NeelKanwal/BERTOLOGY-Based-Extractive-Summarization-for-Clinical-Notes"> [code]</a></td>
		<td> extractive </td>
		<td> fine-tuning </td>
		<td> BERT</td>
		<td> unsupervised </td>
		<td> MIMIC-III</td>
<tr>
	<td><code>      <a href="https://www.researchgate.net/profile/Milad-Moradi-5/publication/336272974_Deep_contextualized_embeddings_for_quantifying_the_informative_content_in_biomedical_text_summarization/links/5d9c45d3a6fdccfd0e811d95/Deep-contextualized-embeddings-for-quantifying-the-informative-content-in-biomedical-text-summarization.pdf"> M. Moradi et.al   </a></code> <a href="https://github.com/BioTextSumm/BERT-based-Summ"> [code]</a></td>
		<td> extractive </td>
		<td> feature-base </td>
		<td> BERT </td>
		<td> unsupervised </td>
		<td> PubMed</td>
<tr>
	<td><code>      <a href="https://www.sciencedirect.com/science/article/pii/S1532046420300800"> M. Moradi et.al</a> </code> <a href="https://github.com/BioTextSumm/Graph-basedSummarizer"> [code]</a></td>
		<td> extractive </td>
		<td> feature-base</td>
		<td> BioBERT</td>
		<td> unsupervised </td>
		<td> PubMed</td>
<tr>
	<td><code>  <a href="https://aclanthology.org/2022.bionlp-1.22/">   GenCompareSum    </a></code> <a href="https://github.com/jbshp/GenCompareSum"> [code]</a></td>
		<td> extractive</td>
		<td> feature-base</td>
		<td> T5</td>
		<td> unsupervised </td>
		<td> PubMed, CORD-19, S2ORC</td>
<tr>
	<td><code> <a href="https://pubmed.ncbi.nlm.nih.gov/35923376/">   RadBERT    </a></td></code> </td></code>
		<td> extractive</td>
		<td> feature-base</td>
		<td> RadBERT</td>
		<td> unsupervised </td>
		<td> - </td>
<tr>                                                                                                                                                                                          
        <td><code><a href="https://arxiv.org/pdf/2006.01997.pdf">   B Tan et.al  </a></code> <a href="https://github.com/VincentK1991/BERT_summarization_1"> [code]</a></td>                                                                                                                               
                <td>hybrid</td>                                                                                                                                                               
                <td>adaption+fine-tuning</td>                                                                                                                                                 
                <td>BERT,GPT-2</td>                                                                                                                                                           
                <td>supervised</td>                                                                                                                                                           
                <td>CORD-19</td>                                                                                                                                                              
<tr>                                                                                                                                                                                          
        <td><code> <a href="https://arxiv.org/pdf/2005.00163.pdf">S. S. Gharebagh et.al</a></td></code>                                                                                                                                  
                <td>abstractive</td>                                                                                                                                                          
                <td>feature-base</td>                                                                                                                                                         
                <td>BERT</td>                                                                                                                                                                 
                <td>supervised</td>                                                                                                                                                           
                <td>MIMIC-CXR</td>                                                                                                                                                            
<tr>                                                                                                                                                                                          
        <td><code>  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16089/15896">Y. Guo et.al </a> </code> <a href="https://github.com/qiuweipku/Plain_language_summarization "> [code]</a></td>                                                                                                                                                                                                                                                    
                <td>hybrid</td>                                                                                                                                                               
                <td>adaption+fine-tuning</td>                                                                                                                                                 
                <td>BERT, BART</td>                                                                                                                                                           
                <td>supervised</td>                                                                                                                                                           
                <td>CDSR</td>                                                                                                                                                                 
<tr>                                                                                                                                                                                       
        <td><code><a href="https://aclanthology.org/2021.bionlp-1.29/">L. Xu et.al </a></td></code>                                                                                                                                     
                <td>abstractive,question</td>                                                                                                                                                 
                <td>adaption+fine-tuning</td>                                                                                                                                                 
                <td>BART,PEGASUS</td>                                                                                                                                                         
                <td>supervised</td>                                                                                                                                                           
                <td>MIMIC-CXR,OpenI,MeQSum</td>                                                                                                                                               
<tr>                                                                                                                                                                                          
        <td><code> <a href="https://aclanthology.org/2021.bionlp-1.10.pdf">W. Zhu et.al</a>   </td></code>
                <td>abstractive</td>
                <td>fine-tuning</td>
                <td>BART,T5,PEGASUS</td>
                <td>supervised</td>
                <td>MIMIC-CXR,OpenI</td>
<tr>
        <td><code>  <a href="https://aclanthology.org/2021.bionlp-1.32.pdf"> R. Kondadadi et.al </a></td></code>
                <td>abstractive</td>
                <td>fine-tuning</td>
                <td>BART,T5,PEGASUS</td>
                <td>supervised</td>
                <td>MIMIC-CXR,OpenI</td>
<tr>
        <td><code><a href="https://aclanthology.org/2021.bionlp-1.11.pdf">S. Dai et.al</a></td></code>
                <td>abstractive</td>
                <td>adaption+fine-tuning</td>
                <td>PEGASUS</td>
                <td>supervised</td>
                <td>MIMIC-CXR,OpenI</td>
<tr>
        <td><code><a href="https://aclanthology.org/2021.bionlp-1.35.pdf">  D. Mahajan et.al</a></td></code>
                <td>abstractive</td>
                <td>adaption+fine-tuning</td>
                <td><a href="https://aclanthology.org/2020.tacl-1.18/">BioRoBERTa</a></td>
                <td>supervised</td>
                <td>MIMIC-CXR,OpenI</td>
<tr>
        <td><code><a href="https://aclanthology.org/2022.acl-long.320.pdf">H. Jingpeng et.al </a></code> <a href="https://github.com/jinpeng01/AIG_CL"> [code]</a></td>                   
        		   <td>abstractive</td>
                <td>fine-tuning</td>
                <td>BioBERT</td>
                <td>supervised</td>
                <td>MIMIC-CXR,OpenI</td>
<tr>
        <td><code> <a href="https://www.sciencedirect.com/science/article/pii/S1532046422000156 ">X. Cai et.al </a></td></code>
                <td>abstractive</td>
                <td>fine-tuning</td>
                <td>SciBERT</td>
                <td>supervised</td>
                <td>CORD-19</td>
<tr>
        <td><code>   <a href="https://arxiv.org/pdf/2204.02208"> A. Yalunin et.al </a></td></code>
                <td>abstractive</td>
                <td>adaption+fine-tuning</td>
                <td>BERT,Longformer</td>
                <td>supervised</td>
                <td>-</td>
<tr>
        <td><code><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8378607/">  B. C. Wallace et.al</a></code> <a href="https://github.com/bwallace/RCT-summarization-data"> [code]</a></td>           
                <td>abstractive,multi-doc</td>
                <td>adaption+fine-tuning</td>
                <td>BART</td>
                <td>supervised</td>
                <td>RCT</td>
<tr>
        <td><code>   <a href="https://arxiv.org/pdf/2104.06486">  J. DeYoung et.al</a></code> <a href="https://github.com/allenai/ms2"> [code]</a></td> 
                <td>abstractive,multi-doc</td>
                <td>fine-tuning</td>
                <td>BART,Longformer</td>
                <td>supervised</td>
                <td>MSˆ2</td>
<tr>
        <td><code>  <a href="https://www.nature.com/articles/s41746-021-00437-0">A. Esteva et.al </a></td></code>
                <td>abstractive,multi-doc</td>
                <td>fine-tuning</td>
                <td>BERT,GPT-2</td>
                <td>supervised</td>
                <td>CORD-19</td>
<tr>
        <td><code><a href="https://aclanthology.org/2020.nlpcovid19-2.14.pdf">CAiRE-COVID</a></code> <a href="https://github.com/HLTCHKUST/CAiRE-COVID"> [code]</a></td> 
                <td>hybrid,multi-doc</td>
                <td>fine-tuning,feature-base</td>
                <td>ALBERT,BART</td>
                <td>un+supervised</td>
                <td>CORD-19</td>
<tr>
        <td><code><a href="https://aclanthology.org/2020.coling-main.63.pdf">HET</a></code> <a href="https://github.com/cuhksz-nlp/HET-MC"> [code]</a></td> 
                <td>extractive,dialogue</td>
                <td>fine-tuning</td>
                <td>BERT</td>
                <td>supervised</td>
                <td>HET-MC</td>
<tr>
        <td><code><a href="https://aclanthology.org/2021.acl-long.384.pdf">CLUSTER2SENT</code> <a href="https://github.com/acmi-lab/modular-summarization"> [code]</a></td> 
                <td>abstractive,dialogue</td>
                <td>fine-tuning</td>
                <td>BERT,T5</td>
                <td>supervised</td>
                <td>-</td>
<tr>
        <td><code><a href="https://www.cs.cmu.edu/~mgormley/papers/zhang+al.emnlp.2021.pdf">L. Zhang et.al </a> </code> <a href="https://github.com/negrinho/medical_conversation_summarization"> [code]</a></td> 
                <td>abstractive,dialogue</td>
                <td>fine-tuning</td>
                <td>BART</td>
                <td>supervised</td>
                <td>-</td>
<tr>
        <td><code><a href='https://aclanthology.org/2021.nlpmc-1.9.pdf'>B. Chintagunt et.al</a></td></code>
                <td>abstractive,dialogue</td>
                <td>fine-tuning</td>
                <td>GPT-3</td>
                <td>supervised</td>
                <td>-</td>
<tr>
        <td><code><a href='https://aclanthology.org/2022.naacl-srw.32.pdf'>  D. F. Navarro et.al</a></td></code>
                <td>abstractive,dialogue</td>
                <td>fine-tuning</td>
                <td>BART,T5, PEGASUS</td>
                <td>supervised</td>
                <td>-</td>
<tr>
        <td><code><a href='https://aclanthology.org/2022.bionlp-1.9.pdf'>BioBART</a></code> <a href="https://github.com/GanjinZero/BioBART"> [code]</a></td> 
                <td>abstractive,dialogue</td>
                <td>fine-tuning</td>
                <td>BioBART</td>
                <td>supervised</td>
                <td>-</td>
<tr>
        <td><code><a href='https://aclanthology.org/2021.bionlp-1.12.pdf'>Y. He et.al</a></td></code>
                <td>abstractive,question</td>
                <td>fine-tuning</td>
                <td>BART,T5,PEGASUS</td>
                <td>supervised</td>
                <td>MeQSum,MIMIC-CXR,OpenI</td> 
<tr>
        <td><code><a href='https://aclanthology.org/2021.acl-short.33.pdf'>S. Yadav et.al</a></td></code>
                <td>abstractive,question</td>
                <td>fine-tuning</td>
                <td>BERT,ProphetNet</td>
                <td>supervised</td>
                <td>MeQSum</td>
<tr>
        <td><code><a href='https://arxiv.org/pdf/2106.00219.pdf'>S. Yadav et.al</a></td></code>
                <td>abstractive,question</td>
                <td>adaption+fine-tuning</td>
                <td><a href="https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html">Minilm</a></td>
                <td>supervised</td>
                <td>MeQSum</td>
<tr>
        <td><code><a href='https://aclanthology.org/2021.acl-long.119.pdf'> K. Mrini et.al</a></code> <a href="https://github.com/KhalilMrini/">[code]</a></td> 
                <td>abstractive,question</td>
                <td>adaption+fine-tuning</td>
                <td>BART,BioBERT</td>
                <td>supervised</td>
                <td>MeQSum</td>

        
</tbody >
</table>
"-" in Dataset stands for "not accessible"
</div>



## Evaluation
### Common metrics
[ROUGE](https://aclanthology.org/W04-1013.pdf): 

* ROUGE-N: N-gram overlap between generated summaries of summarizers and gold summaries(relevancy)
* ROUGE-L: the longest common subsequences between generated summaries of summarizers and gold summaries(fluency)

[BertScore](https://arxiv.org/pdf/1904.09675)

### Factual Consistency

Automatic:

* [CheXbert](https://aclanthology.org/2020.emnlp-main.117/) check binary presence values of disease variables
* [Jensen-Shannon Distance](https://github.com/allenai/ms2/) check directions(increase, decrease, no change)

Human Involved

* [Facts Counting](https://arxiv.org/pdf/2104.04412.pdf) 
* [Correctness of PICO and direction](https://aclanthology.org/2022.acl-long.350/)

## Leader Board
### Pubmed

<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Model</th>
    <th>ROUGE-1</th>
    <th>ROUGE-2</th>
    <th>ROUGE-L</th>
    <th>Paper</th>
    <th>Code Like</th>
    <th>Source</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> Top Down Transformer(AdaPool)                     </td></code>
		<td> 51.05       </td>
		<td> 23.26    </td>
		<td> 46.47 </td>
		<td> Long Document Summarization with Top-down and Bottom-up Inference (https://arxiv.org/pdf/2203.07586v1.pdf) </td>
		<td> https://github.com/kangbrilliant/DCA-Net      </td>
		<td>arxiv</td>
</tr>
<tr>
	<td><code> LongT5                   </td></code>
		<td> 50.23       </td>
		<td> 24.76    </td>
		<td> 46.67 </td>
		<td> LongT5: Efficient Text-To-Text Transformer for Long Sequences (https://arxiv.org/pdf/2112.07916v2.pdf) </td>
		<td> https://github.com/google-research/longt5      </td>
		<td>NAACL</td>
</tr>
<tr>
	<td><code>MemSum (extractive)                   </td></code>
		<td> 49.25	       </td>
		<td> 22.94    </td>
		<td> 44.42 </td>
		<td> MemSum: Extractive Summarization of Long Documents Using Multi-Step Episodic Markov Decision Processes(https://arxiv.org/pdf/2107.08929v2.pdf) </td>
		<td> https://github.com/nianlonggu/memsum     </td>
		<td>ACL</td>
</tr>
<tr>
	<td><code> HAT-BART                 </td></code>
		<td> 48.25       </td>
		<td> 21.35 </td>
		<td> 36.69 </td>
		<td> Hierarchical Learning for Generation with Long Source Sequences(https://arxiv.org/pdf/2104.07545v2.pdf) </td>
		<td>      </td>
		<td>arxiv</td>
</tr>
<tr>
	<td><code> 	DeepPyramidion                    </td></code>
		<td> 47.81       </td>
		<td> 21.14    </td>
		<td> 46.47 </td>
		<td> Sparsifying Transformer Models with Trainable Representation Pooling
(https://aclanthology.org/2022.acl-long.590.pdf) </td>
		<td> https://github.com/applicaai/pyramidions      </td>
		<td><ACL/td>
</tr>
<tr>
	<td><code> 	HiStruct+                   </td></code>
		<td> 46.59       </td>
		<td> 20.39    </td>
		<td> 42.11 </td>
		<td> HiStruct+: Improving Extractive Text Summarization with Hierarchical Structure Information(https://aclanthology.org/2022.findings-acl.102.pdf) </td>
		<td>       </td>
		<td>acl</td>
</tr>
<tr>
	<td><code> DANCER PEGASUS                    </td></code>
		<td> 46.34       </td>
		<td> 19.97    </td>
		<td> 42.42 </td>
		<td> A Divide-and-Conquer Approach to the Summarization of Long Documents[[pdf]](https://arxiv.org/pdf/2004.06190v3.pdf) </td>
		<td> https://github.com/AlexGidiotis/DANCER-summ      </td>
		<td>IEEE/ACM Transactions on Audio Speech and Language Processing</td>
</tr>
<tr>
	<td><code> 	BigBird-Pegasus                     </td></code>
		<td> 46.32       </td>
		<td> 20.65    </td>
		<td> 42.33 </td>
		<td> Big Bird: Transformers for Longer Sequences(https://arxiv.org/pdf/2007.14062v2.pdf) </td>
		<td> https://github.com/google-research/bigbird      </td>
		<td>NeuralIPS</td>
</tr>
<tr>
	<td><code> ExtSum-LG+MMR-Select+                    </td></code>
		<td> 45.39       </td>
		<td> 20.37    </td>
		<td> 40.99 </td>
		<td> Systematically Exploring Redundancy Reduction in Summarizing Long Documents(https://arxiv.org/pdf/2012.00052v1.pdf) </td>
		<td> https://github.com/Wendy-Xiao/redundancy_reduction_longdoc     </td>
		<td>AACL</td>
</tr>
<tr>
	<td><code> 	ExtSum-LG+RdLoss                 </td></code>
		<td> 45.3       </td>
		<td> 20.42    </td>
		<td> 40.95 </td>
		<td> Systematically Exploring Redundancy Reduction in Summarizing Long Documents(https://arxiv.org/pdf/2012.00052v1.pdf) </td>
		<td> https://github.com/Wendy-Xiao/redundancy_reduction_longdoc     </td>
		<td>AACL</td>
</tr>
</tbody >
</table>
</div>


### MS^2

<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Model</th>
    <th>ROUGE-1</th>
    <th>ROUGE-2</th>
    <th>ROUGE-L</th>
    <th>Paper</th>
    <th>Code Like</th>
    <th>Source</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> DAMEN                 </td></code>
		<td> 28.95       </td>
		<td> 9.72    </td>
		<td> 21.83 </td>
		<td> MDiscriminative Marginalized Probabilistic Neural Method for
Multi-Document Summarization of Medical Literature (https://aclanthology.org/2022.acl-long.15.pdf) </td>
		<td> https://disi-unibo-nlp.github.io/projects/damen/      </td>
		<td>ACL</td>
</tr>
<tr>
	<td><code> BART Hierarchical                 </td></code>
		<td> 27.56       </td>
		<td> 9.40    </td>
		<td> 20.80 </td>
		<td> MSˆ2: A Dataset for Multi-Document Summarization of Medical Studies (https://aclanthology.org/2021.emnlp-main.594.pdf) </td>
		<td> https://github.com/allenai/ms2/      </td>
		<td>EMNLP</td>
</tr>
<tr>
	<td><code> LED Flat                 </td></code>
		<td> 26.89       </td>
		<td> 8.91    </td>
		<td> 20.32 </td>
		<td> MSˆ2: A Dataset for Multi-Document Summarization of Medical Studies (https://aclanthology.org/2021.emnlp-main.594.pdf) </td>
		<td> https://github.com/allenai/ms2/      </td>
		<td> EMNLP </td>
</tr>
</tbody >
</table>
</div>

### MIMIC-CXR

<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Model</th>
    <th>ROUGE-1</th>
    <th>ROUGE-2</th>
    <th>ROUGE-L</th>
    <th>Paper</th>
    <th>Code Like</th>
    <th>Source</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code> ClinicalBioBERTSumAbs            </td></code>
		<td> 58.97       </td>
		<td> 47.06    </td>
		<td> 57.37 </td>
		<td> Predicting Doctor’s Impression For Radiology
Reports with Abstractive Text Summarization (https://web.stanford.edu/class/cs224n/reports/final_reports/report005.pdf) </td>
		<td>       </td>
		<td> Stanford CS224N </td>
</tr>
<tr>
	<td><code> DAMEN            </td></code>
		<td> 53.57       </td>
		<td> 40.78    </td>
		<td> 51.81 </td>
		<td> Attend to Medical Ontologies: Content Selection for
Clinical Abstractive Summarization (https://aclanthology.org/2020.acl-main.172.pdf) </td>
		<td>       </td>
		<td> ACL </td>
</tr>
</tbody >
</table>
</div>

### MEQSum

<div style="overflow-x: auto; overflow-y: auto; height: auto; width:100%;">
<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Model</th>
    <th>ROUGE-1</th>
    <th>ROUGE-2</th>
    <th>ROUGE-L</th>
    <th>Paper</th>
    <th>Code</th>
    <th>Source</th>
  </tr>
</thead>
<tbody >
<tr>
	<td><code>Gradually Soft MTL + Data Aug            </td></code>
		<td> 54.5       </td>
		<td> 37.9    </td>
		<td> 50.2 </td>
		<td> A Gradually Soft Multi-Task and Data-Augmented Approach
to Medical Question Understanding (https://aclanthology.org/2021.acl-long.119.pdf) </td>
		<td>   https://github.com/KhalilMrini/Medical-Question-Understanding    </td>
		<td>  ACL</td>
</tr>
<tr>
	<td><code> Explicit QTA Knowledge-Infusion            </td></code>
		<td> 45.20       </td>
		<td> 28.38    </td>
		<td> 48.76 </td>
		<td> Question-aware Transformer Models for Consumer
Health Question Summarization (https://arxiv.org/pdf/2106.00219.pdf) </td>
		<td>       </td>
		<td>  J. Biomed. Informatics</td>
</tr>
<tr>
	<td><code> ProphetNet + QTR + QFR            </td></code>
		<td> 45.52       </td>
		<td> 27.54    </td>
		<td> 48.19 </td>
		<td> Reinforcement Learning for Abstractive Question Summarization with
Question-aware Semantic Rewards (https://aclanthology.org/2021.acl-short.33.pdf) </td>
		<td>   https://github.com/shwetanlp/CHQ-Summ    </td>
		<td>  ACL</td>
</tr>
<tr>
	<td><code> ProphetNet + QFR            </td></code>
		<td> 45.36       </td>
		<td> 27.33    </td>
		<td> 47.96 </td>
		<td> Reinforcement Learning for Abstractive Question Summarization with
Question-aware Semantic Rewards (https://aclanthology.org/2021.acl-short.33.pdf) </td>
		<td>   https://github.com/shwetanlp/CHQ-Summ    </td>
		<td>  ACL</td>
</tr>
<tr>
	<td><code> Multi-Cloze Fusion           </td></code>
		<td> 44.58      </td>
		<td> 27.02   </td>
		<td> 47.81 </td>
		<td> Question-aware Transformer Models for Consumer
Health Question Summarization (https://arxiv.org/pdf/2106.00219.pdf) </td>
		<td>       </td>
		<td> J. Biomed. Informatics </td>
</tr>
<tr>
	<td><code> ProphetNet + QTR          </td></code>
		<td> 44.60      </td>
		<td> 26.69   </td>
		<td> 47.38 </td>
		<td> Reinforcement Learning for Abstractive Question Summarization with
Question-aware Semantic Rewards (https://aclanthology.org/2021.acl-short.33.pdf) </td>
		<td>   https://github.com/shwetanlp/CHQ-Summ    </td>
		<td>  ACL</td>
</tr>
<tr>
	<td><code> Implicit QTA Knowledge-Infusion           </td></code>
		<td> 44.44      </td>
		<td>  26.98  </td>
		<td> 47.66 </td>
		<td> Question-aware Transformer Models for Consumer
Health Question Summarization (https://arxiv.org/pdf/2106.00219.pdf) </td>
		<td>       </td>
		<td> J. Biomed. Informatics </td>
</tr>
<tr>
	<td><code> Minilm           </td></code>
		<td> 43.13      </td>
		<td>  26.03  </td>
		<td> 46.39 </td>
		<td> Minilm: Deep self-attention distillation for task-agnostic compression of pretrained transformers (https://arxiv.org/pdf/2106.00219.pdf) </td>
		<td>    https://github.com/microsoft/unilm/tree/master/minilm   </td>
		<td>  NIPS </td>
</tr>

</tbody >
</table>
</div>




