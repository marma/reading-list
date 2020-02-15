# Reading list

## RDF2Vec: RDF Graph Embeddingsand Their Applications
Petar Ristoski, Jessica Rosati, Tommaso Di Noia, Renato De Leone, Heiko Paulheim

> "Linked Open Data has been recognized as a valuable source for background information in many data mining andinformation retrieval tasks. However, most of the existing tools require features in propositional form, i.e., a vector of nominal ornumerical features associated with an instance, while Linked Open Data sources are graphs by nature. In this paper, we presentRDF2Vec, an approach that uses language modeling approaches for unsupervised feature extraction from sequences of words,and adapts them to RDF graphs. We generate sequences by leveraging local information from graph sub-structures, harvested byWeisfeiler-Lehman Subtree RDF Graph Kernels and graph walks, and learn latent numerical representations of entities in RDFgraphs. We evaluate our approach on three different tasks: (i) standard machine-learning tasks (ii) entity and document modeling(iii) content-based recommender systems. The evaluation shows that the proposed entity embeddings outperform existing tech-niques, and that feature vector representations of general knowledge graphs such as DBpedia and Wikidata can be easily reusedfor different tasks."

http://www.semantic-web-journal.net/system/files/swj1495.pdf

## Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia
Ikuya Yamada, Akari Asai, Jin Sakuma, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, Yuji Matsumoto

> "The embeddings of entities in a large knowledge base (e.g., Wikipedia) are highly beneficial for solving various natural language tasks that involve real world knowledge. In this paper, we present Wikipedia2Vec, a Python-based open-source tool for learning the embeddings of words and entities from Wikipedia. The proposed tool enables users to learn the embeddings efficiently by issuing a single command with a Wikipedia dump file as an argument. We also introduce a web-based demonstration of our tool that allows users to visualize and explore the learned embeddings. In our experiments, our tool achieved a state-of-the-art result on the KORE entity relatedness dataset, and competitive results on various standard benchmark datasets. Furthermore, our tool has been used as a key component in various recent studies. We publicize the source code, demonstration, and the pretrained embeddings for 12 languages at this https URL."

https://arxiv.org/abs/1812.06280

## Combining Word and Entity Embeddings for Entity Linking
Jose G. Moreno, Romaric Besancon, Romain Beaumont, Eva D’hondt, Anne-Laure Ligozat, Sophie Rosset, Xavier Tannier, and Brigitte Grau

> "The correct identification of the link between an entity men-tion in a text and a known entity in a large knowledge base is importantin information retrieval or information extraction. The general approachfor this task is to generate, for a given mention, a set of candidate en-tities from the base and, in a second step, determine which is the bestone. This paper proposes a novel method for the second step which isbased on the joint learning of embeddings for the words in the text andthe entities in the knowledge base. By learning these embeddings in thesame space we arrive at a more conceptually grounded model that canbe used for candidate selection based on the surrounding context. Therelative improvement of this approach is experimentally validated on arecent benchmark corpus from the TAC-EDL 2015 evaluation campaign."

https://perso.limsi.fr/bg/fichiers/2017/combining-word-entity-eswc2017.pdf

## Entity-aware ELMo: Learning Contextual Entity Representation for Entity Disambiguation
Hamed Shahbazi, Xiaoli Z. Fern, Reza Ghaeini,Rasha Obeidat, Prasad Tadepalli

> "We present a new local entity disambiguation system.  The key to our system is anovel approach for learning entity represen-tations.  In our approach we learn an entityaware extension of Embedding for LanguageModel (ELMo) which we call Entity-ELMo(E-ELMo). Given a paragraph containing oneor more named entity mentions, each men-tion is first defined as a function of the en-tire paragraph (including other mentions), thenthey predict the referent entities. Utilizing E-ELMo for local entity disambiguation, we out-perform all of the state-of-the-art local andglobal models on the popular benchmarks byimproving about 0.5% on micro average accuracy for AIDA test-b with Yago candidate set.The evaluation setup of the training data andcandidate set are the same as our baselines forfair comparison."

https://arxiv.org/abs/1908.05762

## Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking
Samuel Broscheit

> "A typical architecture for end-to-end entity linking systems consists of three steps: mention detection, candidate generation and entity disambiguation. In this study we investigate the following questions: (a) Can all those steps be learned jointly with a model for contextualized text-representations, i.e. BERT? (b) How much entity knowledge is already contained in pretrained BERT? (c) Does additional entity knowledge improve BERT’s performance in downstream tasks? To this end we propose an extreme simplification of the entity linking setup that works surprisingly well: simply cast it as a per token classification over the entire entity vocabulary (over 700K classes in our case). We show on an entity linking benchmark that (i) this model improves the entity representations over plain BERT, (ii) that it outperforms entity linking architectures that optimize the tasks separately and (iii) that it only comes second to the current state-of-the-art that does mention detection and entity disambiguation jointly. Additionally, we investigate the usefulness of entity-aware token-representations in the text-understanding benchmark GLUE, as well as the question answering benchmarks SQUAD~V2 and SWAG and also the EN-DE WMT14 machine translation benchmark. To our surprise, we find that most of those benchmarks do not benefit from additional entity knowledge, except for a task with very small training data, the RTE task in GLUE, which improves by 2%."

https://www.aclweb.org/anthology/K19-1063/

## Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering

> "Recently, multilingual question answering became a crucial research topic, and it is receiving increased interest in the NLP community. However, the unavailability of large-scale datasets makes it challenging to train multilingual QA systems with performance comparable to the English ones. In this work, we develop the Translate Align Retrieve (TAR) method to automatically translate the Stanford Question Answering Dataset (SQuAD) v1.1 to Spanish. We then used this dataset to train Spanish QA systems by fine-tuning a Multilingual-BERT model. Finally, we evaluated our QA models with the recently proposed MLQA and XQuAD benchmarks for cross-lingual Extractive QA. Experimental results show that our models outperform the previous Multilingual-BERT baselines achieving the new state-of-the-art value of 68.1 F1 points on the Spanish MLQA corpus and 77.6 F1 and 61.8 Exact Match points on the Spanish XQuAD corpus. The resulting, synthetically generated SQuAD-es v1.1 corpora, with almost 100% of data contained in the original English version, to the best of our knowledge, is the first large-scale QA training resource for Spanish."

https://www.researchgate.net/publication/337904607_Automatic_Spanish_Translation_of_the_SQuAD_Dataset_for_Multilingual_Question_Answering

## The Unreasonable Effectiveness of Recurrent Neural Networks

> "There’s something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I’ve in fact reached the opposite conclusion). Fast forward about a year: I’m training RNNs all the time and I’ve witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me. This post is about sharing some of that magic with you."

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut

> "Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large."

https://arxiv.org/abs/1909.11942

## Fully Convolutional Neural Networks for Newspaper Article Segmentation

> "Segmenting newspaper pages into articles that semantically belong together is a necessary prerequisite for article-based information retrieval on print media collections like e.g. archives and libraries. It is challenging due to vastly differing layouts of papers, various content types and different languages, but commercially very relevant for e.g. media monitoring. We present a semantic segmentation approach based on the visual appearance of each page. We apply a fully convolutional neural network (FCN) that we train in an end-to-end fashion to transform the input image into a segmentation mask in one pass. We show experimentally that the FCN performs very well: it outperforms a deep learning-based commercial solution by a large margin in terms of segmentation quality while in addition being computationally two orders of magnitude more efficient."

https://pd.zhaw.ch/publikation/upload/212962.pdf

## Attention Is All You Need

> "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."

https://arxiv.org/abs/1706.03762

## Music Transformer: Generating Music with Long-Term Structure

> "Generating long pieces of music is a challenging problem, as music contains structure at multiple timescales, from milisecond timings to motifs to phrases to repetition of entire sections. We present Music Transformer, an attention-based neural network that can generate music with improved long-term coherence"

https://magenta.tensorflow.org/music-transformer

## How to Fine-Tune BERT for Text Classification?

> "Language model pre-training has proven to be
useful in learning universal language representations. As a state-of-the-art language model
pre-training model, BERT (Bidirectional Encoder Representations from Transformers) has
achieved amazing results in many language understanding tasks. In this paper, we conduct exhaustive experiments to investigate different fine-tuning methods of BERT on text classification task and provide a general solution for BERT fine-tuning. Finally, the proposed solution obtains new state-of-the-art results on eight widely-studied text classification
datasets."

* https://arxiv.org/pdf/1905.05583.pdf

## Bayesian Mixture Models on Connected Components for Newspaper Article Segmentation

> "In this paper we propose a new method for automated segmentation of scanned newspaper pages into articles. Article regions are produced as a result of merging sub-article level content and title regions. We use a Bayesian Gaussian mixture model to model page Connected Component information and cluster input into sub-article components. The Bayesian model is conditioned on a prior distribution over region features, aiding classification into titles and content. Using a Dirichlet prior we are able to automatically estimate correctly the number of title and article regions. The method is tested on a dataset of digitized historical newspapers, where visual experimental results are very promising."

https://dl.acm.org/citation.cfm?doid=2960811.2967165

## Unsupervised Data Augmentation for Consistency Training

> "Semi-supervised learning lately has shown much promise in improving deep learning models when labeled data is scarce. Common among recent approaches is the use of consistency training on a large amount of unlabeled data to constrain model predictions to be invariant to input noise. In this work, we present a new perspective on how to effectively noise unlabeled examples and argue that the quality of noising, specifically those produced by advanced data augmentation methods, plays a crucial role in semi-supervised learning. By substituting simple noising operations with advanced data augmentation methods, our method brings substantial improvements across six language and three vision tasks under the same consistency training framework. On the IMDb text classification dataset, with only 20 labeled examples, our method achieves an error rate of 4.20, outperforming the state-of-the-art model trained on 25,000 labeled examples. On a standard semi-supervised learning benchmark, CIFAR-10, our method outperforms all previous approaches and achieves an error rate of 2.7% with only 4,000 examples, nearly matching the performance of models trained on 50,000 labeled examples. Our method also combines well with transfer learning, e.g., when finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used."

https://arxiv.org/abs/1904.12848

## Speeding up BERT

> "BERT became an essential ingredient of many NLP deep learning pipelines. It is considered a milestone in NLP, as ResNet is in the computer vision field. The only problem with BERT is its size. BERT-base is model contains 110M parameters. The larger variant BERT-large contains 340M parameters. It’s hard to deploy a model of such size into many environments with limited resources, such as a mobile or embedded systems."

https://blog.inten.to/speeding-up-bert-5528e18bb4ea

## DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

> "As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the-edge and/or under constrained computational training or inference budgets remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device study."

https://arxiv.org/abs/1910.01108

## TinyBERT: Distilling BERT for Natural Language Understanding

> "Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks. However, pre-trained language models are usually computationally expensive and memory intensive, so it is difficult to effectively execute them on some resource-restricted devices. To accelerate inference and reduce model size while maintaining accuracy, we firstly propose a novel transformer distillation method that is a specially designed knowledge distillation (KD) method for transformer-based models. By leveraging this new KD method, the plenty of knowledge encoded in a large teacher BERT can be well transferred to a small student TinyBERT. Moreover, we introduce a new two-stage learning framework for TinyBERT, which performs transformer distillation at both the pre-training and task-specific learning stages. This framework ensures that TinyBERT can capture both the general-domain and task-specific knowledge of the teacher BERT.TinyBERT is empirically effective and achieves more than 96% the performance of teacher BERTBASE on GLUE benchmark while being 7.5x smaller and 9.4x faster on inference. TinyBERT is also significantly better than state-of-the-art baselines on BERT distillation, with only about 28% parameters and about 31% inference time of them."

https://arxiv.org/abs/1909.10351

## Text Similarities : Estimate the degree of similarity between two texts

https://medium.com/@adriensieg/text-similarities-da019229c894

## Misc:
* https://towardsdatascience.com/the-most-important-supreme-court-decision-for-data-science-and-machine-learning-44cfc1c1bcaf
* https://arxiv.org/pdf/1210.0999.pdf

