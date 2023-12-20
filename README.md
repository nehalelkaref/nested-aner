<img width="1259" alt="ANER" src="https://github.com/nehalelkaref/nested-aner/assets/16616024/9729921a-978b-4a86-9b3d-2fb5c8bc3494">

# Nested Arabic NER
Classifying NESTED entities on Arabic data using _StagedNER_ and _Biaffine classifier_ inspired by [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577/) paper

Data: [Wojood corpus](https://aclanthology.org/2022.lrec-1.387.pdf)
  - consists of 21 entity types
  - Majority of data comes in MSA while the rest is in the Levant dialects (Palestinian and Lebanese)

Method: 
  - Staged NER trains two instances of the same transformer
  - One transformer is trained on classifying BIO spans in the first stage and the second transformer is trained on classifying entity types in the second stage
  - For NESTED ANER, we employ a biaffine classifer to classify BIO spans of NESTED entities while the second stage remains as is
  - to reproduce stagedNER checkout our paper [here](https://aclanthology.org/2023.arabicnlp-1.91/)

Demo: (coming soon)

    
*Code in the repositroy provides implementation only for biaffine classifier and post-processing constraints described [here](https://aclanthology.org/2020.acl-main.577/)*
