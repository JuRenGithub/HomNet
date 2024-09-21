# HomNet: Chromosomal Structural Abnormality Diagnosis by Homologous Similarity

[Juren Li](mailto:jrlee@zju.edu.cn)$^{1\bullet}$,
[Fanzhe Fu](mailto:ffanz@zju.edu.cn)$^{1\bullet}$,
[Ran Wei](mailto:ranwei@diagens.com)$^2$,
[Yifei Sun](mailto:yifeisun@zju.edu.cn)$^1$,
[Zeyu Lai](mailto:jerrylai@zju.edu.cn)$^1$,
[Ning Song](mailto:ningsong@diagens.com)$^2$,
[Xin Chen](mailto:xin.21@intl.zju.edu.cn)$^3$,
[Yang Yang](mailto:yangya@zju.edu.cn)$^{1*}$. ($^\bullet$ Equal contribution; $^*$ Correspondence)

$^1$ College of Computer Science and Technology, Zhejiang University

$^2$ Hangzhou Diagens Biotechnology Co., Ltd., China

$^3$ Zhejiang University-University of Illinois Urbana-Champaign Institute

----

This repository provides the code of "Chromosomal Structural Abnormality Diagnosis by Homologous Similarity", which is accepted by the [**KDD'24**](https://kdd2024.kdd.org/) ADS track (20\% acceptance rate).

## HomNet
<!-- [![PaperPDF](https://img.shields.io/badge/Paper-PDF-red)](https://yangy.org/works/application/KDD24_Chromosome.pdf) -->
[![PaperPDF](https://img.shields.io/badge/Paper-KDD-red)](https://dl.acm.org/doi/10.1145/3637528.3671642)
[![VideoBilibili](https://img.shields.io/badge/Video-Bilibili-pink)](https://www.bilibili.com/video/BV1JE421w7xq/?spm_id_from=333.999.0.0&vd_source=09cec99ad9b015a0bf557e4a4d8e06b9)
[![VideoACM](https://img.shields.io/badge/Video-ACM-FF8C00)](https://files.atypon.com/acm/0a1fb334f4d07744950577ba288726af)
[![Blog](https://img.shields.io/badge/推文-中文-green)](https://mp.weixin.qq.com/s/tPk0RMm0NUd4WHC2RjFvtQ)

![framework](./assets/framework.png)
This work proposes a method, HomNet, for diagnosing chromosomal structural abnormalities by leveraging homologous similarity. 
By aligning homologous chromosomes and considering information from multiple pairs simultaneously, HomNet can detect chromosomes with structural abnormalities.

## Code is no longer public

**Due to commercial considerations, the code is no longer available publicly.**

## Contact
If you have any question about the paper, feel free to contact me through Email: [jrlee@zju.edu.cn](mailto:jrlee@zju.edu.cn).

## Citation
If you find HomNet useful in your research or application, please kindly cite:
```
@inproceedings{HomNetljr2024,
author = {Li, Juren and Fu, Fanzhe and Wei, Ran and Sun, Yifei and Lai, Zeyu and Song, Ning and Chen, Xin and Yang, Yang},
title = {Chromosomal Structural Abnormality Diagnosis by Homologous Similarity},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671642},
doi = {10.1145/3637528.3671642},
abstract = {Pathogenic chromosome abnormalities are very common among the general population. While numerical chromosome abnormalities can be quickly and precisely detected, structural chromosome abnormalities are far more complex and typically require considerable efforts by human experts for identification. This paper focuses on investigating the modeling of chromosome features and the identification of chromosomes with structural abnormalities. Most existing data-driven methods concentrate on a single chromosome and consider each chromosome independently, overlooking the crucial aspect of homologous chromosomes. In normal cases, homologous chromosomes share identical structures, with the exception that one of them is abnormal. Therefore, we propose an adaptive method to align homologous chromosomes and diagnose structural abnormalities through homologous similarity. Inspired by the process of human expert diagnosis, we incorporate information from multiple pairs of homologous chromosomes simultaneously, aiming to reduce noise disturbance and improve prediction performance. Extensive experiments on real-world datasets validate the effectiveness of our model compared to baselines.},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {5317–5328},
numpages = {12},
keywords = {chromosome modeling, chromosome structural abnormality, deep neural networks, homologous similarity},
location = {Barcelona, Spain},
series = {KDD '24}
}
```
