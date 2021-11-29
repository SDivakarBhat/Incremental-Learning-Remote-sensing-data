# CILEA-NET: Curriculum-Based Incremental Learning Framework for Remote Sensing Image Classification

Official implementation of [[CILEA_NET](https://ieeexplore.ieee.org/abstract/document/9442875)].


<p align='center'>
    <img src="semgif-readme.png", width="700">
</p>


## Description


In this article, we address class incremental learning (IL) in remote sensing image analysis. Since remote sensing images are acquired continuously over time by Earth's observation sensors, the land-cover/land-use classes on the ground are likely to be found in a gradational manner. This process restricts the deployment of stand-alone classification approaches, which are trained for all the classes together in one iteration. Therefore, for every new set of categories discovered, the entire network consisting of old and new classes requires retraining. This procedure is often impractical, considering vast volumes of data, limited resources, and the complexity of learning models. In this respect, we propose a convolutional-neural-network-based framework (called CILEA-NET, curriculum-based incremental learning framework for remote sensing image classification) to efficiently resolve the difficulties associated with incremental learning paradigm. The framework includes new classes in the already trained model to avoid catastrophic forgetting for the old while ensuring improved generalization for the newly added classes. To manage the IL's stability-plasticity dilemma, we introduce a novel curriculum learning-based approach where the order of the new classes is devised based on their similarity to the already trained classes. We then perform the training in that given order. We notice that the curriculum learning setup distinctly enhances the training time for the new classes. Experimental results on several optical datasets: PatternNet and NWPU-RESISC45, and a hyperspectral dataset: Indian Pines, validate the robustness of our technique.
## Getting Started

### Dependencies

* python3
* pytorch

<!--### Executing program

* To run the main 
* Step-by-step bullets
```
code blocks for commands
```-->


## Citation
> Bhat, S. Divakar, Biplab Banerjee, Subhasis Chaudhuri, and Avik Bhattacharya. "CILEA-NET: A Curriculum-driven Incremental Learning Network for Remote Sensing Image Classification." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2021). [[CILEA_NET](https://ieeexplore.ieee.org/abstract/document/9442875)]

If you want to cite, use the following Bibtex entry:
```
@article{bhat2021cilea,
  title={CILEA-NET: A Curriculum-driven Incremental Learning Network for Remote Sensing Image Classification},
  author={Bhat, S Divakar and Banerjee, Biplab and Chaudhuri, Subhasis and Bhattacharya, Avik},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```
