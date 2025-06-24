# Sim_Fit
Fine-tuning a Sentence Transformer directly with classification objectives. It can be used as a pwoerful retriver, or even as a standalone classifier!

These codes were used to produce results in this paper:
https://www.sciencedirect.com/science/article/pii/S1474034625004021

The codes were built mostly on the HuggingFace Transformer and Sentence Transformer libraries. The `SimFit` folder contains codes to train the models and infer with them as standalone classifiers. Multiple custom losses and evaluators were implemented for the two classifier architectures proposed in the paper. The best results utilized the cross entropy loss, which  reformulated similarity computation as a classification task where normalized Euclidean distances were used as effective logits.


