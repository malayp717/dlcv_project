
# Defense against Adversarial Attacks

Deep Neural Networks are notoriously known for being very overconfident in their predictions. Szegedy et. al. \cite{DBLP:journals/corr/HeZRS15} discovered that Deep Neural Networks can be fooled into making wrong predictions by adding small perturbations to the original image. In our project, we aim to make
targeted classifier models more robust to adversarial attacks. We train an autoencoder, and use this model to effectively counter the adversarial perturbations that have been added to the input image. We also explore defense by generating images that are as close as
possible to the input adversarial image. We implement our own attacks and train
our own baselines to ensure uniform comparison.