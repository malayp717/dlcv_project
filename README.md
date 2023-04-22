<div id="top"></div>

<br />
<div align="center">

<h2 align="center">Defense against Adversarial Attacksn</h2>

  <p align="center">
    Deep Neural Networks are notoriously known for being very overconfident in their predictions. Szegedy et. al. (https://arxiv.org/abs/1312.6199) discovered that Deep Neural Networks can be fooled into making wrong predictions by adding small perturbations to the original image. In our project, we aim to make targeted classifier models more robust to adversarial attacks. We train an autoencoder, and use this model to effectively counter the adversarial perturbations that have been added to the input image. We also explore defense by generating images that are as close as
possible to the input adversarial image. We implement our own attacks and train
our own baselines to ensure uniform comparison.
    <br />
    <br />
    <a href="https://github.com/malayp717/dlcv_project">View Demo</a>
    ·
    <a href="https://github.com/malayp717/dlcv_project/issues">Report Bug</a>
    ·
    <a href="https://github.com/malayp717/dlcv_project/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#problem-definition">Problem Definition</a>
      <ul>
        <li><a href="#corpus-description">Corpus Description</a></li>
      </ul>
    </li>
    <li>
      <a href="#proposed approach">Proposed Approach</a>
      <ul>
        <li><a href="#experiments-and-results">Experiments and Results</a></li>
        <li><a href="#future-directions">Future Direction</a></li>
      </ul>
    </li>
      </ol>
</details>

Deep Neural Networks are notoriously known for being very overconfident in their predictions. Szegedy et. al. \cite{DBLP:journals/corr/HeZRS15} discovered that Deep Neural Networks can be fooled into making wrong predictions by adding small perturbations to the original image. In our project, we aim to make
targeted classifier models more robust to adversarial attacks. We train an autoencoder, and use this model to effectively counter the adversarial perturbations that have been added to the input image. We also explore defense by generating images that are as close as
possible to the input adversarial image. We implement our own attacks and train
our own baselines to ensure uniform comparison.
