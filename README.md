# Traffic-prediction-Hybrid-Model
Advanced Hybrid model for predicting traffic congestion


This project implements a **hybrid traffic speed prediction model** using real-world traffic data. The code predicts traffic speed **1 hour ahead** by combining two approaches: a **historical pattern-matching model** and a **Bayesian residual correction model**. The hybrid of these two provides a more accurate prediction than either model individually.

Traffic congestion refers to a condition where traffic slows down significantly due to excess vehicle demand compared to road capacity. Predicting congestion early helps in traffic management, routing, and planning.

The dataset used in this project is the **METR-LA dataset**, a publicly available traffic dataset containing **5‑minute interval speed measurements** collected from inductive loop detectors on Los Angeles freeways. Since the dataset contains only speed values, density is estimated using a simple traffic-flow relationship.

The code includes:

* Data loading and preprocessing
* Historical sliding‑window prediction
* Residual computation
* Bayesian residual estimation
* Hybrid prediction (Historical + Bayesian)
* sMAPE evaluation and optional congestion analysis

This README provides an overview of the project, dataset, and methodology so users can understand and run the code comfortably.

