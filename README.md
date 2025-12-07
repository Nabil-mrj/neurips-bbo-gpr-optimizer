# Black-Box Optimization with Gaussian Processes  
Project carried out as part of the NeurIPS 2020 BBO Post-Challenge

## Overview
This project is based on the NeurIPS 2020 Black-Box Optimization Challenge, which focuses on evaluating black-box optimization algorithms applied to real-world hyperparameter search tasks.  
The optimizer must interact with a function whose internal structure is unknown and progressively propose new hyperparameters to evaluate. The platform calls the `suggest(...)` method to obtain a candidate point and then invokes `observe(hp, R)` to update the optimizer with the resulting score.

Several variants of Gaussian Process Regression (GPR)–based optimizers were implemented and evaluated to study the impact of different acquisition strategies and uncertainty handling methods.

---

## Developed Methods

### 1. Gaussian Process Regression (GPR) – Baseline  
**Submission ID: 893732 — Score: 91.28**

This baseline relies on a Gaussian Process model combined with an Expected Improvement acquisition function.  
Hyperparameters are transformed depending on their type (categorical, boolean, integer, real, optionally log-scaled).  
The optimizer begins with an exploration phase before progressively focusing on the most promising regions based on accumulated observations.

### 2. GPR with Adaptive Exploration  
**Submission ID: 893739 — Score: 83.24**

This variant adjusts the exploration intensity according to the variance of recent observations.  
Low variance is interpreted as a sign of stagnation and increases exploration, while higher variance reduces it to concentrate on promising areas.  
In this challenge setting, this strategy did not improve performance and tended to cause excessive exploration.

### 3. GPR with Minimum Variance Threshold  
**Submission ID: 893749 — Score: 92.65**

This approach introduces a minimum variance threshold in the acquisition computation.  
When the predicted variance becomes extremely small, the GPR model may become overly confident and suppress exploration too strongly.  
By enforcing a lower bound, the optimizer remains more cautious, continues to explore when needed, and avoids premature convergence.  
This method achieved the best performance among the implemented variants.

---

## Results

### Overall Performance  
The model incorporating a minimum variance threshold shows a steady improvement of the mean score across iterations until it reaches a stable, high-performance plateau.  
This trend reflects a well-balanced trade-off between exploration and exploitation across the benchmark.

<img width="716" height="537" alt="overall-performance" src="https://github.com/user-attachments/assets/6e3f9c0f-2cc4-4d4d-b136-df922b5bb4ef" />

### Per-Task Performance  
The evolution on the “gina” task highlights a rapid improvement during the early iterations, followed by more gradual refinement.  
This behavior shows that the optimizer quickly identifies promising regions before stabilizing around the most effective configurations.

<img width="713" height="533" alt="gina-performance" src="https://github.com/user-attachments/assets/6e155b14-adaf-4f37-a9a3-3a0bdc0ec6d0" />

---

## Conclusion
The experiments demonstrate that controlling the predicted variance of the model plays a key role in the stability of black-box optimization.  
The minimum variance threshold proved to be an effective mechanism to prevent overconfidence and sustain meaningful exploration, leading to the best results in this project.  
Conversely, adaptive exploration was less suited to this context, likely due to overly aggressive adjustments of the exploration level.

---

## Included Implementations
- GPR Optimizer: `optimizer_GPR.py`  
- Adaptive Exploration GPR Optimizer: `optimizer_GPR_adaptive.py`  
- Variance Threshold GPR Optimizer: `optimizer_GPR_threshold.py`
