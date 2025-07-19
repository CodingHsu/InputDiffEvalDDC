# Evaluation Algorithm for Optimal Dataset Discovery in Neural Differential Distinguishers

This repository contains the code and resources accompanying the paper:

**"A Novel Evaluation Algorithm for Identifying Optimal Datasets in Differential‑based Neural Distinguisher"**
*IEEE Transactions on Information Forensics and Security*, Vol. XX, No. XX, 20XX
**Haiyi Xu, Lei Zhang, Yufei Yuan**

---

## Overview

This repository provides implementations of analysis tools and experimental pipelines supporting the evaluation and selection of high‑quality datasets for training differential‑based neural distinguishers (NDs). It focuses on identifying optimal differential inputs and understanding ND behaviour, particularly on the SPECK32/64 block cipher.

---

## Repository Structure

- 
  Supporting code and experiments related to Section 3 of the paper (Empirical Observations).

  - `last_two_rounds_differential_distribution_predict/` — Investigating ND’s recognition of differential patterns.
  - `distinguishing_capability_of_nd/` — Analysing the distinguishing capability of NDs.
  - `differential_distribution_analysis/` — Differential distribution analysis on 3‑round SPECK32/64.
  - `speed_test/` — Runtime comparison of various baseline methods versus the three methods.

- 
  Code implementations related to Sections 4 and 5 of the paper (Algorithm Design and Evaluation).

  - `best_differential_search/` — Full implementation of the two‑stage evaluation algorithm for selecting optimal differential datasets.
  - `validate_search_results/` — Tools for validating the differential inputs found by the evaluation algorithm.

- 
  Experimental framework for an in‑depth comparison between **DDC degree** and **Bias score** on multiple ciphers.

  - `ciphers/`, `des_search_results/`, `simon3264_search_results/`, `speck3264_search_results/` — Data and intermediate results.
  - `validate_search_result/` — Validation utilities.
  - `ddc_vs_bias.py` — Entry‑point script to reproduce all comparison experiments.

---

## Prerequisites

- **Python** 3.8.16
- **TensorFlow‑GPU** 2.6.0
- **CUDA Toolkit** 11.3.1
- **cuDNN** 8.2.1
- **NumPy** 1.20.3
- Additional dependencies listed in `requirements.txt`

Ensure your environment has a compatible GPU with the corresponding CUDA/cuDNN versions to support TensorFlow‑GPU.

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{xu2025optimal,
  title={A Novel Evaluation Algorithm for Identifying Optimal Datasets in Differential‑based Neural Distinguisher},
  author={Haiyi Xu and Lei Zhang and Yufei Yuan},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={XX},
  number={XX},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or issues, please open an issue on this repository or contact the authors at [1455096897@qq.com](mailto:1455096897@qq.com).
