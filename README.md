# Evaluating Time Series Models for Urban Wastewater Management: Predictive Performance, Model Complexity and Resilience
Presented at the 10th International Conference on Smart and Sustainable Technologies (SpliTech 2025)

[Preprint](https://doi.org/10.48550/arXiv.2504.17461)

**Authors**: Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann

---

## Table of Contents

* [Abstract](#abstract)
* [Project Overview](#project-overview)
* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)
    * [Training](#training)
    * [Evaluation](#evaluation)
* [Results](#results)
* [Repository Structure](#repository-structure)
* [Related Publications](#related-publications)
* [Acknowledgement](#acknowledgement)
* [Contact](#contact)

---

## Abstract

Climate change increases the frequency of extreme rainfall, placing a significant strain on urban infrastructures, especially Combined Sewer Systems (CSS). Overflows from overburdened CSS release untreated wastewater into surface waters, posing environmental and public health risks. Although traditional physics-based models are effective, they are costly to maintain and difficult to adapt to evolving system dynamics. Machine Learning (ML) approaches offer cost-efficient alternatives with greater adaptability. To systematically assess the potential of ML for modeling urban infrastructure systems, we propose a protocol for evaluating Neural Network architectures for CSS time series forecasting with respect to predictive performance, model complexity, and robustness to perturbations. In addition, we assess model performance on peak events and critical fluctuations, as these are the key regimes for urban wastewater management. To investigate the feasibility of lightweight models suitable for IoT deployment, we compare global models, which have access to all information, with local models, which rely solely on nearby sensor readings. Additionally, to explore the security risks posed by network outages or adversarial attacks on urban infrastructure, we introduce error models that assess the resilience of models.

Our results demonstrate that while global models achieve higher predictive performance, local models provide sufficient resilience in decentralized scenarios, ensuring robust modeling of urban infrastructure. Furthermore, models with longer native forecast horizons exhibit greater robustness to data perturbations. These findings contribute to the development of interpretable and reliable ML solutions for sustainable urban wastewater management.

## Project Overview

- Comparison of 6 Neural Network architectures for time series forecasting
- Establishing global and local models for resliency against network outages:
![Global / Local Scenario](figures/global_local_scenario.png)
    - Global models: Access to all sensor data
    - Local models: Only use nearby sensor readings
- Robustness analysis of models against realistic errors:
![Realistic Error Types](figures/realistic_error_types.png)
    - Outliers: e. g. sensor miscalibration
    - Missing Values: e. g. mainetenance or network outages
    - Clipping: e. g. physical limitations of sensors
- Evaluation of model performance on peak events and critical fluctuations
- Holistic evaluation of model performance, complexity, and resilience

## Installation

1.  **Clone the repository:**
    ```bash
    git clone ...
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # Using venv
    python -m venv env
    source env/bin/activate # On Windows use `env\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    # Using pip
    pip install -r requirements.txt
    ```
    *Note: The key dependency for running the models is PyTorch with version 2.2.2.*

## Dataset

For carrying out the experiments and training our models we disposed of a real-world dataset of a Combined Sewer System in the city of Duisburg, Germany, provided by the *Wirtschaftsbetriebe Duisburg* (WBD).

**The full dataset cannot be made publicly available, because of information on critical infrastructure.**

For further details on the dataset, please refer to the paper.

## Usage

The code can be used through running the `main.py` script, which allows for training and inference of time series models.

To see all available command line arguments, run:
```bash
python main.py --help
```

### Training
To train a model, use the following command:
```bash
python main.py --data_filename=<path_to_data_csv> --target=<target_column> --future_compatible_covariate=<list_of_future_compatible_covariates> --model_type=<model_type>
```

Where:
- `<path_to_data_csv>`: Path to the CSV file containing the time series data.
- `<target_column>`: The column in the CSV file that contains the target variable to be predicted.
- `<list_of_future_compatible_covariates>`: A comma-separated list of covariates that are known in the future.
- `<model_type>`: The type of model to be trained (e.g., `tft`, `transformer`, `lstm`, `nhits`, `tcn`, `deepar`).

### Evaluation
To evaluate a trained model, use the following command:
```bash
python main.py --data_filename=<path_to_data_csv> --target=<target_column> --future_compatible_covariate=<list_of_future_compatible_covariates> --model_type=<model_type> --inference_model_path=<path_to_model> 
```

Where:
- `<path_to_data_csv>`: Path to the CSV file containing the time series data.
- `<target_column>`: The column in the CSV file that contains the target variable to be predicted.
- `<list_of_future_compatible_covariates>`: A comma-separated list of covariates that are known in the future.
- `<model_type>`: The type of model to be trained (e.g., `tft`, `transformer`, `lstm`, `nhits`, `tcn`, `deepar`).
- `<path_to_model>`: Path to the trained model file that you want to evaluate.

## Results

Here, we provide a summary of the results obtained from the experiments conducted with the different time series models on the dataset from the CSS in Duisburg, Germany. For the visualization and the discussion of the results, please refer to the paper.

- Global:
    | Model type  | MSE (q=0.25) | MSE (median) | MSE (q=0.75) | Median MSE at Peak Events | Inference time [ms] | Model Size [MB] |
    | :---------- | :----------: | :----------: | :----------: | :-----------------------: | :-----------------: | :-------------: |
    | TFT         |   **0.28** |   **0.30** |   **0.34** |         **0.67** |        2.36         |      11.36      |
    | Transformer |     0.60     |     0.61     |     0.61     |           1.38            |        0.87         |     144.26      |
    | LSTM        |     0.51     |     0.64     |     0.83     |           1.15            |      **0.81** |      0.32       |
    | N-HiTS      |     0.67     |     0.68     |     0.69     |           1.43            |        0.85         |    1158.38      |
    | TCN         |     0.97     |     0.98     |     0.99     |           1.90            |        0.84         |      0.07       |
    | DeepAR      |     1.14     |     1.31     |     1.56     |           2.07            |        0.88         |      1.72       |
- Local:
    | Model type  | MSE (q=0.25) | MSE (median) | MSE (q=0.75) | Median MSE at Peak Events | Inference time [ms] | Model Size [MB] |
    | :---------- | :----------: | :----------: | :----------: | :-----------------------: | :-----------------: | :-------------: |
    | TFT         |     0.48     |     0.50     |     0.53     |           1.23            |        0.94         |      6.22       |
    | Transformer |     0.62     |     0.63     |     0.64     |           1.41            |        0.88         |     184.59      |
    | LSTM        |     0.63     |     0.79     |     0.99     |           1.42            |      **0.81** |      0.16       |
    | N-HiTS      |   **0.48** |   **0.48** |   **0.49** |         **1.23** |        0.83         |      27.10      |
    | TCN         |     1.00     |     1.01     |     1.03     |           1.97            |        0.82         |      0.04       |
    | DeepAR      |     1.28     |     1.45     |     1.64     |           2.13            |        0.88         |      1.65       |

- Predictive Performance, Computational Complexity and Robustness (for visualizations refer to the paper):
    - Global:

        | Model | MSE | CCI | RI |
        | :--- | :--- | :--- | :--- |
        | TFT | **0.30** | 0.50 | 0.27 |
        | Transformer | 0.61 | 0.22 | 0.22 |
        | LSTM | 0.64 | **0.14** | 0.82 |
        | N-HiTS | 0.68 | 0.65 | 0.24 |
        | TCN | 0.98 | 0.18 | **0.12** |
        | DeepAR | 1.31 | 0.25 | 0.90 |

        ---
        - **MSE** = Mean Squared Error (lower is better)
        - **CCI** = Computational Complexity Index (includes inference time and model size, lower is better)
        - **RI** = Robustness Index (includes robustness to perturbations and IQR of the MSE, lower is better)

    - Local:

        | Model | MSE | CCI | IQR of the MSE |
        | :--- | :--- | :--- | :--- |
        | TFT | 0.50 | 0.32 | 0.06 |
        | Transformer | 0.63 | 0.73 | **0.01** |
        | LSTM | 0.79 | **0.17** | 0.36 |
        | N-HiTS | **0.48** | 0.26 | **0.01** |
        | TCN | 1.01 | 0.27 | 0.02 |
        | DeepAR | 1.45 | 0.50 | 0.35 |

        ---
        - **MSE** = Mean Squared Error (lower is better)
        - **CCI** = Computational Complexity Index (includes inference time and model size, lower is better)
        - **IQR** = Interquartile Range (lower is better)

## Repository Structure

The repository is structured as follows (*only relevant files displayed*):

* [`main.py`](./main.py): The main script and entry point for running experiments.
* [`requirements.txt`](./requirements.txt): Lists the Python packages required to run the code.
* [`README.md`](./README.md): Provides an overview of the project, setup instructions, and how to run the experiments.
* [`data/`](./data/): Contains scripts for data loading, processing, and exploratory data analysis (EDA).
    * [`TimeSeriesDatasetCreator.py`](./data/TimeSeriesDatasetCreator.py): Creates the time series dataset for the experiments.
    * [`VierlindenDataProcessor.py`](./data/VierlindenDataProcessor.py): Processes the specific "Vierlinden" dataset.
    * [`eda/`](./data/eda/): Holds notebooks and reports from the exploratory data analysis phase.
* [`models/`](./models/): Includes modules for building and loading the forecasting models.
    * [`build_model.py`](./models/build_model.py): Constructs the different time-series models (e.g., DeepAR, LSTM, TCN).
    * [`load_model.py`](./models/load_model.py): Loads pre-trained models for evaluation or inference.
* [`utils/`](./utils/): A collection of helper scripts for the core logic of the experiments.
    * [`ErrorGeneration.py`](./utils/ErrorGeneration.py): Generates different types of errors (e.g., outliers, missing values) to test model resilience.
    * [`ExperimentRunner.py`](./utils/ExperimentRunner.py): Manages the execution of the entire experimental workflow.
    * [`HyperparameterOptimizer.py`](./utils/HyperparameterOptimizer.py): Handles the hyperparameter optimization (HPO) process.
    * [`ModelTrainer.py`](./utils/ModelTrainer.py): Contains the logic for training the models.
    * [`ModelEvaluator.py`](./utils/ModelEvaluator.py): Evaluates model performance using various metrics.
* [`args_files/`](./args_files/): Stores configuration and argument files for different experimental setups.
    * [`best_hp/`](./args_files/best_hp/): Contains the best hyperparameter configurations found for each model, separated by `global` and `local` scenarios.
    * [`hpo_args_files/`](./args_files/hpo_args_files/): Arguments for running hyperparameter optimization sweeps.
    * [`errorgen_exp/`](./args_files/errorgen_exp/): Scripts and arguments for running the error generation experiments.
* [`hpo_configs/`](./hpo_configs/): YAML configuration files for the hyperparameter optimization sweeps for each model.
* [`archives/`](./archives/): Contains notebooks and detailed analyses from various experimental stages.
    * [`errorgen_analysis/`](./archives/errorgen_analysis/): In-depth analysis of the error generation experiments, including plots and explanations of different error types.
    * [`wandb_visualizations/`](./archives/wandb_visualizations/): Notebooks and results related to visualizing experiment data.
 
## Related Publications

This work builds on a series of related publications, including our foundational research on methodologies and evaluation frameworks as well as our continued work that informs the current implementation.

1. [Data-driven Modeling of Combined Sewer Systems for Urban Sustainability: An Empirical Evaluation](https://ceur-ws.org/Vol-3958/piai24-short1.pdf)<br>
   **Authors:** Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann<br>
   **Published in:** 47th German Conference on AI (2nd Workshop on Public Interest AI) 2025

2. Automated Energy-Aware Time-Series Model Deployment on Embedded FPGAs for Resilient Combined Sewer Overflow Management<br>
   **Authors:** Tianheng Ling, Vipin Singh, Chao Qian, Felix Biessmann, Gregor Schiele<br>
   **Accepted at:** 11th IEEE International Smart Cities Conference 2025

## Acknowledgement

This work is supported by the German Federal Ministry for Economic Affairs and Climate Action under the RIWWER project (01MD22007C, 01MD22007H).

## Contact

For questions or feedback, please feel free to open an issue or contact us at [vipin.singh@bht-berlin.de](mailto:vipin.singh@bht-berlin.de)
