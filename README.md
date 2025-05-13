# TimeTrace Insight

Source code and supporting materials for the research paper: **"TimeTrace Insight: An Intelligent Visual Analytics Framework for Longitudinal Clinical Data and Medical Knowledge Integration."**

## Description

TimeTrace Insight is a novel Visual Analytics (VA) framework designed for the integrated exploration and analysis of longitudinal clinical data enriched with medical knowledge. Its layered architecture synergistically combines:

1.  **Intelligent Temporal Data Processing**: Including alignment and Temporal Feature Abstraction (TFA) for automatically deriving high-level states, trends, shapes, and sequential patterns.
2.  **Seamless Data-Knowledge Linking**: Using rule-based inference to connect patient data with medical knowledge context (e.g., ontologies, guidelines), generating actionable annotations.
3.  **Coordinated Multi-Perspective Visualizations**: Featuring integrated knowledge overlays and adaptive layouts to support interactive exploration.

This repository contains the Python implementation used for the case study analyzing longitudinal perioperative Hemoglobin (HB) and Glucose (GLUC) data, as presented in the paper.

## Features Demonstrated in Code

*   **Temporal-Centric Design**: Data alignment and preparation.
*   **Data Cleaning**: Outlier detection (IQR) and imputation (ffill).
*   **Feature Derivation**: Calculation of BMI and delta values.
*   **Temporal Feature Abstraction (TFA)**:
    *   State Identification based on thresholds.
    *   Trend Analysis from delta values.
    *   Shape Matching for predefined temporal patterns.
    *   Sequential Pattern Mining (GSP-like for L2 and L3 patterns).
*   **Intelligent Data-Knowledge Linking**: Application of rules from a mock knowledge base.
*   **Multi-Perspective Visualization**: Generation of multi-panel plots for individual patient trajectories (Enhanced Timeline, Indicator Trend Views with knowledge overlays and abstracted features) and cohort-level pattern summaries.
*   **Adaptive Visualization Layout**: Simulation of adaptive rendering principles.

## Repository Structure

```
.
├── clinical_data.csv         # Example clinical input data
├── lab_data.csv              # Example lab input data
├── track_names.csv           # Metadata for tracks (if applicable)
├── clinical_parameters.csv   # Metadata for clinical parameters (if applicable)
├── lab_parameters.csv        # Metadata for lab parameters (if applicable)
├── timetrace_insight_main.py # Main Python script demonstrating the framework (or your script name)
├── all_cases_visualizations_pdf/ # Directory for output visualizations
│   ├── summary_case_4.pdf
│   ├── summary_case_5.pdf
│   ├── summary_case_10.pdf
│   ├── frequent_event_pairs_l2.pdf
│   └── frequent_event_triplets_l3.pdf
├── README.md                 # This README file
└── requirements.txt          # Python dependencies (recommended)
```

*(Adjust the file and directory names above to match your actual project structure.)*

## Setup and Usage

### Prerequisites

*   Python 3.x
*   (List other major prerequisites if any, e.g., specific database, OS)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/timetrace-insight.git
    cd timetrace-insight
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    If you have a `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    Otherwise, list key libraries to install manually, e.g.:
    ```bash
    pip install pandas numpy matplotlib
    ```

### Running the Code

1.  **Place your input data CSV files** (`clinical_data.csv`, `lab_data.csv`, etc.) in the root directory of the project, or update the `data_dir` variable in the script to point to their location.
    *(Note: The provided sample CSV files are for demonstration purposes as described in the paper's case study.)*
2.  **Execute the main script:**
    ```bash
    python timetrace_insight_main.py
    ```
3.  **Output:**
    *   The script will print processing logs to the console.
    *   Generated visualizations (PDF files for individual cases and frequent patterns) will be saved in the `all_cases_visualizations_pdf/` directory.

## Case Study Data

The example clinical_data.csv and lab_data.csv files used in this repository are preprocessed subsets derived from the VitalDB database, a high-fidelity multi-parameter vital signs database collected from surgical patients. These data have been curated to represent typical perioperative clinical scenarios, with a particular focus on longitudinal measurements of Hemoglobin (HB) and Glucose (GLUC).

The original dataset is publicly available and cited as:

@misc{BibEntry2025May,
  title = {{VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients v1.0.0}},
  year = {2025},
  month = may,
  note = {[Online; accessed 13. May 2025]},
  url = {https://physionet.org/content/vitaldb/1.0.0}
}

All data used in this case study have been anonymized and utilized under the data usage guidelines provided by the source.


