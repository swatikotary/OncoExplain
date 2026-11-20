\# OncoExplain

\### Stability Analysis of SHAP-Based Explanations in Ensemble Machine Learning for Breast Cancer Classification



OncoExplain is an explainable machine learning research project that investigates the stability of SHAP-based explanations in breast cancer classification.



The project goes beyond evaluating whether a model predicts correctly. It examines whether the explanations behind those predictions remain consistent when the same model is retrained using different training samples.



\## Research Problem



Machine learning models can achieve high classification performance while their explanations may vary across repeated training runs.



This raises an important question:



\*\*Does prediction consistency necessarily imply explanation consistency?\*\*



OncoExplain investigates this relationship using repeated model training, SHAP feature attribution analysis, and quantitative stability metrics.



\## Objectives



\- Develop machine learning models for benign and malignant breast cancer classification.

\- Compare the performance of multiple classification algorithms.

\- Analyse feature contributions using SHAP.

\- Evaluate prediction consistency across repeated model training.

\- Measure global and individual-level explanation stability.

\- Present the experimental findings through an interactive Streamlit dashboard.



\## Dataset



The project uses the \*\*Wisconsin Diagnostic Breast Cancer (WDBC)\*\* dataset.



\- 569 observations

\- 30 numerical diagnostic features

\- Two classes: Benign and Malignant



The features describe characteristics of cell nuclei obtained from digitised fine needle aspirate images.



\## Methodology



1\. Dataset loading, inspection and preprocessing.

2\. Stratified training and testing split.

3\. Model development and performance comparison.

4\. SHAP-based feature attribution analysis.

5\. Repeated training of the selected ensemble model using different stratified training samples.

6\. Evaluation of prediction and explanation stability on a fixed test set.

7\. Visualisation of results through a Streamlit dashboard.



\## Models



The initial model comparison includes:



\- Logistic Regression

\- Random Forest

\- Support Vector Machine (SVM)

\- Soft Voting Ensemble



The Soft Voting Ensemble is used in the repeated-training explanation stability experiment.



\## Key Experimental Findings



The repeated-training experiment uses 10 models and a fixed test set of 114 observations.



| Metric | Result |

|---|---:|

| Average prediction agreement | 99.386% |

| Records with perfect prediction agreement | 111 / 114 |

| Average global SHAP Spearman correlation | 0.9484 |

| Average global Top-10 Jaccard similarity | 1.0000 |

| Average SHAP direction consistency | 95.3333% |

| Average individual-record Spearman correlation | 0.9559 |

| Average individual-record Top-5 Jaccard similarity | 0.8173 |



These results indicate that high prediction agreement can coexist with variation in individual-level feature attribution rankings.



\## Interactive Dashboard



The project includes a Streamlit dashboard with six sections:



1\. \*\*Overview\*\* – Research summary and project highlights.

2\. \*\*Model Performance\*\* – Comparison of classification models.

3\. \*\*SHAP Explainability\*\* – Global and individual feature explanations.

4\. \*\*Explanation Stability\*\* – Analysis of SHAP attribution consistency.

5\. \*\*Prediction Stability\*\* – Comparison of predictions across repeated models.

6\. \*\*Live Prediction \& Explanation\*\* – Interactive prediction and explanation for an input record.



\## Technology Stack



\- Python

\- Scikit-learn

\- SHAP

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Streamlit



\## Project Structure



```text

OncoExplain/

│

├── app.py

├── expand\_shap\_experiment.py

├── inspect\_data.py

├── test\_live\_shap.py

├── .gitignore

│

├── assets/

│

├── data/

│   └── processed/

│

├── results/

│   ├── figures/

│   ├── shap\_baseline/

│   ├── shap\_values/

│   └── stability\_experiment/

│

└── utils/

```



\## Running the Project Locally



Clone the repository:



```bash

git clone https://github.com/swatikotary/OncoExplain.git

cd OncoExplain

```



Create and activate a virtual environment:



```bash

python -m venv .venv

```



Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



Run the dashboard:



```bash

streamlit run app.py

```



\*Note: The requirements file will be added separately.\*



\## Limitations and Responsible Use



\- The study uses the WDBC dataset and does not establish generalisation to all clinical populations.

\- SHAP explanations describe model behaviour; they do not establish causal relationships.

\- Explanation stability does not guarantee clinical validity or correctness.

\- OncoExplain is an academic research project and is not intended for medical diagnosis or clinical decision-making.



\## Future Work



\- Compare explanation stability across Logistic Regression and Soft Voting Ensemble models.

\- Investigate stability across additional datasets.

\- Explore alternative explainability methods.

\- Extend the evaluation to additional model configurations.



\## Author



\*\*Swati Jayaram Kotary\*\*  

Computer Science and Engineering  

Ramaiah Institute of Technology



\*\*GitHub:\*\* https://github.com/swatikotary



\---



\*Developed as an academic AI/ML research project.\*

