# HAR-WISDM-ML

ML models for Human Activity Recognition; based on handcrafted feature-set (Shallow Model) &amp; raw accelerometer signals (Deep Model).

The models were developed to classify unlabelled test-data samples, submitting to a Kaggle Competititon for scoring (based on F1 score).

Specifically, the notebooks/code document the whole journey for developing these models:

1. Exploration/Diagnostics
2. Preprocessing
3. Development
4. Experimentation
5. Documentation & Reporting

## Data
Directory containing the data for models:
```
├── data
│   ├── empty_rows.csv
│   ├── processed
│   │   ├── predictions_allzero.csv
│   │   ├── test_feature_clean.csv
│   │   ├── test_signal_clean.csv
│   │   ├── train_feature_clean.csv
│   │   └── train_signal_clean.csv
│   ├── raw
│   │   ├── test_feature.csv
│   │   ├── test_signal.csv
│   │   ├── train_feature.csv
│   │   └── train_signal.csv
```
* **Raw** data are the input files for train/test for either model.
* **Processed** is that which is output from ***preprocessing.ipynb***.
## Code

###  Jupyter Notebooks

In order of development:

1. ***preprocessing.ipynb***: Transforms the downloaded datafiles into new datafiles for later usage (**NOTE**: this code produces a file called *predictions_allzero.csv*, which seperately classifies problematic-samples that are in the test-data - this is concatenated at the submission stage (see `save_submission_file()` in ***custom_functions.py***))
2. ***diagnostic_plots.ipynb***: Extra plots for diagnostics/exploration
3. ***shallow_model.ipynb***: Space for developing & testing aspects of the *Random Forest* model on the *train_features.csv* data.
4. ***mlp.ipynb***: Comparison model for *train_features.csv*, using an MLP Classifier - this is not mentioned in the report.
5. ***deep_model_v1.ipynb***: Space for developing the CNN/LSTM model using the *train_signal.csv* data.
6. ***deep_model_v2.ipynb***: Space for refining `deep_model_v1.ipynb`, and development of further adversarial training method -- **This notebook was developed within Colab.**

### Python scripts

***custom_functions.py*** is a functional way to hold functions that are intended for use across notebooks. 

Mostly, these are functions for **Reading/Writing Model configurations** to the config file, and **Creating Submission files**.

## Other files
### Project requirements (*requirements.txt*)

Since the project was developed using a Conda environment, *requirements.txt* contains many unnecessary elements, but is included for the sake of completeness. To recreate, run:

<pre><code> conda create --name dat81_project_env --file requirements.txt</code></pre>

### The Report (*report.pdf*)
Developing the models was one part of the whole project; the other part involved producing a *Written report* of the relevant models/development process. This was produced using LaTeX and included as a pdf file herein.