# Other Python library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import time
from imblearn.pipeline import make_pipeline as make_imb_pipeline

CONFIG_FILE = 'shallow_model_configs.jsonl'
MY_SEED = 6
STEP_CLASS_MAP = {
    "standardscaler": StandardScaler,
    "pca": PCA,
    "randomforestclassifier": RandomForestClassifier,
    "randomundersampler": RandomUnderSampler,
    'smote': SMOTE,
    'mlpclassifier': MLPClassifier   
}

def extract_pipeline_config(pipeline: Pipeline):
    steps = dict(pipeline.named_steps)
    model_config = {
    }
    for k in steps.keys():
        model_config[k] = steps[k].get_params()
    return model_config

def save_model_config(model_config: dict | Pipeline, model_name, score = None, notes = None, fit_submit_time = None, excluded_features = None, filename = CONFIG_FILE, override = False):
    if isinstance(model_config, Pipeline):
        model_config = extract_pipeline_config(pipeline=model_config)
    config = {
        'model_name': model_name,
        'steps': model_config,
        'score': score,
        'notes': notes,
        'fit_submit_time': fit_submit_time,
        'excluded_features': excluded_features
    }
    
    if override:
        with open(filename, 'r') as f:
            configs = [json.loads(line) for line in f]
        found = False
        for i, cfg in enumerate(configs):
            if cfg['model_name'] == model_name:
                configs[i] = config
                found = True
                break
        if not found:
            configs.append(config)
        with open(filename, 'w') as f:
            for cfg in configs:
                f.write(json.dumps(cfg) + '\n')
        pass
    else:
        if model_name in get_model_names(filename):
            raise ValueError(f'"{model_name}" already exists. Use a new name for the model configuration.')
        else:
            with open(filename, "a") as f:
                f.write(json.dumps(config) + "\n")
                
def get_model_names(filename = CONFIG_FILE):
    with open(filename, 'r') as f:
        configs = [json.loads(line) for line in f]
    config_lookup = {config['model_name']: config for config in configs}
    return list(config_lookup.keys())

def get_pipeline(model_config: dict) -> Pipeline:
    pipeline_steps = []
    step_config = model_config['steps']
    for step_name, params in step_config.items():
        cls = STEP_CLASS_MAP.get(step_name.lower())
        if cls is None:
            raise ValueError(f"'{step_name}' is not a recognized Step. Alter the STEP_CLASS_MAP?")
        step_instance = cls(**params)
        pipeline_steps.append((step_name, step_instance))

    return Pipeline(pipeline_steps)

def best_model(config_file = CONFIG_FILE):
    with open(config_file, 'r') as f:
        configs = [json.loads(line) for line in f]
    scores = [cfg['score'] for cfg in configs]
    for cfg in configs:
        if cfg['score'] == max(scores):
            return cfg['model_name']

def load_model_config(model_name, filename = CONFIG_FILE, into_pipeline = False):
    with open(filename, 'r') as f:
        configs = [json.loads(line) for line in f]
    f.close()
    config_lookup = {config['model_name']: config for config in configs}
    cfg = config_lookup[model_name]
    if not into_pipeline:
        # if cfg['excluded_features'] is not None:
            # cfg['excluded_features'] = ast.literal_eval(cfg['excluded_features'])
        return cfg
    else:
        return get_pipeline(config_lookup[model_name])

def save_submission_file(train_data_file, test_data_file, model_name, config_file = CONFIG_FILE, as_filename: str = None, class_column = 'activity', exception_file = '../data/processed/predictions_allzero.csv'):
    t_start = time.time()
    
    df_train = pd.read_csv(train_data_file)
    df_test = pd.read_csv(test_data_file)
    
    if as_filename is None:
        submission_filename = f'feature_{model_name}.csv'
    else:
        submission_filename = as_filename
    
    model_config = load_model_config(model_name, CONFIG_FILE, into_pipeline=False)
    excluded_features = model_config['excluded_features']
    features = list(df_train.columns[3:])
    
    if excluded_features is not None:
        for f in excluded_features:
            features.remove(f)
    
    X_train = df_train[features].values
    y_train = df_train[class_column]
    
    X_test = df_test[features].values
    
    pipeline = get_pipeline(model_config)
    
    try:
        pipeline.fit(X_train, y_train)
    except TypeError:
        steps = model_config['steps']
        ppl_steps = []
        for k, v in steps.items():
            s = STEP_CLASS_MAP[k]
            ppl_steps.append(s(**v))
        pipeline = make_imb_pipeline(*ppl_steps)
        pipeline.fit(X_train, y_train)
    pd.concat(
        [pd.DataFrame(data = {
            'id': df_test['id'],
            'predicted': pipeline.predict(X_test)}),
        pd.read_csv(exception_file)]
        ).to_csv('predictions/Feature/' + submission_filename, index=False)

    t_end = time.time()
    model_config['fit_submit_time'] = t_end - t_start
    
    save_model_config(
        model_config=model_config['steps'],
        model_name = model_name,
        score=model_config['score'],
        notes=model_config['notes'],
        filename = config_file,
        override=True,
        excluded_features=excluded_features,
        fit_submit_time=t_end - t_start
    ) 
def get_training_splits(model_config, train_data_file, test_data_file, class_column = 'activity'):
    
    df_train = pd.read_csv(train_data_file)
    df_test = pd.read_csv(test_data_file)
    
    excluded_features = model_config['excluded_features']
    features = list(df_train.columns[3:])
    
    if excluded_features is not None:
        for f in excluded_features:
            features.remove(f)
    
    X_train = df_train[features].values
    y_train = df_train[class_column]
    
    X_test = df_test[features].values

    return X_train, y_train, X_test, features
