import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
import shap
from matplotlib_venn import venn3
from hate_crime_model import LSTMModel
from preprocessing_utils import (
    preprocess_data, 
    prepare_data_for_cross_validation, 
    RedditHateCrimeDataset,
    preprocess_data_og
)
# Import your models
from model_comparison import (
    LSTMModel, GRUModel, BiLSTMModel,
    TCNModel,  TransformerModel, 
    CNNLSTMModel, LSTMAttentionModel
)

# Add after imports
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# === 1. Feature Selection Techniques ===

def correlation_based_selection(X, y, feature_names, threshold=0.05):
    """
    Select features based on correlation with target variable.
    Returns feature mask and selected feature names.
    """
    correlations = []
    for i in range(X.shape[1]):
        correlation = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        correlations.append(correlation)
    
    # Create a mask for features above threshold
    mask = np.array(correlations) > threshold
    
    selected_features = np.array(feature_names)[mask].tolist()
    
    print(f"Selected {sum(mask)} features based on correlation threshold {threshold}")
    print(f"Selected features: {selected_features}")
    
    # Plot correlations
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(correlations)
    plt.barh(np.array(feature_names)[sorted_idx], np.array(correlations)[sorted_idx])
    plt.xlabel('Absolute Correlation with Target')
    plt.title('Feature Correlation with Hate Crime Incidence')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    
    return mask, selected_features

def mutual_information_selection(X, y, feature_names, threshold=0.01):
    """
    Select features based on mutual information with target variable.
    Returns feature mask and selected feature names.
    """
    mi_scores = mutual_info_regression(X, y)
    
    # Create a mask for features above threshold
    mask = mi_scores > threshold
    
    selected_features = np.array(feature_names)[mask].tolist()
    
    print(f"Selected {sum(mask)} features based on mutual information threshold {threshold}")
    print(f"Selected features: {selected_features}")
    
    # Plot mutual information
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(mi_scores)
    plt.barh(np.array(feature_names)[sorted_idx], mi_scores[sorted_idx])
    plt.xlabel('Mutual Information with Target')
    plt.title('Feature Mutual Information with Hate Crime Incidence')
    plt.tight_layout()
    plt.savefig('feature_mutual_info.png')
    
    return mask, selected_features

def recursive_feature_elimination(X, y, feature_names, step=1, cv=5):
    """
    Use recursive feature elimination with cross-validation to select features.
    Returns feature mask and selected feature names.
    """
    # Create a random forest estimator for feature importance
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create RFECV object
    selector = RFECV(estimator, step=step, cv=cv, scoring='neg_mean_squared_error', verbose=1)
    
    # Fit the selector
    selector.fit(X, y)
    
    # Create mask of selected features
    mask = selector.support_
    
    selected_features = np.array(feature_names)[mask].tolist()
    
    print(f"Selected {sum(mask)} features using recursive feature elimination")
    print(f"Selected features: {selected_features}")
    
    # Plot number of features vs CV score (updated for newer sklearn versions)
    # plt.figure(figsize=(10, 6))

    # Plot number of features vs CV score
    plt.figure(figsize=(10, 6))
    n_features_list = range(1, len(selector.cv_results_['mean_test_score']) + 1)
    mean_scores = -selector.cv_results_['mean_test_score']  # Convert back to positive MSE
    std_scores = selector.cv_results_['std_test_score']

    plt.errorbar(n_features_list, mean_scores, yerr=std_scores, 
             marker='o', capsize=5, capthick=2)
    plt.axvline(x=selector.n_features_, color='red', linestyle='--', 
            label=f'Optimal: {selector.n_features_} features')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Recursive Feature Elimination with Cross-Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rfecv_scores.png', dpi=300, bbox_inches='tight')
    
    # # For sklearn >= 0.20
    # if hasattr(selector, 'cv_results_'):
    #     plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
    #             -selector.cv_results_['mean_test_score'])
    # # For older sklearn versions
    # elif hasattr(selector, 'grid_scores_'):
    #     plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    # else:
    #     print("Could not find CV scores to plot")
    
    # plt.xlabel('Number of Features')
    # plt.ylabel('Mean Squared Error')
    # plt.title('Recursive Feature Elimination with Cross-Validation')
    # plt.tight_layout()
    # plt.savefig('rfecv_scores.png')
    
    return mask, selected_features

def shap_feature_importance(model, X_test_tensor, feature_names, device):
    """
    Use SHAP values to explain the model and determine feature importance.
    """
    model.eval()
    # Create a background dataset (a sample of the training data)
    background = X_test_tensor[:100].to(device)
    
    # Create an explainer
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_tensor[:100].to(device))
    
    # Convert to numpy for visualization
    shap_values = [np.array(sv) for sv in shap_values]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[0], X_test_tensor[:100].cpu().numpy(), 
                     feature_names=feature_names, plot_type="bar")
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png')
    
    # Return mean absolute SHAP values as importance measure
    importance = np.mean(np.abs(shap_values[0]), axis=0)
    return importance

def plot_feature_selection_comparison(corr_mask, mi_mask, rfe_mask, feature_names):
    """
    Create a comprehensive comparison of feature selection methods.
    """
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Correlation': corr_mask.astype(int),
        'Mutual_Info': mi_mask.astype(int),
        'RFE': rfe_mask.astype(int)
    })
    comparison_df['Total_Votes'] = comparison_df[['Correlation', 'Mutual_Info', 'RFE']].sum(axis=1)
    comparison_df = comparison_df.sort_values('Total_Votes', ascending=True)
    
    # Create heatmap
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.3)))
    
    # Prepare data for heatmap
    heatmap_data = comparison_df[['Correlation', 'Mutual_Info', 'RFE']].values
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=['Correlation', 'Mutual Information', 'RFE'],
                yticklabels=comparison_df['Feature'],
                cmap='RdYlGn',
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Selected (1) / Not Selected (0)'})
    
    plt.title('Feature Selection Method Comparison')
    plt.xlabel('Selection Method')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_selection_heatmap.png', dpi=300, bbox_inches='tight')
    
    return comparison_df

# === 2. Hyperparameter Tuning with Optuna ===

def objective(trial, model_class, X, y, feature_names, input_size, output_size, 
             seq_length=8, n_splits=5, epochs=100):
    """
    Objective function for Optuna optimization.
    """
    # Define hyperparameter search space
    # Common hyperparameters for all models
    #OG
    batch_size = trial.suggest_int('batch_size', 10, 30, log=True)
    # batch_size = trial.suggest_int('batch_size', 20, 21)



    # OG
    # learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)

    # learning_rate = trial.suggest_float('learning_rate', 0.0009, 0.0011)  # Previously 1e-5 to 1e-2

    # transofrmer
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    # OG
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    # dropout = trial.suggest_float('dropout', 0.43, 0.45)
    
    # Different hyperparameters based on model type
    if model_class.__name__ in ['LSTMModel', 'GRUModel', 'BiLSTMModel', 'CNNLSTMModel']:
        hidden_size = trial.suggest_int('hidden_size', 16, 512, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        model_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
    elif model_class.__name__ == 'TCNModel':
        channels = trial.suggest_int('channels', 32, 128, log=True)
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        model_params = {
            'num_channels': [channels] * trial.suggest_int('num_levels', 1, 4),
            'kernel_size': kernel_size,
            'dropout': dropout
        }
    elif model_class.__name__ == 'TransformerModel':
        # Suggest nhead first (since hidden_size needs to be divisible by it)
        # nhead = trial.suggest_int('nhead', 2, 8)  # Minimum 2, max 8 heads
        # In your objective function, change:
        nhead = trial.suggest_int('nhead', 2, 8, step=2)  # Only even numbers: 2,4,6,8
        
        # Then suggest hidden_size as a multiple of nhead
        min_hidden = max(32, nhead * 4)  # Ensure minimum hidden_size is at least nhead*4
        max_hidden = 256
        # Round up to nearest multiple of nhead
        hidden_size = trial.suggest_int('hidden_size', min_hidden, max_hidden, step=nhead)
        
        model_params = {
            'hidden_size': hidden_size,
            'nhead': nhead,
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': dropout
        }

     # ADD THIS SECTION FOR ATTENTION LSTM
    elif model_class.__name__ == 'LSTMAttentionModel':
        hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)

        

        # NEW WOW!(12.05 rmse wow!)
        # 'dropout': 0.4484252256167552,
        # 'hidden_size': 45,
        # 'num_layers': 1,
        # 'attention_size': 84
        # hidden_size = trial.suggest_int('hidden_size', 44, 46)  # Previously 32-256

        num_layers = trial.suggest_int('num_layers', 1, 3)  # Usually fewer layers with attention'

        attention_size = trial.suggest_int('attention_size', 30, 150, log=True)
        attention_dropout = trial.suggest_float('attention_dropout', 0.0, 0.5)


        model_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'attention_size': attention_size,  # Size of attention layer
            'attention_dropout': attention_dropout
        }
    
    # Perform time series cross-validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {
        'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'mape': []
    }
    
    for train_idx, test_idx in tscv.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale data
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
            
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test_scaled)
        
        # Create datasets
        train_dataset = RedditHateCrimeDataset(X_train_tensor, y_train_tensor, seq_length)
        test_dataset = RedditHateCrimeDataset(X_test_tensor, y_test_tensor, seq_length)
        
        # Initialize model
        if model_class.__name__ == 'TCNModel':
            model = model_class(input_size=input_size, output_size=output_size, **model_params)
        elif model_class.__name__ == 'TransformerModel':
            model = model_class(input_size=input_size, output_size=output_size, **model_params)
        else:  # For LSTM, GRU, BiLSTM, CNN-LSTM
            model = model_class(input_size=input_size, 
                              output_size=output_size,
                              **model_params)
        
        model = model.to(device)
        
        # Train model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                # transformer only use

                # warmup_steps = 1000
                # scheduler = optim.lr_scheduler.LambdaLR(
                #     optimizer,
                #     lambda step: min(1.0, step / warmup_steps)
                # )

                optimizer.step()
                # scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(test_loader)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Evaluate best model
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                
                pred = outputs.cpu().numpy()
                true = y_batch.numpy()
                
                pred = y_scaler.inverse_transform(pred)
                true = y_scaler.inverse_transform(true)
                
                predictions.extend(pred)
                actuals.extend(true)
        
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(np.abs(actuals), 1e-10))) * 100
        
        cv_scores['mse'].append(mse)
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        cv_scores['mape'].append(mape)
    
    # Return the average MSE as the objective to minimize
    return np.mean(cv_scores['mse'])

def tune_hyperparameters(best_model_class, X, y, feature_names, input_size, output_size, 
                         n_trials=100, seq_length=8, n_splits=5):
    """
    Perform hyperparameter tuning using Optuna for the best model.
    """
    model_name = best_model_class.__name__
    print(f"Tuning hyperparameters for {model_name}")
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize', 
                               study_name=f"{model_name}_optimization")
    study.optimize(lambda trial: objective(trial, best_model_class, X, y, feature_names, 
                                         input_size, output_size, seq_length, n_splits),
                   n_trials=n_trials, timeout=3600)  # 1 hour timeout
    
    # Print results
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(f"{model_name} Hyperparameter Optimization History")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_opt_history.png")
    
    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title(f"{model_name} Hyperparameter Importance")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_param_importance.png")
    
    return study.best_params

def combined_feature_selection(X, y, feature_names, threshold_corr=0.05, threshold_mi=0.01):
    """
    Combine multiple feature selection methods to get a consensus on important features.
    """
    print("Running combined feature selection...")
    
    # Method 1: Correlation-based selection
    corr_mask, corr_features = correlation_based_selection(X, y, feature_names, threshold_corr)
    
    # Method 2: Mutual information
    mi_mask, mi_features = mutual_information_selection(X, y, feature_names, threshold_mi)
    
    # Method 3: Recursive feature elimination (more computationally intensive)
    # Only use if number of features is manageable
    if X.shape[1] < 50:  # Only run RFE if we have a reasonable number of features
        rfe_mask, rfe_features = recursive_feature_elimination(X, y, feature_names)
    else:
        rfe_mask = np.ones(X.shape[1], dtype=bool)  # Use all features if too many
        rfe_features = feature_names
    
    # Combine masks - select features identified by at least two methods
    combined_mask = ((corr_mask.astype(int) + mi_mask.astype(int) + rfe_mask.astype(int)) >= 2)
    selected_features = np.array(feature_names)[combined_mask].tolist()
    
    print(f"\nFeatures selected by correlation: {sum(corr_mask)}")
    print(f"Features selected by mutual information: {sum(mi_mask)}")
    print(f"Features selected by RFE: {sum(rfe_mask)}")
    print(f"Features selected by consensus (at least 2 methods): {sum(combined_mask)}")
    print(f"Final selected features: {selected_features}")
    
    # Create a Venn diagram to visualize overlap
    plt.figure(figsize=(10, 7))
    from matplotlib_venn import venn3
    venn3([set(np.array(feature_names)[corr_mask]),
          set(np.array(feature_names)[mi_mask]),
          set(np.array(feature_names)[rfe_mask])],
         ('Correlation', 'Mutual Information', 'RFE'))
    plt.title('Feature Selection Method Overlap')
    plt.savefig('feature_selection_venn.png')

    comparison_df = plot_feature_selection_comparison(corr_mask, mi_mask, rfe_mask, feature_names)
    
    return combined_mask, selected_features

def train_final_model(model_class, X, y, feature_names, selected_features, best_params, 
                      seq_length=8, batch_size=16, epochs=200, patience=25):
    """
    Train the final model with the best hyperparameters and selected features.
    """
    print(f"Training final {model_class.__name__} model with selected features and best hyperparameters")
    
    # Filter out parameters that aren't part of the model's __init__
    model_params = {
        'input_size': len(selected_features),
        'output_size': 1
    }
    
    # Add model-specific parameters from best_params
    if model_class.__name__ in ['LSTMModel', 'GRUModel', 'BiLSTMModel', 'CNNLSTMModel']:
        valid_params = ['hidden_size', 'num_layers', 'dropout']
    elif model_class.__name__ == 'TCNModel':
        valid_params = ['num_channels', 'kernel_size', 'dropout']
    elif model_class.__name__ == 'TransformerModel':
        valid_params = ['hidden_size', 'nhead', 'num_layers', 'dropout']
    elif model_class.__name__ == 'LSTMAttentionModel':
        valid_params = ['hidden_size', 'num_layers',  'dropout', 'attention_size']
    
    for param in valid_params:
        if param in best_params:
            model_params[param] = best_params[param]
    
    # Get indices of selected features
    feature_indices = [feature_names.index(feat) for feat in selected_features]
    print("Feature indices: ", feature_indices)
    # Select only those features
    print("X: ", X.shape)

    X_selected = X[:, feature_indices]
    print("y: ", y)
    # X_selected = X_selected[X_selected[:, -3].argsort()]    
    # y = y[X_selected[:, -3].argsort()]
    # Split data into train-test (80-20)
    train_size = int(0.8 * len(X_selected))
    test_size = len(X_selected) - train_size
   
    X_train, X_test = X_selected[:train_size], X_selected[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
        
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    print("X_train_scaled: ", X_train_scaled.shape)
    print("X_test_scaled: ", X_test_scaled.shape)
    
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Create datasets
    train_dataset = RedditHateCrimeDataset(X_train_tensor, y_train_tensor, seq_length)
    test_dataset = RedditHateCrimeDataset(X_test_tensor, y_test_tensor, seq_length)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with filtered parameters
    model = model_class(**model_params)
    model = model.to(device)
    
    # Train model
    criterion = nn.MSELoss()
    print("Best learning rate: ", best_params.get('learning_rate', 0.001))
    optimizer = optim.Adam(model.parameters(), lr=best_params.get('learning_rate', 0.001))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    # Train for a fixed number of epochs (could add early stopping)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            # OG
            # loss = criterion(outputs, y_batch)
            loss = torch.sqrt(criterion(outputs, y_batch))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            # transformer only use

            # warmup_steps = 1000
            # scheduler = optim.lr_scheduler.LambdaLR(
            #     optimizer,
            #     lambda step: min(1.0, step / warmup_steps)
            # )

            optimizer.step()
            # scheduler.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                # OG
                # loss = criterion(outputs, y_batch)
                loss = torch.sqrt(criterion(outputs, y_batch))                
                val_loss += loss.item()
        
        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # # Plot training curve
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'{model_class.__name__} Training Curve')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'{model_class.__name__.lower()}_training_curve.png')

    
    # Evaluate final model
    model.eval()
    all_predictions = []
    all_actuals = []
    
    # with torch.no_grad():
    #     for loader in [train_loader , test_loader]:
    #         for X_batch, y_batch in loader:
    #         #for X_batch, y_batch in test_loader:
    #             X_batch = X_batch.to(device)
    #             outputs = model(X_batch)
                
    #             pred = outputs.cpu().numpy()
    #             true = y_batch.numpy()
                
    #             pred_rescaled = y_scaler.inverse_transform(pred)
    #             true_rescaled = y_scaler.inverse_transform(true)
                
    #             all_predictions.extend(pred_rescaled)
    #             all_actuals.extend(true_rescaled)

    # Evaluate on the test set
    with torch.no_grad():
        # For each point where we want a prediction
        for i in range(len(X_selected) - seq_length + 1):
            # Get sequence
            seq = X_selected[i:i+seq_length]
            seq_tensor = torch.FloatTensor(X_scaler.transform(seq)).unsqueeze(0).to(device)
            
            # Make prediction
            outputs = model(seq_tensor)
            
            # Get the actual value corresponding to this prediction
            # (the target value at the end of the sequence)
            true_value = y[i+seq_length-1]
            
            # Convert prediction to numpy and rescale
            pred = outputs.cpu().numpy()
            pred_rescaled = y_scaler.inverse_transform(pred)
            
            # Store the values
            all_predictions.extend(pred_rescaled)
            all_actuals.append(true_value)  # No need to rescale, it's already in original scale

    
    all_predictions = np.array(all_predictions).flatten()
    all_actuals = np.array(all_actuals).flatten()
    
    # Calculate metrics
    # train_size = 0
    mse = mean_squared_error(all_actuals[train_size:], all_predictions[train_size:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_actuals[train_size:], all_predictions[train_size:])
    r2 = r2_score(all_actuals[train_size:], all_predictions[train_size:])

    # Calculate MAPE - Mean Absolute Percentage Error
    # Avoid division by zero by adding a small epsilon where actual values are zero
    epsilon = 1e-10
    abs_percentage_errors = np.abs((all_actuals[train_size:] - all_predictions[train_size:]) / 
                                 np.maximum(np.abs(all_actuals[train_size:]), epsilon)) * 100
    mape = np.mean(abs_percentage_errors)
    
    print("\nFinal Model Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    
    # Get indices for time axis (assuming sequential data)
    time_idx = np.arange(seq_length, len(all_actuals)+seq_length)
    
    plt.plot(time_idx, all_actuals, 'b-', label='Actual')
    plt.plot(time_idx, all_predictions, 'r--', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Hate Crime Incidents')
    plt.title(f'{model_class.__name__} Model Predictions vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_class.__name__.lower()}_predictions.png')
    
    # Add residual plot
    residuals = all_actuals - all_predictions
    
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.scatter(all_predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=20)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{model_class.__name__.lower()}_residuals.png')

    # Add prediction confidence intervals
    if len(all_predictions) > 50:  # Only if we have enough predictions
        # Calculate rolling statistics
        window_size = min(10, len(all_predictions) // 5)
        rolling_std = pd.Series(all_predictions).rolling(window=window_size).std()
    
        plt.figure(figsize=(14, 8))
        plt.plot(time_idx, all_actuals, 'b-', label='Actual', linewidth=2)
        plt.plot(time_idx, all_predictions, 'r--', label='Predicted', linewidth=2)
    
        # Add confidence bands
        upper_bound = all_predictions + 1.96 * rolling_std.fillna(rolling_std.mean())
        lower_bound = all_predictions - 1.96 * rolling_std.fillna(rolling_std.mean())
    
        plt.fill_between(time_idx, lower_bound, upper_bound, 
                     alpha=0.2, color='red', label='95% Confidence Interval')
    
        plt.xlabel('Time')
        plt.ylabel('Hate Crime Incidents')
        plt.title(f'{model_class.__name__} Predictions with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_class.__name__.lower()}_predictions_with_ci.png', dpi=300, bbox_inches='tight')

        # Add metric comparison radar chart
        metrics_names = ['RMSE', 'MAE', 'R²', 'MAPE']
        metrics_values = [rmse, mae, r2, mape]

        # Normalize values for radar chart (you may need to adjust the normalization)
        normalized_values = []
        for i, (name, value) in enumerate(zip(metrics_names, metrics_values)):
            if name == 'R²':
                normalized_values.append(value)  # R² is already 0-1
            else:
                # Normalize other metrics (this is a simple approach, adjust as needed)
                normalized_values.append(1 / (1 + value))  # Inverse for error metrics

        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        normalized_values = normalized_values + [normalized_values[0]]  # Complete the circle

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, normalized_values, 'o-', linewidth=2)
        ax.fill(angles, normalized_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        plt.title(f'{model_class.__name__} Performance Metrics', size=16, y=1.1)
        plt.tight_layout()
        plt.savefig(f'{model_class.__name__.lower()}_radar_chart.png', dpi=300, bbox_inches='tight')
    
    # Try to analyze feature importance with SHAP
    try:
        importance = shap_feature_importance(model, X_test_tensor[:100], selected_features, device)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(np.array(selected_features), importance)
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{model_class.__name__.lower()}_feature_importance.png')
    except Exception as e:
        print(f"Could not calculate SHAP values: {str(e)}")
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'selected_features': selected_features,
        'hyperparameters': best_params
    }, f'final_{model_class.__name__.lower()}_model.pth')
    
    print(f"\nFinal model saved as 'final_{model_class.__name__.lower()}_model.pth'")
    
    return {
        'model': model,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'selected_features': selected_features,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    }

# === 3. Model Comparison ===

def compare_models(X, y, feature_names, seq_length=8, n_splits=5):
    """
    Compare different model architectures to find the best one.
    """
    print("Comparing different model architectures...")
    
    # Define models to compare
    models = [
        LSTMModel,
        GRUModel,
        BiLSTMModel,
        TCNModel,
        TransformerModel,
        CNNLSTMModel,
        LSTMAttentionModel
    ]
    
    # Parameters for each model
    model_params = {
        'LSTMModel': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        'GRUModel': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        'BiLSTMModel': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        'TCNModel': {'num_channels': [64, 64, 64], 'kernel_size': 3, 'dropout': 0.2},
        'TransformerModel': {'hidden_size': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2},
        'CNNLSTMModel': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        'LSTMAttentionModel': {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2, 'attention_size': 32}
    }
    
    # Use TimeSeriesSplit for evaluation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {}
    
    for model_class in models:
        model_name = model_class.__name__
        print(f"\nEvaluating {model_name}...")
        
        model_results = {
            'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'mape': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale data
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
                
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)
            
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
            y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled)
            
            # Create datasets
            train_dataset = RedditHateCrimeDataset(X_train_tensor, y_train_tensor, seq_length)
            test_dataset = RedditHateCrimeDataset(X_test_tensor, y_test_tensor, seq_length)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model
            if model_name == 'TCNModel':
                model = model_class(input_size=X.shape[1], output_size=1, **model_params[model_name])
            elif model_name == 'TransformerModel':
                model = model_class(input_size=X.shape[1], output_size=1, **model_params[model_name])
            else:  # For LSTM, GRU, BiLSTM, CNN-LSTM
                model = model_class(input_size=X.shape[1], 
                                  output_size=1,
                                  **model_params[model_name])
            
            model = model.to(device)
            
            # Train model
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Train for a fixed number of epochs (could add early stopping)
            for epoch in range(100):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # Evaluate
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    
                    pred = outputs.cpu().numpy()
                    true = y_batch.numpy()
                    
                    pred = y_scaler.inverse_transform(pred)
                    true = y_scaler.inverse_transform(true)
                    
                    predictions.extend(pred)
                    actuals.extend(true)
            
            predictions = np.array(predictions).flatten()
            actuals = np.array(actuals).flatten()
            
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / np.maximum(np.abs(actuals), 1e-10))) * 100
            
            model_results['mse'].append(mse)
            model_results['rmse'].append(rmse)
            model_results['mae'].append(mae)
            model_results['r2'].append(r2)
            model_results['mape'].append(mape)
        
        # Calculate average metrics across all folds
        avg_results = {
            'mse': np.mean(model_results['mse']),
            'rmse': np.mean(model_results['rmse']),
            'mae': np.mean(model_results['mae']),
            'r2': np.mean(model_results['r2']),
            'mape': np.mean(model_results['mape'])
        }
        
        results[model_name] = avg_results
        
        print(f"{model_name} Results:")
        print(f"  MSE: {avg_results['mse']:.4f}")
        print(f"  RMSE: {avg_results['rmse']:.4f}")
        print(f"  MAE: {avg_results['mae']:.4f}")
        print(f"  R²: {avg_results['r2']:.4f}")
        print(f"  MAPE: {avg_results['mape']:.2f}%")

    # Create and save results table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df = results_df.sort_values('rmse')  # Sort by RMSE (best first)

    # Save as CSV
    results_df.to_csv('model_comparison_results.csv')

    # Create a formatted table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Color code the table - best values in green, worst in red
    table_data = []
    for idx, (model, metrics) in enumerate(results_df.iterrows()):
        row = [model] + [f"{v:.4f}" for v in metrics.values]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data,
                colLabels=['Model'] + [col.upper() for col in results_df.columns],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code cells based on performance (excluding model names column)
    for i in range(1, len(results_df.columns) + 1):
        values = results_df.iloc[:, i-1].values
        if results_df.columns[i-1] == 'r2':  # Higher is better for R²
            best_idx = np.argmax(values)
            worst_idx = np.argmin(values)
        else:  # Lower is better for MSE, RMSE, MAE, MAPE
            best_idx = np.argmin(values)
            worst_idx = np.argmax(values)
    
        # Color the best and worst cells
        table[(best_idx + 1, i)].set_facecolor('#90EE90')  # Light green
        table[(worst_idx + 1, i)].set_facecolor('#FFB6C1')  # Light red

    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
    
    # Plot comparison results
    plt.figure(figsize=(14, 10))
    
    metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
    model_names = list(results.keys())
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        values = [results[model][metric] for model in model_names]
        
        # For R², higher is better - invert for consistent coloring
        if metric == 'r2':
            values = [-v for v in values]  # For coloring purposes only
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        bars = plt.bar(model_names, values, color=colors)
        
        plt.title(metric.upper())
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if metric == 'r2':
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{-height:.3f}' if metric == 'r2' else f'{height:.3f}',
                        ha='center', va='bottom')
            else:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')

    # Create and save results table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df = results_df.sort_values('rmse')  # Sort by RMSE (best first)

    # Save as CSV
    results_df.to_csv('model_comparison_results.csv')

    # Create a formatted table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Color code the table - best values in green, worst in red
    table_data = []
    for idx, (model, metrics) in enumerate(results_df.iterrows()):
        row = [model] + [f"{v:.4f}" for v in metrics.values]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data,
                colLabels=['Model'] + [col.upper() for col in results_df.columns],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code cells based on performance (excluding model names column)
    for i in range(1, len(results_df.columns) + 1):
        values = results_df.iloc[:, i-1].values
        if results_df.columns[i-1] == 'r2':  # Higher is better for R²
            best_idx = np.argmax(values)
            worst_idx = np.argmin(values)
        else:  # Lower is better for MSE, RMSE, MAE, MAPE
            best_idx = np.argmin(values)
            worst_idx = np.argmax(values)
        
        # Color the best and worst cells
        table[(best_idx + 1, i)].set_facecolor('#90EE90')  # Light green
        table[(worst_idx + 1, i)].set_facecolor('#FFB6C1')  # Light red

    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
    
    # Determine best model based on RMSE
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model_class = next((m for m in models if m.__name__ == best_model_name), None)
    
    print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]['rmse']:.4f}")
    
    return results, best_model_class

# === 4. Main Execution ===

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_path = '/Users/meghanreilly/Desktop/BAP-Comp-IR/blm_with_hate_crimes.csv'
    df = pd.read_csv(data_path)
    # df_complete = preprocess_data_og(df)
    df_complete = preprocess_data(df)

    # Temporal (optional)
    # consistennt subreddit order
    # Sort by 'week' first, then by 'subreddit'
    #df_complete = df_complete.sort_values(by=['week_num', 'subreddit']).reset_index(drop=True)    
    victim_mean = df_complete['weekly_victim_count'].mean()
    df_complete["month"] = df_complete['week_start_date'].dt.month  # Seasonality

    # unique_subreddits = df_complete['subreddit'].unique()
    df_complete['lag_4w'] = df_complete['weekly_victim_count'].shift(4)  # 4-week cycle
    df_complete['lag_4w'] = df_complete['lag_4w'].fillna(victim_mean)



    target_col = 'weekly_victim_count'
    
    # Identify feature columns (exclude non-feature columns)
    #exclude_cols = ['subreddit', 'week_start_date', 'date_range', 'week_num', 'analysis', 'week_start_date', target_col]
    exclude_cols = ["weekly_victim_count", "week_start_date"]
    feature_cols = [col for col in df_complete.columns if col not in exclude_cols]

    summary_stats = df_complete[feature_cols + [target_col]].describe()
    summary_stats.to_csv('dataset_summary_statistics.csv')

    print(df_complete[feature_cols].head(21))
    # Extract data
    X = df_complete[feature_cols].values
    y = df_complete[target_col].values
    # plt.plot(df_complete["week_num"], df_complete[target_col])
    # plt.show()
    feature_names = feature_cols
    
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Feature names: {feature_names}")

    # Perform feature selection
    print("\nPerforming feature selection...")
    # Adjust thresholds for your specific case
    feature_mask, selected_features = combined_feature_selection(
        X, y, feature_names, 
        threshold_corr=0.3,  # Lower correlation threshold
        threshold_mi=0.05    # Lower mutual info threshold
    )
    
    # Ensure we have at least some features
    if len(selected_features) < 3:
        print("Warning: Very few features selected. Using all features as fallback.")
        selected_features = feature_names
        feature_mask = np.ones(len(feature_names), dtype=bool)
    
    X_selected = X[:, feature_mask]
    
    # Compare models to find the best architecture
    print("\nComparing model architectures...")
    model_results, best_model_class = compare_models(X_selected, y, selected_features)

    
    #best_model_class = LSTMAttentionModel # Placeholder for the best model class

    # Hyperparameter tuning for the best model
    print(f"\nTuning hyperparameters for {best_model_class.__name__}...")
    # best_params = tune_hyperparameters(best_model_class, X_selected, y, selected_features,
    #                                 input_size=len(selected_features), output_size=1, n_trials=100, 
    #                                 seq_length=8
    #                                 )
    
    #Train final model with best parameters and selected features
    # selected_features = ['week_num', 
    #                      'AskThe_Donald_echo_chamber', 
    #                      'AskThe_Donald_sentiment', 
    #                      'AskThe_Donald_hostility', 
    #                      'AskThe_Donald_dehumanizing', 
    #                      'BlackLivesMatter_hostility', 
    #                      'Conservative_hostility', 
    #                      'Conservative_dehumanizing', 
    #                      'Republican_echo_chamber', 
    #                      'Republican_sentiment', 
    #                      'Republican_hostility', 
    #                      'Republican_dehumanizing', 
    #                      'conservatives_echo_chamber', 
    #                      'conservatives_sentiment', 
    #                      'conservatives_hostility', 
    #                      'conservatives_dehumanizing', 
    #                      'democrats_framing', 'democrats_echo_chamber', 'democrats_sentiment', 'democrats_hostility', 'politics_hostility', 'prev_weekly_victim_count', 'month', 'lag_4w']
   
    best_params = {
    'batch_size': 21,
    'learning_rate': 0.000516195693625742,
    'dropout': 0.4484252256167552,
    'hidden_size': 45,
    'num_layers': 1,
    'attention_size': 84}
    
    print("\nTraining final model...")
    final_model = train_final_model(best_model_class, X, y, feature_names, selected_features,
                                  best_params, batch_size=best_params["batch_size"], seq_length=8,
                                
                                  )
    
    print("\nModel training and evaluation complete!")

if __name__ == '__main__':
    main()