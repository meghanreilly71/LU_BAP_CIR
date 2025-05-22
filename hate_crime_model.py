import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class RedditHateCrimeDataset(Dataset):
    def __init__(self, X, y, seq_length=8):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]
        y_target = self.y[idx+self.seq_length]
        return X_seq, y_target

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        context = output[:, -1, :]
        output = self.fc1(context)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

def preprocess_data(df):
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df = df.sort_values(by=['subreddit', 'week_start_date'])
    
    subreddits = df['subreddit'].unique()
    all_weeks = sorted(df['week_start_date'].unique())
    
    complete_index = pd.MultiIndex.from_product(
        [subreddits, all_weeks], 
        names=['subreddit', 'week_start_date']
    )

    df_complete = df.set_index(['subreddit', 'week_start_date']).reindex(complete_index)
    df_complete = df_complete.reset_index()
    df_complete['has_data'] = (~df_complete['weekly_victim_count'].isna()).astype(int)
    
    numerical_cols = [
        'submission_count', 'comment_count', 'framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing'
    ]
    
    week_victim_map = {}
    for _, row in df[~df['weekly_victim_count'].isna()].iterrows():
        week_num = row.get("week_num")
        if pd.notna(week_num):
            if week_num not in week_victim_map:
                week_victim_map[week_num] = row['weekly_victim_count']

    sorted_weeks = sorted(week_victim_map.keys())
    week_victim_map = {week: week_victim_map[week] for week in sorted_weeks}

    unique_weeks = sorted(df_complete['week_start_date'].unique())
    
    if len(unique_weeks) != len(sorted_weeks):
        week_mapping = {unique_weeks[i]: i+1 for i in range(len(unique_weeks))}
        df_complete['week_num'] = df_complete['week_start_date'].map(week_mapping)
        
        for i, week in enumerate(unique_weeks):
            if i < len(sorted_weeks):
                week_num = sorted_weeks[i]
                victim_count = week_victim_map.get(week_num, np.nan)
                df_complete.loc[df_complete['week_start_date'] == week, 'weekly_victim_count'] = victim_count
    else:
        week_mapping = {unique_weeks[i]: sorted_weeks[i] for i in range(len(unique_weeks))}
        df_complete['week_num'] = df_complete['week_start_date'].map(week_mapping)
        df_complete['weekly_victim_count'] = df_complete['week_num'].map(week_victim_map)

    victim_mean = df['weekly_victim_count'].mean()
    df_complete['weekly_victim_count'] = df_complete['weekly_victim_count'].fillna(victim_mean)
    
    for col in numerical_cols:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].fillna(0)

    df_complete['prev_weekly_victim_count'] = df_complete.groupby('subreddit')['weekly_victim_count'].shift(1)
    df_complete['prev_weekly_victim_count'] = df_complete['prev_weekly_victim_count'].fillna(victim_mean)
    
    min_date = df_complete['week_start_date'].min()
    df_complete['days_since_start'] = (df_complete['week_start_date'] - min_date).dt.days
    
    if df_complete[numerical_cols + ['weekly_victim_count', 'prev_weekly_victim_count']].isna().any().any():
        print("Warning: DataFrame still contains NaN values after preprocessing")
    
    return df_complete

def prepare_data_for_training(df, target_col='weekly_victim_count', test_size=0.3, seq_length=8, scaler_type='standard'):
    feature_cols = [
        'submission_count', 'comment_count', 'framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing', 'prev_weekly_victim_count',
        'days_since_start'
    ]
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    if df[feature_cols].isna().any().any():
        raise ValueError("Input features contain NaN values")
    
    if df[target_col].isna().any():
        raise ValueError("Target variable contains NaN values")
    
    # Get unique weeks for proper splitting
    unique_weeks = df['week_start_date'].unique()
    split_week_idx = int(len(unique_weeks) * (1 - test_size))
    split_week = unique_weeks[split_week_idx]
    
    # Split based on the week boundary
    train_mask = df['week_start_date'] < split_week
    test_mask = df['week_start_date'] >= split_week
    
    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_col].values.reshape(-1, 1)
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, target_col].values.reshape(-1, 1)
    
    # Store the test weeks for later plotting
    test_weeks = df.loc[test_mask, 'week_start_date'].unique()
    
    # Scale features
    if scaler_type == 'standard':
        X_scaler = StandardScaler()
    else:
        X_scaler = MinMaxScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Scale target
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Create dataset objects
    train_dataset = RedditHateCrimeDataset(X_train_tensor, y_train_tensor, seq_length)
    test_dataset = RedditHateCrimeDataset(X_test_tensor, y_test_tensor, seq_length)
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'test_weeks': test_weeks,
        'feature_names': feature_cols,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }

def train_model(train_dataset, test_dataset, input_size, hidden_size=64, num_layers=2, 
               learning_rate=0.001, batch_size=16, num_epochs=100, patience=20):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, output_size=1).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, history

def evaluate_model(model, test_dataset, y_scaler, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            pred = outputs.cpu().numpy()
            true = y_batch.numpy()
            
            pred = y_scaler.inverse_transform(pred)
            true = y_scaler.inverse_transform(true.reshape(-1, 1))
            
            predictions.extend(pred)
            actuals.extend(true)
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    valid_indices = ~(np.isnan(predictions) | np.isnan(actuals))
    if not np.all(valid_indices):
        predictions = predictions[valid_indices]
        actuals = actuals[valid_indices]
    
    if len(predictions) == 0 or len(actuals) == 0:
        return {
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'predictions': [],
            'actuals': []
        }
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')
    print(f'Test R²: {r2:.4f}')
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }

def plot_results(history, eval_results, plot_dates, title=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    if isinstance(plot_dates[0], str):
        plot_dates = pd.to_datetime(plot_dates)
    
    ax2.plot(plot_dates, eval_results['actuals'], label='Actual')
    ax2.plot(plot_dates, eval_results['predictions'], label='Predicted', linestyle='--')
    
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Hate Crimes')
    ax2.legend()
    ax2.grid(True)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig('hate_crime_prediction_results.png')
    plt.show()

def main():
    file_path = '/Users/meghanreilly/Desktop/BAP-Comp-IR/blm_with_hate_crimes.csv'
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Preprocessing data...")
    try:
        df_processed = preprocess_data(df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return
    
    print("Preparing data for training...")
    try:
        data = prepare_data_for_training(
            df_processed,
            target_col='weekly_victim_count',
            test_size=0.3,
            seq_length=8,
            scaler_type='minmax'
        )
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    input_size = len(data['feature_names'])
    print(f"Training LSTM model with {input_size} features...")
    
    try:
        model, history = train_model(
            data['train_dataset'],
            data['test_dataset'],
            input_size,
            hidden_size=64,
            num_layers=2,
            learning_rate=0.001,
            batch_size=16,
            num_epochs=100,
            patience=15
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    print("Evaluating model...")
    try:
        eval_results = evaluate_model(model, data['test_dataset'], data['y_scaler'], batch_size=16)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Get test dates accounting for sequence length
    test_weeks = data['test_weeks']
    plot_dates = test_weeks[16:]  # Skip first 16 weeks needed for first prediction
    
    print(f"\nDate alignment check:")
    print(f"Total test weeks: {len(test_weeks)}")
    print(f"Plot dates available: {len(plot_dates)}")
    print(f"Predictions available: {len(eval_results['actuals'])}")
    
    # Ensure equal lengths
    min_length = min(len(plot_dates), len(eval_results['actuals']))
    plot_dates = plot_dates[:min_length]
    eval_results['actuals'] = eval_results['actuals'][:min_length]
    eval_results['predictions'] = eval_results['predictions'][:min_length]
    
    assert len(plot_dates) == len(eval_results['actuals']), "Date-prediction mismatch"
    
    try:
        plot_results(history, eval_results, plot_dates, title="Hate Crime Predictions")
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    try:
        torch.save(model.state_dict(), 'hate_crime_lstm_model.pth')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()