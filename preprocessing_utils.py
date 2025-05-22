import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def preprocess_data_og(df):
    """
    Preprocess Reddit hate crime data for time series analysis.
    
    This function handles:
    - Missing data imputation
    - Time series alignment
    - Feature engineering
    - Preparation for sequence-based modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data containing Reddit metrics and hate crime counts
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe ready for model training
    """
    # Ensure datetime format
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    # df = df.sort_values(by=['subreddit', 'week_start_date'])
    df = df.sort_values(by=['week_start_date', 'subreddit'])
    
    # Create complete time series grid (for all subreddits and all weeks)
    subreddits = df['subreddit'].unique()
    all_weeks = sorted(df['week_start_date'].unique())
    
    complete_index = pd.MultiIndex.from_product(
        [subreddits, all_weeks], 
        names=['subreddit', 'week_start_date']
    )


    df_complete = df.set_index(['subreddit', 'week_start_date']).reindex(complete_index)
    print("df size: ", df_complete.shape)
    # exit(0)
    df_complete = df_complete.reset_index()

    # TODO change weekly_victim_count to smth else
    # df_complete['has_data'] = (~df_complete['weekly_victim_count'].isna()).astype(int)
   

    # Define numerical columns that need imputation
    numerical_cols = [
        'submission_count', 'comment_count', 'framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing'
    ]
    
    # Create week to victim count mapping
    week_victim_map = {}
    for _, row in df[~df['weekly_victim_count'].isna()].iterrows():
        week_num = row.get("week_num")
        if pd.notna(week_num):
            if week_num not in week_victim_map:
                week_victim_map[week_num] = row['weekly_victim_count']

    sorted_weeks = sorted(week_victim_map.keys())
    week_victim_map = {week: week_victim_map[week] for week in sorted_weeks}
    
    # plt.plot([int(week) for week in list(week_victim_map.keys())], [int(count) for count in list(week_victim_map.values())])
    # plt.xlabel("Week number")
    # plt.ylabel("Numer of hate crimes")
    # plt.show()
              

    unique_weeks = sorted(df_complete['week_start_date'].unique())
    
    # Handle different week numbering scenarios
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

     
   
    # # Impute missing victim counts with mean
    victim_mean = df['weekly_victim_count'].mean()
    df_complete['weekly_victim_count'] = df_complete['weekly_victim_count'].fillna(victim_mean)
    
    # Impute missing numerical features with zeros
    for col in numerical_cols:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].fillna(0)


    # Add lagged feature: previous week's victim count per subreddit
    df_complete['prev_weekly_victim_count'] = df_complete.groupby('subreddit')['weekly_victim_count'].shift(1)
    df_complete['prev_weekly_victim_count'] = df_complete['prev_weekly_victim_count'].fillna(victim_mean)
    
    # Add time feature: days since start of dataset
    min_date = df_complete['week_start_date'].min()
    df_complete['days_since_start'] = (df_complete['week_start_date'] - min_date).dt.days
    
    # Final check for NaN values
    if df_complete[numerical_cols + ['weekly_victim_count', 'prev_weekly_victim_count']].isna().any().any():
        print("Warning: DataFrame still contains NaN values after preprocessing")
    
    # Drop unnecessary columnsdf['month'] = 
    print(("DF SIZE: ",    df_complete.shape))

    return df_complete

def preprocess_data(df):
    """
    Preprocess Reddit hate crime data for time series analysis.
    
    This function handles:
    - Missing data imputation
    - Time series alignment
    - Feature engineering
    - Preparation for sequence-based modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data containing Reddit metrics and hate crime counts
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe ready for model training
    """
    # Ensure datetime format
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df = df.sort_values(by=['subreddit', 'week_start_date'])
    
    # Create complete time series grid (for all subreddits and all weeks)
    subreddits = df['subreddit'].unique()
    all_weeks = sorted(df['week_start_date'].unique())
    
    complete_index = pd.MultiIndex.from_product(
        [subreddits, all_weeks], 
        names=['subreddit', 'week_start_date']
    )


    df_cols = df.columns.tolist()

    # Initialize column names

    numerical_cols = [
        'submission_count', 'comment_count', 'framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing'
    ]
    feature_cols = ['framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing']
    
    df_cols = ["week_num"] + [f"{subreddit}_{col}" for subreddit in subreddits for col in feature_cols] 

    # print("df_cols: ", df_cols)

    df_complete = pd.DataFrame(0, index=range(len(all_weeks)) ,columns=df_cols)
    df_complete["week_num"] = df_complete.index + 1
    
    for _, row in df.iterrows():
        subreddit = row['subreddit']
        week_num = row['week_num']
        df_complete.loc[week_num-1, "week_start_date"] = row["week_start_date"]


        for feat in feature_cols:
            col_name = f"{subreddit}_{feat}"

            df_complete.loc[week_num-1, col_name] = row[feat]
            # df_complete["week_num"][col_name] = row[feat]
        
    # Create week to victim count mapping
    week_victim_map = {}
    for _, row in df[~df['weekly_victim_count'].isna()].iterrows():
        week_num = row.get("week_num")
        if pd.notna(week_num):
            if week_num not in week_victim_map:
                week_victim_map[week_num] = row['weekly_victim_count']

    sorted_weeks = sorted(week_victim_map.keys())
    week_victim_map = {week: week_victim_map[week] for week in sorted_weeks}
    print(week_victim_map)
    for week, victim_count in week_victim_map.items():
        row_idx = week - 1
        df_complete.loc[row_idx, "weekly_victim_count"] = victim_count
        if row_idx < 1:
            df_complete.loc[row_idx, "prev_weekly_victim_count"] = sum(week_victim_map.values()) / len(week_victim_map)
        else:
            df_complete.loc[week-1, "prev_weekly_victim_count"] = df_complete.loc[row_idx-1, "weekly_victim_count"]

            
    print("df_complete: ", df_complete.head())
    print("df shape: ", df_complete.shape)
    return  df_complete
    exit(0)

    df_complete = df.set_index(['subreddit', 'week_start_date']).reindex(complete_index)
    df_complete = df_complete.reset_index()

    # TODO change weekly_victim_count to smth else
    # df_complete['has_data'] = (~df_complete['weekly_victim_count'].isna()).astype(int)
   

    # Define numerical columns that need imputation
   
    
    # Create week to victim count mapping
    week_victim_map = {}
    for _, row in df[~df['weekly_victim_count'].isna()].iterrows():
        week_num = row.get("week_num")
        if pd.notna(week_num):
            if week_num not in week_victim_map:
                week_victim_map[week_num] = row['weekly_victim_count']

    sorted_weeks = sorted(week_victim_map.keys())
    week_victim_map = {week: week_victim_map[week] for week in sorted_weeks}

    unique_weeks = sorted(df_complete['week_start_date'].unique())
    
    # Handle different week numbering scenarios
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

     
   
    # # Impute missing victim counts with mean
    victim_mean = df['weekly_victim_count'].mean()
    df_complete['weekly_victim_count'] = df_complete['weekly_victim_count'].fillna(victim_mean)
    
    # Impute missing numerical features with zeros
    for col in numerical_cols:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].fillna(0)


    # Add lagged feature: previous week's victim count per subreddit
    df_complete['prev_weekly_victim_count'] = df_complete.groupby('subreddit')['weekly_victim_count'].shift(1)
    df_complete['prev_weekly_victim_count'] = df_complete['prev_weekly_victim_count'].fillna(victim_mean)
    
    # Add time feature: days since start of dataset
    min_date = df_complete['week_start_date'].min()
    df_complete['days_since_start'] = (df_complete['week_start_date'] - min_date).dt.days
    
    # Final check for NaN values
    if df_complete[numerical_cols + ['weekly_victim_count', 'prev_weekly_victim_count']].isna().any().any():
        print("Warning: DataFrame still contains NaN values after preprocessing")
    
    # Drop unnecessary columnsdf['month'] = 
    print(("DF SIZE: ",    df_complete.shape))

    exit(0)
    return df_complete

def prepare_data_for_training(df, target_col='weekly_victim_count', test_size=0.3, seq_length=8, scaler_type='standard'):
    """
    Prepare preprocessed data for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe from preprocess_data function
    target_col : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
    seq_length : int
        Length of input sequences for recurrent models
    scaler_type : str
        Type of scaler to use ('standard' or 'minmax')
        
    Returns:
    --------
    dict
        Dictionary containing train/test datasets and related information
    """
    feature_cols = [
        'submission_count', 'comment_count', 'framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing', 'prev_weekly_victim_count',
        'days_since_start'
    ]
    
    # Validate input data
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    if df[feature_cols].isna().any().any():
        raise ValueError("Input features contain NaN values")
    
    if df[target_col].isna().any():
        raise ValueError("Target variable contains NaN values")
    
    # Get unique weeks for proper time-based splitting
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

def prepare_data_for_cross_validation(df, target_col='weekly_victim_count', seq_length=8):
    """
    Prepare data for time series cross-validation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe from preprocess_data function
    target_col : str
        Name of the target column
    seq_length : int
        Length of input sequences for recurrent models
        
    Returns:
    --------
    tuple
        X and y arrays ready for cross-validation
    """
    feature_cols = [
        'submission_count', 'comment_count', 'framing', 'echo_chamber',
        'sentiment', 'hostility', 'dehumanizing', 'prev_weekly_victim_count',
        'days_since_start'
    ]
    
    # Validate input data
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    if df[feature_cols].isna().any().any():
        raise ValueError("Input features contain NaN values")
    
    if df[target_col].isna().any():
        raise ValueError("Target variable contains NaN values")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y, feature_cols