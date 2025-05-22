import pandas as pd
from datetime import datetime, timedelta

# Load the dataset
df = pd.read_csv('hate_crime.csv')

# Filter for Anti-Black hate crimes
df_black = df[df['bias_desc'] == 'Anti-Black or African American']

# Convert incident_date to datetime
df_black['incident_date'] = pd.to_datetime(df_black['incident_date'])

# Filter for the time period (2020-01-01 to 2022-12-31)
mask = (df_black['incident_date'] >= '2020-01-01') & (df_black['incident_date'] <= '2022-12-31')
df_filtered = df_black[mask]

# Define a function to assign week start dates
def get_week_start(date):
    # Find the number of days to subtract to get to the previous Wednesday (or 0 if it's a Wednesday)
    # For weeks starting on January 1st
    start_date = datetime(2020, 1, 1)
    days_since_start = (date - start_date).days
    week_number = days_since_start // 7
    return start_date + timedelta(days=week_number * 7)

# Apply the function to create week start dates
df_filtered['week_start_date'] = df_filtered['incident_date'].apply(get_week_start)

# Group by the start date of each week, summing the victim counts
weekly_counts = df_filtered.groupby('week_start_date')['victim_count'].sum().reset_index()

# Format the dates to YYYY-MM-DD format
weekly_counts['week_start_date'] = weekly_counts['week_start_date'].dt.strftime('%Y-%m-%d')

# Rename columns for clarity
weekly_counts = weekly_counts.rename(columns={'victim_count': 'weekly_victim_count'})

# Sort by date
weekly_counts = weekly_counts.sort_values('week_start_date')

# Save to CSV
weekly_counts.to_csv('hate_crimes_weekly.csv', index=False)

print(f"Analysis complete. Output saved to 'hate_crimes_weekly.csv'")
print(f"Total weeks analyzed: {len(weekly_counts)}")
print(f"First week starts on: {weekly_counts['week_start_date'].iloc[0]}")
print(f"Total victim count: {weekly_counts['weekly_victim_count'].sum()}")