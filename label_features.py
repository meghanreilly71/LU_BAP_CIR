import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
import os
from tqdm import tqdm
import requests
import time
import json

def group_submissions_with_comments(submissions_df, comments_df):
    # Create a dictionary with submission_id as keys and comments as values
    submission_comments = {}
    
    # Group comments by submission_id
    for _, comment in comments_df.iterrows():
        submission_id = comment['link_id'].split('_')[1]  # Reddit format is t3_SUBMISSION_ID
        if submission_id not in submission_comments:
            submission_comments[submission_id] = []
        submission_comments[submission_id].append(comment)
    
    print("Total comments grouped:", sum(len(v) for v in submission_comments.values()))

    if 'comments' not in submissions_df.columns:
        submissions_df['comments'] = [[] for _ in range(len(submissions_df))]

    # Add comments to each submission
    for i, submission in submissions_df.iterrows():
        submission_id = submission['id']
        if submission_id in submission_comments:
            submissions_df.at[i, 'comments'] = submission_comments[submission_id]
        else:
            submissions_df.at[i, 'comments'] = []
    
    return submissions_df

def divide_into_weeks_and_subreddits(grouped_df):
    """Divide submissions into weeks and then by subreddit within each week"""
    # Convert timestamps to datetime
    grouped_df['created_utc'] = pd.to_datetime(grouped_df['created_utc'], unit='s')
    
    # Define start date and calculate week boundaries
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Generate week intervals
    weeks = []
    current_date = start_date
    week_num = 1
    
    while current_date <= end_date:
        week_end = current_date + timedelta(days=7)
        weeks.append({
            'week_num': week_num,
            'start_date': current_date,
            'end_date': week_end,
            'subreddits': {}  # Dictionary to hold submissions by subreddit
        })
        current_date = week_end
        week_num += 1
    
    # Assign submissions to weeks and subreddits
    for _, submission in grouped_df.iterrows():
        submission_date = submission['created_utc']
        subreddit = submission['subreddit']
        
        for week in weeks:
            if week['start_date'] <= submission_date < week['end_date']:
                # Initialize subreddit entry if it doesn't exist
                if subreddit not in week['subreddits']:
                    week['subreddits'][subreddit] = []
                
                # Add submission to the appropriate subreddit
                week['subreddits'][subreddit].append(submission)
                break
    
    return weeks

def prepare_subreddit_week_for_analysis(week_data, subreddit):
    """Prepare a specific subreddit's data for a week for GPT analysis."""
    submissions = week_data['subreddits'].get(subreddit, [])
    
    # Count submissions and comments
    submission_count = len(submissions)
    comment_count = sum(len(submission.get('comments', [])) for submission in submissions)
    
    if submission_count == 0:
        return None  # Skip if no submissions
    
    # Format content for analysis
    formatted_content = f"Week {week_data['week_num']}: {week_data['start_date'].strftime('%Y-%m-%d')} to {week_data['end_date'].strftime('%Y-%m-%d')}\n"
    formatted_content += f"Subreddit: r/{subreddit}\n\n"
    formatted_content += f"Total submissions: {submission_count}\n"
    formatted_content += f"Total comments: {comment_count}\n\n"
    
    # Include sample of submissions and comments (limited to avoid exceeding token limits)
    sample_size = min(100, submission_count)
    sampled_submissions = submissions[:sample_size]
    
    for i, submission in enumerate(sampled_submissions):
        formatted_content += f"SUBMISSION {i+1}:\n"
        formatted_content += f"Title: {submission.get('title', 'No title')}\n"
        selftext = submission.get('selftext', '')
        if not isinstance(selftext, str): 
            selftext = ''
        formatted_content += f"Content: {selftext[:1000]}...\n\n"

        
        # Add a sample of comments
        comments = submission.get('comments', [])
        comment_sample = min(10, len(comments))  # Limit to 10 comments per submission
        
        formatted_content += f"SAMPLE COMMENTS ({comment_sample} of {len(comments)}):\n"
        for j, comment in enumerate(comments[:comment_sample]):
            formatted_content += f"Comment {j+1}: {comment.get('body', 'No content')[:500]}...\n\n"
    
    return {
        "week_num": week_data['week_num'],
        "start_date": week_data['start_date'],
        "end_date": week_data['end_date'],
        "subreddit": subreddit,
        "submission_count": submission_count,
        "comment_count": comment_count,
        "formatted_content": formatted_content
    }

def analyze_subreddit_week_with_gpt(subreddit_week_data, openai_api_key):
    """Analyze subreddit's weekly data using GPT-4 API."""
    client = OpenAI(api_key=openai_api_key)
    prompt = f"""
    Please analyze the following Reddit content about Black Lives Matter from r/{subreddit_week_data['subreddit']} for the period {subreddit_week_data['start_date'].strftime('%Y-%m-%d')} to {subreddit_week_data['end_date'].strftime('%Y-%m-%d')}.

    CONTENT:
    {subreddit_week_data['formatted_content']}

    TASK: Classify the overall discourse according to the dimensions below. Use the example codes and descriptions to choose the most fitting category.


    1. Framing & Narrative Construction (choose one):
       (1) Strongly Oppositional: BLM framed as dangerous, criminal, or a threat to societal order.
       (2) Moderately Oppositional: Criticizes BLM but with some nuance.
       (3) Neutral/Objective: Frames BLM as a movement without explicit support or opposition.
       (4) Moderately Supportive: Expresses sympathy for BLM's goals but may critique certain aspects.
       (5) Strongly Supportive: Frames BLM as necessary for social justice.

    2. Echo Chamber Effects (choose one) :
       (1) Strong Echo Chamber: Almost entirely one-sided discussion, with no dissenting voices.
       (2) Moderate Echo Chamber: Predominantly one-sided but with occasional counterarguments.
       (3) Mixed Exposure: Balanced discussion with both supportive and oppositional perspectives.
       (4) Moderate Counter-Exposure: Primarily consists of opposing viewpoints but allows some engagement.
       (5) Strong Counter-Exposure: Almost entirely dominated by perspectives that challenge prevailing view.

    3. Sentiment Categories (choose one):
       (1) Positive Mobilization: Hopeful, optimistic, calls for unity and reform.
       (2) Moral Outrage: Indignation over injustice.
       (3) Fear & Anxiety: Concerns about safety, riots, government overreach.
       (4) Anger & Hostility: Resentment toward groups, institutions, or opposing views.
       (5) Resignation & Apathy: Posts expressing helplessness or disengagement.

    4. Hostility Level (choose one):
       (1) None: Neutral tone, no emotional hostility
       (2) Irritation: Mild frustration, passive-aggressive or dismissive language
       (3) Contempt: Sarcastic, demeaning, or mocking without threat
       (4) Anger: Direct anger at individuals/groups, often with insults
       (5) Threatening: Implied threat or endorsement of harm
       (6) Incitement: Direct or indirect call for violence

    5. Dehumanizing Language (choose one):
       (1) None: e.g. "They're people with different views."
       (2) Implicit: e.g. "They just don't think like real Americans."
       (3) Mild: e.g. "They're like animals."
       (4) Severe: e.g. "They are parasites that need to be exterminated."

    Also note the volume statistics: {subreddit_week_data['submission_count']} submissions and {subreddit_week_data['comment_count']} comments.

    Here are two example analyses for context:
    ### EXAMPLE 1:
    Subreddit: r/BlackLivesMatter
    Date Range: 2020-06-01 to 2020-06-07
    Sample Content: 
    "These protests are necessary. People are tired of the systemic racism. It's time for the police to be defunded and held accountable."

    ```json
    {{
    "framing": "5",
    "echo_chamber": "3",
    "sentiment": "2",
    "hostility": "2",
    "dehumanizing": "1",
    }}

    ### EXAMPLE 2:
    Subreddit: r/Conservative
    Date Range: 2020-06-01 to 2020-06-07
    Sample Content:
    "BLM is a terrorist organization. These thugs are destroying our cities. If they want a war, they'll get one."

    ```json
    {{
     "framing": "1",
     "echo_chamber": "3",
     "sentiment": "4",
     "hostility": "6",
     "dehumanizing": "4",
    }}
    
    Format your response as a structured JSON object with these fields:
```json
    {{
    "framing": "category number",
    "echo_chamber": "category number",
    "sentiment": "category number",
    "hostility": "category number",
    "dehumanizing": "category number"
    }}
    """
    
    try:
        # Try first with the newer model that supports response_format
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",  # or whatever newer model you have access to
                messages=[
                    {"role": "system", "content": "You are an expert content analyst focused on social media discourse around social movements."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.3
            )
        except Exception as e:
            if "response_format" in str(e):
                # Fall back to older model without response_format
                print(f"Falling back to older model without response_format for r/{subreddit_week_data['subreddit']}")
                response = client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "You are an expert content analyst focused on social media discourse around social movements. Return all answers in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
            else:
                raise e
        
        analysis = response.choices[0].message.content
        
        # Parse the JSON analysis - enhanced error handling
        try:
            # Clean the response before parsing
            json_start = analysis.find('{')
            json_end = analysis.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                clean_json = analysis[json_start:json_end]
                analysis_json = json.loads(clean_json)
            else:
                # If can't find JSON brackets, try to parse the whole string
                analysis_json = json.loads(analysis)
        except json.JSONDecodeError:
            # If JSON parsing fails, extract using regex or return as text
            import re
            # Try to extract keys and values with regex
            analysis_json = {}
            for key in ["framing", "echo_chamber", "sentiment", "hostility", "dehumanizing"]:
                match = re.search(f'"{key}"\\s*:\\s*"?([^,"}}]+)"?', analysis)
                if match:
                    analysis_json[key] = match.group(1)
            
            # If regex extraction failed too
            if not analysis_json:
                analysis_json = {"error": "Failed to parse JSON", "raw_analysis": analysis}
        
        # Create analysis result
        result = {
            "week_num": subreddit_week_data['week_num'],
            "date_range": f"{subreddit_week_data['start_date'].strftime('%Y-%m-%d')} to {subreddit_week_data['end_date'].strftime('%Y-%m-%d')}",
            "subreddit": subreddit_week_data['subreddit'],
            "submission_count": subreddit_week_data['submission_count'],
            "comment_count": subreddit_week_data['comment_count'],
            "analysis": analysis_json
        }
        
        return result
    
    except Exception as e:
        print(f"Error analyzing r/{subreddit_week_data['subreddit']} for week {subreddit_week_data['week_num']}: {e}")
        return {
            "week_num": subreddit_week_data['week_num'],
            "date_range": f"{subreddit_week_data['start_date'].strftime('%Y-%m-%d')} to {subreddit_week_data['end_date'].strftime('%Y-%m-%d')}",
            "subreddit": subreddit_week_data['subreddit'],
            "error": str(e)
        }
    
def run_reddit_analysis_pipeline(submissions_path, comments_path, output_path, openai_api_key, subreddits_to_analyze=None):
    """
    Full pipeline to process and analyze Reddit BLM data by week and subreddit
    
    Args:
        submissions_path: Path to submissions CSV/JSON file
        comments_path: Path to comments CSV/JSON file
        output_path: Path to save analysis results
        openai_api_key: API key for OpenAI GPT
        subreddits_to_analyze: Optional list of subreddits to analyze (if None, analyze all)
    """
    print("Starting Reddit BLM analysis pipeline...")

    if not openai_api_key:
        raise ValueError("OpenAI API key is required but not provided")
    
    # 1. Load data
    print("Loading datasets...")
    if submissions_path.endswith('.csv'):
        submissions_df = pd.read_csv(submissions_path)
        comments_df = pd.read_csv(comments_path)
    else:  # Assume JSON
        submissions_df = pd.read_json(submissions_path)
        comments_df = pd.read_json(comments_path)
    
    # 2. Group submissions with comments
    print("Grouping submissions with their comments...")
    grouped_df = group_submissions_with_comments(submissions_df, comments_df)
    
    # 3. Divide into weekly intervals and by subreddit
    print("Dividing data into weekly intervals and by subreddit...")
    weeks_data = divide_into_weeks_and_subreddits(grouped_df)

    print()
    
    # Get unique subreddits if not specified
    if not subreddits_to_analyze:
        all_subreddits = set()
        for week in weeks_data:
            all_subreddits.update(week['subreddits'].keys())
        subreddits_to_analyze = list(all_subreddits)
    
    print(f"Will analyze {len(subreddits_to_analyze)} subreddits across {len(weeks_data)} weeks")
    
    # 4. Prepare and analyze each week and subreddit combination
    print("Analyzing weeks and subreddits with GPT...")
    analysis_results = []
    
    # Create a dedicated output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process each week
    for week in tqdm(weeks_data, desc="Processing weeks"):
        # Process each subreddit within the week
        for subreddit in subreddits_to_analyze:
            # Skip if subreddit doesn't exist in this week
            if subreddit not in week['subreddits'] or not week['subreddits'][subreddit]:
                continue
            
            # Prepare data for this subreddit and week
            prepared_data = prepare_subreddit_week_for_analysis(week, subreddit)
            if not prepared_data:
                continue
            
            # Rate limit to avoid API throttling
            time.sleep(1)
            
            # Analyze with GPT
            analysis = analyze_subreddit_week_with_gpt(prepared_data, openai_api_key)
            analysis_results.append(analysis)
            
            # Save intermediate results periodically
            if len(analysis_results) % 10 == 0:
                interim_df = pd.DataFrame(analysis_results)
                interim_df.to_csv(f"{output_path}_interim.csv", index=False)
                print(f"Saved interim results: {len(analysis_results)} analyses completed")
    
    # 5. Save final results
    print("Saving final analysis results...")
    results_df = pd.DataFrame(analysis_results)
    
    # Normalize the nested JSON analysis for better CSV export
    if 'analysis' in results_df.columns:
        # Extract fields from the analysis JSON
        for field in ['framing', 'echo_chamber', 'sentiment', 'hostility', 'dehumanizing']:
            results_df[field] = results_df['analysis'].apply(
                lambda x: x.get(field, None) if isinstance(x, dict) else None
            )
    
    # Save CSV
    results_df.to_csv(output_path, index=False)
    
    # Save comprehensive JSON
    results_df.to_json(f"{output_path.replace('.csv', '.json')}", orient='records')
    
    print(f"Analysis complete. Results saved to {output_path}")
    return results_df

# Example usage
if __name__ == "__main__":
    # Set the API keys directly
    openai_api_key = ""
    
    # Paths
    submissions_path = "/Users/meghanreilly/Desktop/BAP-Comp-IR/all_submissions.csv"
    comments_path = "/Users/meghanreilly/Desktop/BAP-Comp-IR/all_comments.csv"
    output_path = "results/blm_reddit_analysis.csv"
    
    # Optional: Specify specific subreddits to analyze
    subreddits_to_analyze = [
        "BlackLivesMatter", 
        "politics", 
        "Conservative", 
        "196",
        "AgainstHateSubreddits",
        "AskThe_Donald",
        "conservatives",
        "ConsumeProduct",
        "democrats",
        "FreeSpeech",
        "MGTOW",
        "progressive",
        "Republican",
        "The_Donald",
    ]
    
    # Run pipeline
    results = run_reddit_analysis_pipeline(
        submissions_path=submissions_path,
        comments_path=comments_path,
        output_path=output_path,
        openai_api_key=openai_api_key,
        subreddits_to_analyze=subreddits_to_analyze
    )