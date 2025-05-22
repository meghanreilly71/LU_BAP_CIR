import os
import json
import csv
import zstandard as zstd
from glob import glob
import re

def extract_data_from_zst(file_path):
    """
    Extract data from a zst compressed file.
    Each line in the file is expected to be a JSON object.
    """
    with open(file_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def process_files(input_dir, output_comments_path, output_submissions_path):
    """
    Process all zst files in the input directory and save to two CSV files:
    - One for comments
    - One for submissions
    """
    # Find all zst files
    comment_files = glob(os.path.join(input_dir, "*_comments.zst"))
    submission_files = glob(os.path.join(input_dir, "*_submissions.zst"))
    
    # Define fields for CSVs
    comment_fields = [
        'id', 'author', 'subreddit', 'body', 'score', 'created_utc', 
        'parent_id', 'link_id', 'distinguished'
    ]
    
    submission_fields = [
        'id', 'author', 'subreddit', 'title', 'selftext', 'url', 'score',
        'created_utc', 'num_comments', 'distinguished', 'is_self'
    ]
    
    # Process comment files
    with open(output_comments_path, 'w', newline='', encoding='utf-8') as comments_csv:
        comment_writer = csv.DictWriter(comments_csv, fieldnames=comment_fields)
        comment_writer.writeheader()
        
        for file_path in comment_files:
            subreddit = re.search(r'([^/\\]+)_comments\.zst$', file_path).group(1)
            print(f"Processing comments from r/{subreddit}...")
            
            for comment in extract_data_from_zst(file_path):
                # Extract only the fields we want
                row = {field: comment.get(field, '') for field in comment_fields}
                comment_writer.writerow(row)
    
    # Process submission files
    with open(output_submissions_path, 'w', newline='', encoding='utf-8') as submissions_csv:
        submission_writer = csv.DictWriter(submissions_csv, fieldnames=submission_fields)
        submission_writer.writeheader()
        
        for file_path in submission_files:
            subreddit = re.search(r'([^/\\]+)_submissions\.zst$', file_path).group(1)
            print(f"Processing submissions from r/{subreddit}...")
            
            for submission in extract_data_from_zst(file_path):
                # Extract only the fields we want
                row = {field: submission.get(field, '') for field in submission_fields}
                submission_writer.writerow(row)

if __name__ == "__main__":
    import io
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine Reddit data files into CSVs')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing zst files')
    parser.add_argument('--output_comments', type=str, default='all_comments.csv', help='Output path for comments CSV')
    parser.add_argument('--output_submissions', type=str, default='all_submissions.csv', help='Output path for submissions CSV')
    
    args = parser.parse_args()
    
    process_files(args.input_dir, args.output_comments, args.output_submissions)
    print("Processing complete!")