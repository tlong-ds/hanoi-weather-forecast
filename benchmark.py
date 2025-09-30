import numpy as np
import ast
import re
import time
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples, stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from helper import get_llm, eval_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

np.random.seed(42)

llm_lists = [
    # "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # "gpt-4-turbo",
    # "gpt-4",
]

# nltk.download('twitter_samples')
# nltk.download('stopwords')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[4000:]

test_x = test_pos + test_neg
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

def evaluate_model(name, y_true, y_pred):
    """Prints precision for Positive=1 and Negative=0, plus full report and confusion matrix."""
    print(f"\n=== {name} ===")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nROC-AUC: {roc_auc_score(y_true, y_pred)}")

def process_batch(llm, batch_tweets, batch_idx, start_idx):
    """Process a single batch of tweets - for potential parallel processing"""
    try:
        batch_text = "\n".join([f"Tweet {j+1}: {tweet}" for j, tweet in enumerate(batch_tweets)])
        response = llm.invoke(eval_prompt.format(text=batch_text))
        
        if response and response.content:
            predictions = ast.literal_eval(response.content.strip())
            # Convert to binary efficiently
            batch_pred = np.array([1 if pred == "positive" else 0 for pred in predictions])
            return start_idx, batch_pred, len(batch_tweets), None
        else:
            return start_idx, None, len(batch_tweets), "Empty response"
    except Exception as e:
        return start_idx, None, len(batch_tweets), str(e)

def run_benchmark_parallel(max_workers=3):
    """
    Parallel version - use only if your LLM API supports concurrent requests
    and you don't have strict rate limits
    """
    for model in llm_lists:
        print(f"\nEvaluating model: {model} (Parallel)")
        llm = get_llm(model_name=model)
        y_pred = np.zeros(len(test_x), dtype=int)
        
        batch_size = 100
        total_batches = (len(test_x) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        # Prepare all batches
        batches = []
        for i in range(0, len(test_x), batch_size):
            end_idx = min(i + batch_size, len(test_x))
            batch_tweets = test_x[i:end_idx]
            batches.append((batch_tweets, len(batches), i))
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_batch, llm, batch_tweets, batch_idx, start_idx): (batch_idx, start_idx, len(batch_tweets))
                for batch_tweets, batch_idx, start_idx in batches
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_batch):
                batch_idx, start_idx, expected_size = future_to_batch[future]
                completed += 1
                print(f"Completed batch {completed}/{total_batches}")
                
                try:
                    start_idx_result, batch_pred, expected_size_result, error = future.result()
                    
                    if error:
                        print(f"Error in batch {batch_idx + 1}: {error}")
                    elif batch_pred is not None:
                        end_idx = start_idx + len(batch_pred)
                        y_pred[start_idx:end_idx] = batch_pred
                    
                except Exception as e:
                    print(f"Exception in batch {batch_idx + 1}: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"Total processing time (parallel): {elapsed_time:.2f} seconds")
        
        evaluate_model(f"LLM: {model} (Parallel)", test_y.flatten(), y_pred)

def run_benchmark():
    for model in llm_lists:
        print(f"\nEvaluating model: {model}")
        llm = get_llm(model_name=model)
        y_pred = np.zeros(len(test_x), dtype=int)  # 1D array, correct dtype
        
        # Process tweets in batches - O(n) complexity
        batch_size = 100  # Adjust based on your LLM's context limit
        total_batches = (len(test_x) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for batch_idx, i in enumerate(range(0, len(test_x), batch_size)):
            print(f"Processing batch {batch_idx + 1}/{total_batches}...")
            
            # Get current batch
            end_idx = min(i + batch_size, len(test_x))
            batch_tweets = test_x[i:end_idx]
            
            # Process batch
            start_idx, batch_pred, expected_size, error = process_batch(llm, batch_tweets, batch_idx, i)
            
            if error:
                print(f"Error processing batch {batch_idx + 1}: {error}")
            elif batch_pred is not None and len(batch_pred) == expected_size:
                # Vectorized assignment - O(batch_size) instead of O(batch_size²)
                y_pred[i:end_idx] = batch_pred
            else:
                print(f"Warning: Expected {expected_size} predictions, got {len(batch_pred) if batch_pred is not None else 0}")
            
            # Rate limiting - only sleep if not the last batch
            if batch_idx < total_batches - 1:
                time.sleep(1)
        
        elapsed_time = time.time() - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Average time per batch: {elapsed_time/total_batches:.2f} seconds")
        
        evaluate_model(f"LLM: {model}", test_y.flatten(), y_pred)
        

if __name__ == "__main__":
    # Choose between sequential and parallel processing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        print("Running parallel benchmark...")
        run_benchmark_parallel(max_workers=3)  # Adjust max_workers based on API limits
    else:
        print("Running sequential benchmark...")
        print("Use --parallel flag for parallel processing (if API supports it)")
        run_benchmark()