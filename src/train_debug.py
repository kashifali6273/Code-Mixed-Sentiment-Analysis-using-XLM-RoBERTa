from datasets import load_dataset

# Try to load the LinCE dataset for sentiment analysis
try:
    # List available configurations (sub-tasks)
    print("Available configurations for LinCE:")
    lince_configs = load_dataset('lince', trust_remote_code=True)
    print(lince_configs.keys())
    
    # Let's try to load the 'sentiment_analysis' subset for Spanish-English
    print("\nLoading sentiment_analysis - spa_eng subset...")
    dataset = load_dataset('lince', 'sentiment_analysis_spa_eng', trust_remote_code=True)
    
    # Print a sample to see the structure
    print("\nSample from the training set:")
    print(dataset['train'][0])
    
except Exception as e:
    print(f"An error occurred: {e}")