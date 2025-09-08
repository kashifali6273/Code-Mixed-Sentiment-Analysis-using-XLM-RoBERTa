import pandas as pd
import re
from collections import Counter
from pathlib import Path

# Regex for Urdu script (Arabic Unicode range)
urdu_pattern = re.compile(r"[\u0600-\u06FF]+")

def get_word_frequencies(texts):
    words = " ".join(texts).split()
    return Counter(words)

if __name__ == "__main__":
    stats_path = Path("data/processed/stats")
    stats_path.mkdir(parents=True, exist_ok=True)

    all_texts = []
    urdu_texts = []
    eng_texts = []

    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/{split}.csv")
        texts = df["text"].astype(str).tolist()
        all_texts.extend(texts)

        # Urdu vs Roman/English separation
        for t in texts:
            if urdu_pattern.search(t):
                urdu_texts.append(t)
            else:
                eng_texts.append(t)

    # Frequency counters
    overall_freq = get_word_frequencies(all_texts)
    urdu_freq = get_word_frequencies(urdu_texts)
    eng_freq = get_word_frequencies(eng_texts)

    # Save top 50 words each
    pd.DataFrame(overall_freq.most_common(50), columns=["word", "count"]).to_csv(stats_path / "overall_top50.csv", index=False, encoding="utf-8")
    pd.DataFrame(urdu_freq.most_common(50), columns=["word", "count"]).to_csv(stats_path / "urdu_top50.csv", index=False, encoding="utf-8")
    pd.DataFrame(eng_freq.most_common(50), columns=["word", "count"]).to_csv(stats_path / "eng_top50.csv", index=False, encoding="utf-8")

    print("✅ Word frequency stats saved in data/processed/stats/")
