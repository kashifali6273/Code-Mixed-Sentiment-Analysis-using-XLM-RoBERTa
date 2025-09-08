import random
import pandas as pd

# Expanded phrase banks (Roman Urdu + Urdu script)
positive_urdu = [
    "bohat acha", "mast", "pasand aya", "kamaal", "acha experience",
    "دل خوش ہوگیا", "بہت زبردست", "best", "شاندار", "مزے کا",
    "awesome tha", "pyara", "acha laga", "satisfying", "زبردست",
    "bohot hi acha", "impressive"
]

negative_urdu = [
    "bura", "bekaar", "pasand nahi aya", "ghatia", "slow service",
    "کچرا", "فضول", "بدترین", "زیرو سٹار", "ghanda",
    "ghaleez", "worst tha", "بیکار", "وَسٹ آف ٹائم", "achha nahi"
]

neutral_urdu = [
    "theek thaak", "average", "normal", "chalega", "ok",
    "ٹھیک ہی تھا", "ordinary", "neutral", "casual", "بہت عام",
    "timepass", "decent", "not special", "acceptable", "balanced"
]

positive_eng = [
    "great", "awesome", "loved it", "fantastic", "superb",
    "excellent", "amazing", "brilliant", "outstanding", "wonderful",
    "perfect", "mind-blowing", "very good", "enjoyed a lot", "impressive"
]

negative_eng = [
    "bad", "boring", "hate it", "worst", "terrible",
    "horrible", "disgusting", "pathetic", "very bad", "trash",
    "awful", "lame", "waste", "dreadful", "poor quality"
]

neutral_eng = [
    "okay", "fine", "average", "not bad", "normal",
    "nothing special", "mediocre", "casual", "so-so", "just ok",
    "fair", "balanced", "timepass", "acceptable", "neutral"
]

# Sentence structures
structures = [
    "{u} movie was {e}",
    "{e} service, bilkul {u}",
    "Overall {u}, experience {e}",
    "{e} app hai, bohat {u}",
    "Mujhe laga {u}, but overall {e}",
    "{e} product, totally {u}",
    "This is {e}, sach me {u}",
    "Feedback: {u} and also {e}",
    "{e} quality, {u} performance",
    "{u} feeling after use, really {e}"
]

# Generator function
def make_sentence(urdu_list, eng_list):
    template = random.choice(structures)
    return template.format(u=random.choice(urdu_list), e=random.choice(eng_list))

# Generate dataset
def generate_dataset(n_per_class=3333):
    data = []
    for _ in range(n_per_class):
        data.append((make_sentence(positive_urdu, positive_eng), "positive"))
        data.append((make_sentence(negative_urdu, negative_eng), "negative"))
        data.append((make_sentence(neutral_urdu, neutral_eng), "neutral"))
    random.shuffle(data)
    return pd.DataFrame(data, columns=["text", "label"])

if __name__ == "__main__":
    df = generate_dataset(3333)  # ~10k rows
    df.to_csv("data/raw/ur_en_generated.csv", index=False, encoding="utf-8")
    print("✅ Dataset saved with shape:", df.shape)
    print(df.head(10))
