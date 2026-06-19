import os
import json
import hashlib
import unicodedata
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import jieba
from tqdm import tqdm
import binascii

class NewsDeduplicator:
    def __init__(self):
        self.title_threshold = 0.8
        self.content_threshold = 0.75  # Content overlap threshold, set to 0.75
        self.simhash_threshold = 3
        self.minhash_permutations = 128
        self.processed_data = []

    def unicode_normalize(self, text: str) -> str:
        """Unicode normalization"""
        if not text:
            return ""
        # Unicode standardization
        text = unicodedata.normalize('NFKC', text)
        # Remove redundant whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters
        text = re.sub(r'[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\s\.\!\?\,\;\:]', '', text)
        return text
    
    def edit_distance(self, s1: str, s2: str) -> float:
        """Compute the edit distance and normalize it"""
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

        max_len = max(m, n)
        return 1 - (dp[m][n] / max_len) if max_len > 0 else 0.0

    def text_to_tfidf_vector(self, texts: List[str]) -> np.ndarray:
        """Convert text into TF-IDF vectors"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        try:
            vectors = vectorizer.fit_transform(texts)
            return vectors.toarray()
        except:
            return np.zeros((len(texts), 1000))

    def title_similarity(self, title1: str, title2: str) -> float:
        """Compute title similarity: edit distance + cosine similarity"""
        title1 = self.unicode_normalize(title1)
        title2 = self.unicode_normalize(title2)

        # Edit distance similarity
        edit_sim = self.edit_distance(title1, title2)

        # Cosine similarity
        if title1 and title2:
            vectors = self.text_to_tfidf_vector([title1, title2])
            if vectors.shape[0] == 2:
                cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            else:
                cos_sim = 0.0
        else:
            cos_sim = 0.0

        # Combined similarity: take the average
        return (edit_sim + cos_sim) / 2

    def get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Generate k-shingles"""
        text = self.unicode_normalize(text)
        words = list(jieba.cut(text))
        if len(words) < k:
            return {text}
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}

    def minhash_signature(self, shingles: Set[str]) -> List[int]:
        """Compute the MinHash signature"""
        if not shingles:
            return [0] * self.minhash_permutations

        signature = [float('inf')] * self.minhash_permutations

        for shingle in shingles:
            shingle_bytes = shingle.encode('utf-8')
            for i in range(self.minhash_permutations):
                # Use different salt values to simulate different hash functions
                hash_input = shingle_bytes + str(i).encode('utf-8')
                hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
                signature[i] = min(signature[i], hash_value)

        return signature

    def jaccard_similarity_minhash(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity using MinHash"""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def content_overlap(self, content1: str, content2: str) -> float:
        """Compute content overlap using MinHash"""
        shingles1 = self.get_shingles(content1)
        shingles2 = self.get_shingles(content2)

        sig1 = self.minhash_signature(shingles1)
        sig2 = self.minhash_signature(shingles2)

        return self.jaccard_similarity_minhash(sig1, sig2)

    def hash_string(self, text: str) -> int:
        """Convert a string into an integer hash value"""
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)

    def simhash(self, text: str) -> int:
        """Compute the SimHash value"""
        text = self.unicode_normalize(text)
        words = list(jieba.cut(text))

        if not words:
            return 0

        # Create a 64-bit feature vector
        features = [0] * 64

        for word in words:
            word_hash = self.hash_string(word)
            for i in range(64):
                bit = (word_hash >> i) & 1
                if bit:
                    features[i] += 1
                else:
                    features[i] -= 1

        # Generate the final SimHash value
        simhash_value = 0
        for i in range(64):
            if features[i] > 0:
                simhash_value |= (1 << i)

        return simhash_value

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute the Hamming distance"""
        return bin(hash1 ^ hash2).count('1')

    def semantic_similarity(self, content1: str, content2: str) -> int:
        """Compute semantic similarity (returns the Hamming distance)"""
        hash1 = self.simhash(content1)
        hash2 = self.simhash(content2)
        return self.hamming_distance(hash1, hash2)

    def is_duplicate(self, item1: Dict, item2: Dict) -> bool:
        """Determine whether two news items are duplicates"""
        # Extract titles and content
        title1 = item1.get('title', '') or item1.get('doc', '')[:100]
        title2 = item2.get('title', '') or item2.get('doc', '')[:100]
        content1 = item1.get('doc', '')
        content2 = item2.get('doc', '')

        # Compute the three similarity measures
        title_sim = self.title_similarity(title1, title2)
        content_sim = self.content_overlap(content1, content2)
        semantic_dist = self.semantic_similarity(content1, content2)

        # Apply the triple threshold
        return (title_sim > self.title_threshold and
                content_sim > self.content_threshold and
                semantic_dist <= self.simhash_threshold)
    
    def load_and_preprocess_data(self, csv_file_path: str = os.path.join(os.getenv("DATA_DIR", "/mnt/data/Finance/risk_nasdaq"), "2.csv")):
        """Load and preprocess data from a local CSV file"""
        print("Loading CSV file...")

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            print(f"CSV file loaded successfully, {len(df)} rows in total")

            # Convert to the format we need
            processed_items = []
            for index, row in df.iterrows():
                # Extract the article content, preferring the Article column, falling back to Textrank_summary
                article_content = str(row.get('Article', '')) or str(row.get('Textrank_summary', ''))

                # Extract the title
                title = str(row.get('Article_title', '')) or str(row.get('Stock_symbol', ''))

                # Extract the label (risk score)
                risk_score = row.get('risk_deepseek', '')
                labels = f"risk_score:{risk_score}" if pd.notna(risk_score) else ""

                # Extract the stock symbol
                stock_symbol = str(row.get('Stock_symbol', '')) if pd.notna(row.get('Stock_symbol')) else ""

                # Extract the date
                date = str(row.get('Date', '')) if pd.notna(row.get('Date')) else ""

                processed_item = {
                    'source': 'local_csv',
                    'doc': self.unicode_normalize(article_content),
                    'labels': labels,
                    'title': self.unicode_normalize(title),
                    'stock_symbol': stock_symbol,
                    'date': date,
                    'original_index': index
                }

                if processed_item['doc']:  # Keep only entries that have content
                    processed_items.append(processed_item)

            print(f"Preprocessing complete, number of valid samples: {len(processed_items)}")
            return processed_items

        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            print("Please make sure the CSV file path is correct and the file is readable")
            return []

    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """Perform deduplication"""
        print("Starting deduplication...")
        unique_items = []
        duplicate_count = 0

        for i, current_item in enumerate(tqdm(data, desc="Progress")):
            is_dup = False

            # Compare against the unique items kept so far
            for unique_item in unique_items:
                if self.is_duplicate(current_item, unique_item):
                    is_dup = True
                    duplicate_count += 1
                    break

            if not is_dup:
                unique_items.append(current_item)

        print(f"Deduplication complete:")
        print(f"  Original items: {len(data)}")
        print(f"  Duplicate items: {duplicate_count}")
        print(f"  Retained items: {len(unique_items)}")

        return unique_items

    def save_to_jsonl(self, data: List[Dict], output_file: str):
        """Save data to a JSONL file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Data saved to: {output_file}")

    def process_dataset(self, csv_file_path: str = os.path.join(os.getenv("DATA_DIR", "/mnt/data/Finance/risk_nasdaq"), "2.csv"),
                       output_file: str = "deduplicated_news.jsonl"):
        """Full data processing pipeline"""
        # 1. Load and preprocess data
        data = self.load_and_preprocess_data(csv_file_path)

        if not data:
            print("No valid data was loaded; exiting")
            return []

        # 2. Perform deduplication
        unique_data = self.deduplicate(data)

        # 3. Save the results
        self.save_to_jsonl(unique_data, output_file)

        return unique_data

# Usage example
if __name__ == "__main__":
    # Create a deduplicator instance
    deduplicator = NewsDeduplicator()

    # Process the dataset
    try:
        unique_news = deduplicator.process_dataset(
            csv_file_path=os.path.join(os.getenv("DATA_DIR", "/mnt/data/Finance/risk_nasdaq"), "risk_deepseek_cleaned_nasdaq_news_full.csv"), #  Replace with your own path
            output_file="deduplicated_risk_nasdaq.jsonl"
        )

        print(f"\nProcessing complete! Retained {len(unique_news)} news items in total")

        # Display sample data
        if unique_news:
            print("\nSample data:")
            sample = unique_news[0]
            print(f"Source: {sample['source']}")
            print(f"Stock symbol: {sample.get('stock_symbol', 'None')}")
            print(f"Date: {sample.get('date', 'None')}")
            print(f"Title: {sample.get('title', 'Untitled')[:100]}...")
            print(f"Content: {sample['doc'][:200]}...")
            print(f"Labels: {sample['labels']}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print("Please make sure the required dependencies are installed:")
        print("pip install pandas scikit-learn jieba tqdm numpy")