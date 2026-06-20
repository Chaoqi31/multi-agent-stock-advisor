import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

def load_trained_sentiment_model(model_path=os.getenv("QWEN_SENTIMENT_MODEL", "/root/code/Finance/qwen_sentiment_model")):
    """Load the trained sentiment analysis model"""
    print("Loading the trained sentiment analysis model...")

    # Load the base model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        os.getenv("QWEN_BASE_MODEL", "/root/code/Finance/Qwen"),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()
    return model, tokenizer

def create_sentiment_test_prompt(text, stock_symbol="STOCK"):
    """Create the sentiment analysis test prompt"""
    system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
    
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant:"""
    
    return conversation

def predict_sentiment(model, tokenizer, text, stock_symbol="STOCK"):
    """Predict the sentiment score"""
    prompt = create_sentiment_test_prompt(text, stock_symbol)

    # Encode the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate the prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the predicted sentiment score
    assistant_response = generated_text.split("Assistant:")[-1].strip()

    # Try to extract a number
    try:
        sentiment_score = int(assistant_response.split()[0])
        if 1 <= sentiment_score <= 5:
            return sentiment_score
    except:
        pass

    return None

def test_sentiment_model():
    """Test the sentiment analysis model"""
    # Load the model
    model, tokenizer = load_trained_sentiment_model()

    # Test data
    test_cases = [
        ("Apple reported strong quarterly earnings with revenue growth of 15%", "AAPL"),
        ("Apple faces supply chain disruptions and production delays", "AAPL"),
        ("Apple announces new iPhone with innovative features", "AAPL"),
        ("Apple stock price remains stable amid market volatility", "AAPL"),
        ("Apple CEO resigns amid scandal and controversy", "AAPL"),
        ("Tesla delivers record number of vehicles in Q4", "TSLA"),
        ("Microsoft announces major layoffs affecting 10,000 employees", "MSFT"),
        ("Google reports disappointing ad revenue decline", "GOOGL"),
        ("Amazon Prime membership reaches new milestone", "AMZN"),
        ("Netflix loses subscribers for the first time", "NFLX")
    ]
    
    print("\n=== Sentiment Analysis Model Test Results ===")
    for i, (text, symbol) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"News: {text}")
        print(f"Stock: {symbol}")

        predicted_sentiment = predict_sentiment(model, tokenizer, text, symbol)

        if predicted_sentiment:
            sentiment_map = {1: "Negative", 2: "Somewhat Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}
            print(f"Predicted sentiment: {predicted_sentiment} ({sentiment_map[predicted_sentiment]})")
        else:
            print("Predicted sentiment: parsing failed")

    # Test with real data
    print("\n=== Real Data Test ===")
    try:
        df = pd.read_csv(os.path.join(os.getenv("DATA_DIR", "nasdaq_news_sentiment"), "1.csv"), nrows=10)
        df = df[df['Lsa_summary'].notna() & df['sentiment_deepseek'].notna()]

        correct_predictions = 0
        total_predictions = 0

        for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
            text = row['Lsa_summary']
            true_sentiment = int(row['sentiment_deepseek'])
            stock_symbol = row.get('Stock_symbol', 'STOCK')

            predicted_sentiment = predict_sentiment(model, tokenizer, text, stock_symbol)

            print(f"\nReal test {i}:")
            print(f"Stock: {stock_symbol}")
            print(f"News summary: {text[:100]}...")
            print(f"True sentiment: {true_sentiment}")
            print(f"Predicted sentiment: {predicted_sentiment}")

            if predicted_sentiment is not None:
                total_predictions += 1
                if predicted_sentiment == true_sentiment:
                    correct_predictions += 1
                    print(f"Correct: ✓")
                else:
                    print(f"Correct: ✗")
            else:
                print(f"Correct: parsing failed")

        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions * 100
            print(f"\nOverall accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")

    except Exception as e:
        print(f"Real data test failed: {e}")

def test_sentiment_distribution():
    """Test the model's performance across different sentiment categories"""
    print("\n=== Sentiment Distribution Test ===")

    model, tokenizer = load_trained_sentiment_model()

    # Test cases for each sentiment category
    sentiment_test_cases = {
        1: [  # Negative
            "Company files for bankruptcy protection",
            "CEO arrested for fraud charges",
            "Stock crashes 50% in single day"
        ],
        2: [  # Somewhat negative
            "Quarterly earnings miss analyst expectations",
            "Company faces regulatory investigation",
            "Product recall affects sales"
        ],
        3: [  # Neutral
            "Company maintains steady performance",
            "Stock price remains unchanged",
            "Quarterly report meets expectations"
        ],
        4: [  # Positive
            "Company beats earnings expectations",
            "New product launch receives positive reviews",
            "Stock price increases 10%"
        ],
        5: [  # Very positive
            "Company reports record-breaking profits",
            "Stock soars 30% on breakthrough announcement",
            "Revolutionary product disrupts entire industry"
        ]
    }

    for expected_sentiment, test_texts in sentiment_test_cases.items():
        print(f"\n--- Testing sentiment category {expected_sentiment} ---")
        correct = 0
        total = len(test_texts)

        for text in test_texts:
            predicted = predict_sentiment(model, tokenizer, text, "TEST")
            match = "✓" if predicted == expected_sentiment else "✗"
            print(f"Expected: {expected_sentiment}, Predicted: {predicted} {match}")
            if predicted == expected_sentiment:
                correct += 1

        accuracy = correct / total * 100
        print(f"Category accuracy: {correct}/{total} = {accuracy:.1f}%")

if __name__ == "__main__":
    test_sentiment_model()
    test_sentiment_distribution() 