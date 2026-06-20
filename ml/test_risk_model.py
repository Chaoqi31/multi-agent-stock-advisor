import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

def load_trained_risk_model(model_path=os.getenv("QWEN_RISK_MODEL", "/root/code/Finance/qwen_risk_model")):
    """Load the trained risk assessment model"""
    print("Loading the trained risk assessment model...")

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

def create_risk_test_prompt(text, stock_symbol="STOCK"):
    """Create the risk assessment test prompt"""
    system_prompt = "Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. 1 summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."
    
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 3

User: {user_content}
Assistant:"""
    
    return conversation

def predict_risk(model, tokenizer, text, stock_symbol="STOCK"):
    """Predict the risk score"""
    prompt = create_risk_test_prompt(text, stock_symbol)

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

    # Extract the predicted risk score
    assistant_response = generated_text.split("Assistant:")[-1].strip()

    # Try to extract a number
    try:
        risk_score = int(assistant_response.split()[0])
        if 1 <= risk_score <= 5:
            return risk_score
    except:
        pass

    return None

def test_risk_model():
    """Test the risk assessment model"""
    # Load the model
    model, tokenizer = load_trained_risk_model()

    # Test data
    test_cases = [
        ("Apple reported strong quarterly earnings with revenue growth of 15%", "AAPL"),
        ("Apple faces major supply chain disruptions and production delays", "AAPL"),
        ("Apple announces bankruptcy filing and CEO resignation", "AAPL"),
        ("Apple stock price remains stable amid market volatility", "AAPL"),
        ("Apple receives regulatory approval for new product launch", "AAPL"),
        ("Tesla recalls 500,000 vehicles due to safety concerns", "TSLA"),
        ("Microsoft announces layoffs affecting 10,000 employees", "MSFT")
    ]
    
    print("\n=== Risk Assessment Model Test Results ===")
    for i, (text, symbol) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"News: {text}")
        print(f"Stock: {symbol}")

        predicted_risk = predict_risk(model, tokenizer, text, symbol)

        if predicted_risk:
            risk_map = {1: "Very Low Risk", 2: "Low Risk", 3: "Moderate Risk", 4: "High Risk", 5: "Very High Risk"}
            print(f"Predicted risk: {predicted_risk} ({risk_map[predicted_risk]})")
        else:
            print("Predicted risk: parsing failed")

    # Test with real data
    print("\n=== Real Data Test ===")
    try:
        df = pd.read_csv(os.path.join(os.getenv("DATA_DIR", "risk_nasdaq"), "2.csv"), nrows=5)
        df = df[df['Lsa_summary'].notna() & df['risk_deepseek'].notna()]

        for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
            text = row['Lsa_summary']
            true_risk = int(row['risk_deepseek'])
            stock_symbol = row.get('Stock_symbol', 'STOCK')

            predicted_risk = predict_risk(model, tokenizer, text, stock_symbol)

            print(f"\nReal test {i}:")
            print(f"Stock: {stock_symbol}")
            print(f"News summary: {text[:100]}...")
            print(f"True risk: {true_risk}")
            print(f"Predicted risk: {predicted_risk}")
            print(f"Correct: {'✓' if predicted_risk == true_risk else '✗'}")

    except Exception as e:
        print(f"Real data test failed: {e}")

if __name__ == "__main__":
    test_risk_model() 