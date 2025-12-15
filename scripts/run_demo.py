from truthlense.claim.preprocess import preprocess_claim

if __name__ == "__main__":
    claim = input("Enter a claim: ")

    try:
        processed = preprocess_claim(claim)
        print("\nProcessed Claim:", processed)
    except Exception as e:
        print("Error:", e)
