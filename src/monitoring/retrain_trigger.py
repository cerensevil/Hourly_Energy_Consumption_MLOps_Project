from src.training.evaluate_and_promote_flow import evaluate_and_promote


def trigger_retraining(reason):

    print("\n⚠️ Retraining triggered")
    print(f"Reason: {reason}")

    evaluate_and_promote()

    print("✅ Retraining completed")