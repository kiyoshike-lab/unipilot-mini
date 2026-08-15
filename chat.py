import argparse
from inference.generate import generate_text, load_model, model_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sanity-100/checkpoint-step-100.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, args.device)
    print("UniPilot Mini started.\n" + model_summary(model, device, payload))
    history = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "終了"}: break
        history.append(("user", question))
        prompt = "<BOS>" + "".join(f"<{role.upper()}>\n{text}\n" for role, text in history[-6:]) + "<ASSISTANT>\n"
        answer, _ = generate_text(model, tokenizer, prompt)
        answer = answer.split("<", 1)[0].strip()
        print(f"\nUniPilot Mini: {answer}")
        history.append(("assistant", answer))


if __name__ == "__main__": main()
