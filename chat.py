import argparse
from inference.generate import generate_text, load_model, model_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unipilot-v02-step-1000/checkpoint-step-1000.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, args.device)
    print("UniPilot Mini started.\n" + model_summary(model, device, payload))
    history = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "終了"}: break
        history.append(("user", question))
        prompt = "<BOS>" + "".join(f"<{role.upper()}>\n{text}\n" for role, text in history[-6:]) + "<ASSISTANT>\n"
        answer, _ = generate_text(model, tokenizer, prompt, args.max_new_tokens, args.temperature,
                                  args.top_k, args.top_p, args.repetition_penalty)
        answer = answer.split("<", 1)[0].strip()
        print(f"\nUniPilot Mini: {answer}")
        history.append(("assistant", answer))


if __name__ == "__main__": main()
