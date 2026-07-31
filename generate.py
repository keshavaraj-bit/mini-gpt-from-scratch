# Load the trained model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTLanguageModel().to(device)

model.load_state_dict(
    torch.load("gpt_shakespeare.pth", map_location=device)
)

model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(context, max_new_tokens=500)

print(decode(generated[0].tolist()))