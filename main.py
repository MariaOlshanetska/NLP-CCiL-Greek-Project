import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8') #to avoid an error we had with Greek letters
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import requests
import unicodedata
import random

# ---------------------- DOWNLOAD & LOAD DATA ----------------------

def download_data(url, filename="names_greek.txt"):
    """Downloads a text file from a given URL if not already present."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print("File downloaded and saved successfully.")
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

def load_data(filename="names_greek.txt"):
    """Loads and processes text data from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        words = [normalize(w.strip().lower()) for w in f.readlines()]
    return words

# ---------------------- DATA NORMALIZATION ----------------------

def normalize(text):
    """Removes diacritics from Greek characters."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

# ---------------------- BUILD CHARACTER MAPPINGS ----------------------

def build_vocab(words):
    """Creates character-to-index and index-to-character mappings."""
    chars = sorted(set("".join(words)))
    chars.remove(" ")  # Remove spaces
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi["."] = 0  # Special token for end of name
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

# ---------------------- BUILD DATASET ----------------------

def build_dataset(words, block_size=3):
    """Builds input-output pairs for the model."""
    X, Y = [], []
    for w in words:
        w = w.replace(" ", "")
        context = [0] * block_size
        for idx, ch in enumerate(w):
            if ch == "ς" and idx < len(w) - 1:
                ch = "σ"  # Convert 'ς' to 'σ' except at the end
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # Shift context
        # Append termination example: signal end of the name.
        X.append(context)
        Y.append(0)
    return torch.tensor(X), torch.tensor(Y)


# ---------------------- INITIALIZE TRAINING PARAMETERS ----------------------

def init_params():
    """Initializes model parameters with reproducibility."""
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((26, 10), generator=g)
    W1 = torch.randn((30, 200), generator=g)
    b1 = torch.randn(200, generator=g)
    W2 = torch.randn((200, 26), generator=g)
    b2 = torch.randn(26, generator=g)
    parameters = [C, W1, b1, W2, b2]
    
    for p in parameters:
        p.requires_grad = True
    
    return parameters

# ---------------------- TRAINING FUNCTION ----------------------

def train(Xtr, Ytr, parameters, epochs=200_000, batch_size=32, lr1=0.1, lr2=0.01):
    """Trains the neural network with minibatch gradient descent."""
    C, W1, b1, W2, b2 = parameters
    loss_log = []

    for i in range(epochs):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))  # Mini-batch sampling

        # Forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update weights
        lr = lr1 if i < epochs // 2 else lr2
        for p in parameters:
            p.data -= lr * p.grad

        if i % 10_000 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")
            loss_log.append(loss.item())

    return loss_log

# ---------------------- EVALUATION FUNCTION ----------------------

@torch.no_grad()
def evaluate(X, Y, parameters):
    """Evaluates model on a given dataset."""
    C, W1, b1, W2, b2 = parameters
    emb = C[X]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()

# ---------------------- VISUALIZATION FUNCTION ----------------------

def plot_embedding(C, itos):
    """Visualizes character embeddings in 2D space."""
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].detach(), C[:, 1].detach(), s=200)
    
    for i in range(C.shape[0]):
        char = itos[i] if i in itos else "."
        plt.text(C[i, 0].item(), C[i, 1].item(), char, ha="center", va="center", color="white")
    
    plt.grid(True)
    plt.show()

# ---------------------- NAME GENERATION ----------------------

def generate_names(parameters, itos, block_size=3):
    C, W1, b1, W2, b2 = parameters
    g = torch.Generator().manual_seed(2147483647 + 10)
    
    num_names = int(input("Enter the number of names to generate: "))
    
    for _ in range(num_names):
        out = []
        context = [0] * block_size  # Start with empty context
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print("".join(itos[i] for i in out))

# ---------------------- MAIN EXECUTION ----------------------

if __name__ == "__main__":
    # Download & Load Data
    url = "https://raw.githubusercontent.com/MariaOlshanetska/NLP-CCiL-Greek-Project/main/names_greek.txt"
    download_data(url)
    words = load_data()

    # Build Mappings & Datasets
    stoi, itos = build_vocab(words)
    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    # Initialize Parameters
    parameters = init_params()

    # Train Model
    loss_log = train(Xtr, Ytr, parameters)

    # Evaluate Model
    print("Training Loss:", evaluate(Xtr, Ytr, parameters))
    print("Validation Loss:", evaluate(Xdev, Ydev, parameters))
    print("Test Loss:", evaluate(Xte, Yte, parameters))

    # Plot Embeddings
    plot_embedding(parameters[0], itos)

    # Generate Names
    generate_names(parameters, itos)
