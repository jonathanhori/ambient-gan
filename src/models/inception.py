"""
inception.py — MNIST classifier for inception score computation
================================================================
Reproduces the inference network from Bora et al. 2018 (src/mnist/inf/).
Trained once on clean MNIST, then used frozen to evaluate GAN quality.

Architecture matches model_def.py exactly:
  Conv(5x5, 32) + ReLU + MaxPool
  Conv(5x5, 64) + ReLU + MaxPool
  FC(3136 → 1024) + ReLU + Dropout(0.5)
  FC(1024 → 10)

Usage:
  # Train
  python classifier.py --train --save_path ./outputs/classifier.pt

  # Evaluate IS for a trained generator
  python classifier.py --eval --classifier_path ./outputs/classifier.pt \
                               --generator_path  ./outputs/checkpoints/G_final.pt
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Generator, LATENT_DIM
from utils import sample_latent


# ---------------------------------------------------------------------------
# Classifier architecture
# ---------------------------------------------------------------------------

class MNISTClassifier(nn.Module):
    """
    Two-conv-layer CNN matching src/mnist/inf/model_def.py exactly.

    Forward pass shapes:
      Input:  (B, 1, 28, 28)  values in [0, 1]
      Output: (B, 10)         raw logits, no softmax
                              use F.softmax(logits, dim=1) for probabilities
    """

    def __init__(self):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Sequential(
            # Input is greyscale image (B, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            # Input is convolved image (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        # Fully connected block 1
        # Input is flattened image (B, 7*7*64=3136)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)

        # Fully connected block 2
        # Input is previous FC layer after dropout
        # Ouputs logits
        self.fc2 = nn.Linear(1024, 10)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) in [0, 1]
        Returns:
            logits: (B, 10) raw scores, no softmax
        """
        B = x.shape[0]
        # Two Conv + Pool layers 
        h = self.conv1(x)
        h = self.conv2(h)
        # Reshape for Linear
        h = h.view(B, -1)
        # Two fully connected layers
        h = self.fc1(h)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(save_path: str,
                     data_dir: str = './data',
                     batch_size: int = 64,
                     lr: float = 1e-4,
                     max_iters: int = 20000,
                     device: torch.device = None) -> MNISTClassifier:
    """
    Train the classifier on clean MNIST.
    Matches hparams in Bora src/mnist/inf/model_def.py:
      batch_size=64, lr=1e-4, max_train_iter=20000

    Args:
        save_path:  where to save the trained weights
        data_dir:   MNIST download directory
        batch_size: matches their batch_size=64
        lr:         matches their learning_rate=1e-4
        max_iters:  matches their max_train_iter=20000
        device:     torch device

    Returns:
        trained MNISTClassifier
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST normalised to [0, 1] — same as generator output range
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=2)

    model = MNISTClassifier().to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)

    # Loss - Cross entropy between logits and integer class labels
    criterion = nn.CrossEntropyLoss()

    model.train()
    data_iter = iter(dataloader)
    best_acc = 0.0

    for iteration in range(max_iters):

        # Refill iterator when exhausted
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass, loss, backward, step
        opt.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        # Log accuracy every 500 iterations
        if (iteration + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                model_preds = logits.argmax(dim = 1)
                acc = (model_preds == labels).float().mean().item()
            model.train()
            print(f"Iter {iteration+1}/{max_iters}  loss={loss.item():.4f}  acc={acc:.4f}")

    # Final test accuracy
    test_acc = evaluate_classifier(model, data_dir, device)
    print(f"Final test accuracy: {test_acc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved classifier to {save_path}")

    return model


def evaluate_classifier(model: MNISTClassifier,
                         data_dir: str = './data',
                         device: torch.device = None) -> float:
    """
    Compute accuracy on the MNIST test set.

    Args:
        model:    trained MNISTClassifier
        data_dir: MNIST data directory
        device:   torch device

    Returns:
        accuracy: float in [0, 1]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            test_preds = logits.argmax(dim = 1)
            
            correct += (test_preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total


# ---------------------------------------------------------------------------
# Inception score
# ---------------------------------------------------------------------------


def _calc_score():
            # Translate their get_inception_score directly:
        #
        #
        #   p_y = np.mean(p_y_given_x, axis=0)               # marginal (10,)
        #   terms = p_y_given_x * (np.log(p_y_given_x) - np.log(p_y))
        #   kl_div = np.mean(np.sum(terms, axis=1))
        #   score = np.exp(kl_div)
    return

def compute_inception_score(generator: nn.Module,
                             classifier: MNISTClassifier,
                             n_rounds: int = 16,
                             n_samples_per_round: int = 5000,
                             batch_size: int = 64,
                             latent_dim: int = LATENT_DIM,
                             device: torch.device = None) -> tuple:
    """
    Compute inception score by running n_rounds of sampling and averaging.
    Matches paper's save_inception_data which runs 16 rounds.
    
    IS = exp{ E_x [ KL(p(y|x) || p(y)) ] }
    The final IS returned is the average of n_rounds of IS
    Each round's IS's expectation is taken over n_samples_per_round of samples from the model

    Args:
        generator:          trained Generator, already in eval mode
        classifier:         trained MNISTClassifier
        n_rounds:           number of rounds to average over (they use 16)
        n_samples_per_round: samples per round (they use ~batch_size * num_batches)
        batch_size:         generation batch size
        latent_dim:         generator latent dimension
        device:             torch device

    Returns:
        (mean_IS, std_IS): mean and std of IS across rounds
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier.eval()
    generator.eval()

    round_scores = []

    for _ in range(n_rounds):

        # --- generate n_samples_per_round images ---
        all_probs = []
        n_batches = n_samples_per_round // batch_size

        with torch.no_grad():
            for _ in range(n_batches):
                # Generate #batch_size images
                z = sample_latent(batch_size, latent_dim, device)
                gen_images = generator(z)

                # Run through classifier to get probabilities [ P(0| X), ..., P(9| X) ]
                logits = classifier(gen_images)
                label_probs = F.softmax(logits, dim = 1)
                all_probs.append(label_probs.cpu().numpy())

        # Compute inception score for this round
        p_y_given_x = np.concatenate(all_probs, axis=0)  # (n_samples_per_round, 10)
        score = _calc_score(p_y_given_x)
        round_scores.append(score)
        
    return np.mean(round_scores), np.std(round_scores)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',            action='store_true')
    parser.add_argument('--eval',             action='store_true')
    parser.add_argument('--save_path',        default='./outputs/inception.pt')
    parser.add_argument('--classifier_path',  default='./outputs/inception.pt')
    parser.add_argument('--generator_path',   default=None)
    parser.add_argument('--data_dir',         default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        train_classifier(
            save_path=args.save_path,
            data_dir=args.data_dir,
            device=device
        )

    if args.eval:
        assert args.generator_path is not None, "Provide --generator_path"

        # Load classifier
        classifier = MNISTClassifier().to(device)
        classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))

        # Load generator
        G = Generator(latent_dim=LATENT_DIM).to(device)
        G.load_state_dict(torch.load(args.generator_path, map_location=device))

        is_mean, is_std = compute_inception_score(G, classifier, device=device)
        print(f"Inception score: {is_mean:.3f} ± {is_std:.3f}")
        print("Their reported score (fully observed): 8.99")