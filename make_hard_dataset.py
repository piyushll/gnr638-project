"""
make_hard_dataset.py — render 50 hard deep-learning MCQs as PNG pages,
matching the visual style of the official sample (serif body, mathtext,
"Options" header, A/B/C/D, page number footer).

Output layout (mirrors the grader's expected test directory):

    dataset_hard/
      images/image_1.png ... image_50.png
      test.csv               # image_name column only (matches sample shape)
      sample_submission.csv  # id,image_name,option   (dummy: option=5)
      answers.csv            # id,image_name,option   (ground truth)

Run:
    python make_hard_dataset.py

Then evaluate end-to-end:
    python inference.py --test_dir <abs_path>/dataset_hard
    python score.py    # compares submission.csv against answers.csv
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "dataset_hard"
IMG_DIR = OUT_DIR / "images"


# ---------- 50 questions --------------------------------------------------
# Schema:
#   topic        : title shown after "Ques: "
#   body         : list[str]; mathtext via $...$
#   options      : list[str] of length 4; \n inside is honored
#   code_options : True  -> render options in a monospace box (code)
#                  False -> render as inline serif
#   answer       : int in {1,2,3,4}  (1=A, 2=B, 3=C, 4=D)
QUESTIONS: list[dict] = [
    # ---- CNN shape arithmetic -------------------------------------------
    {
        "topic": "Conv + Pool Output Shape",
        "body": [
            r"An input of size $224 \times 224$ is passed through:",
            r"  $\bullet$ Conv2d, kernel $5\times 5$, stride 1, padding 0",
            r"  $\bullet$ MaxPool2d, kernel size 4 (default stride = kernel)",
            "",
            "What is the final spatial size?",
        ],
        "options": [r"$54 \times 54$", r"$55 \times 55$", r"$56 \times 56$", r"$110 \times 110$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "ResNet Stem Arithmetic",
        "body": [
            r"On a $224 \times 224$ input, three layers are applied in order:",
            r"  $\bullet$ Conv2d $7\times 7$, stride 2, padding 3",
            r"  $\bullet$ Conv2d $3\times 3$, stride 2, padding 1",
            r"  $\bullet$ Conv2d $3\times 3$, stride 1, padding 1",
            "",
            "What is the final spatial size?",
        ],
        "options": [r"$28 \times 28$", r"$112 \times 112$", r"$56 \times 56$", r"$32 \times 32$"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "ConvTranspose Output",
        "body": [
            r"An $8 \times 8$ feature map is passed through",
            r"$\mathrm{ConvTranspose2d}$ with kernel $4\times 4$, stride 2, padding 1.",
            "",
            "What is the output spatial size?",
        ],
        "options": [r"$14 \times 14$", r"$15 \times 15$", r"$16 \times 16$", r"$17 \times 17$"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "Dilated Convolution",
        "body": [
            r"A $3\times 3$ convolution with dilation 2, stride 1, padding 0 is",
            r"applied to a $32 \times 32$ feature map.",
            "",
            "What is the output spatial size?",
        ],
        "options": [r"$30 \times 30$", r"$28 \times 28$", r"$26 \times 26$", r"$32 \times 32$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Conv Parameter Count",
        "body": [
            r"How many trainable parameters does",
            r"$\mathrm{nn.Conv2d}(512, 256, \mathrm{kernel\_size}=1, \mathrm{bias}=\mathrm{True})$ have?",
        ],
        "options": ["131,072", "131,328", "131,584", "262,400"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Receptive Field",
        "body": [
            r"What is the receptive field of three stacked $3\times 3$ convolutions",
            "with stride 1 and no dilation?",
        ],
        "options": [r"$5 \times 5$", r"$7 \times 7$", r"$9 \times 9$", r"$11 \times 11$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "AdaptiveAvgPool Output",
        "body": [
            r"An input tensor of shape $(B, 512, 28, 28)$ is fed to",
            r"$\mathrm{nn.AdaptiveAvgPool2d}(7)$. What is the output shape?",
        ],
        "options": [
            r"$(B, 7, 28, 28)$",
            r"$(B, 512, 28, 28)$",
            r"$(B, 7, 7, 512)$",
            r"$(B, 512, 7, 7)$",
        ],
        "code_options": False,
        "answer": 4,
    },
    {
        "topic": "Padding 'same' Behavior",
        "body": [
            r"$\mathrm{nn.Conv2d}(64, 64, 3, \mathrm{stride}=1, \mathrm{padding}=$'same'$)$ is",
            r"applied to a $(1, 64, 32, 32)$ input.",
            "",
            "What is the output spatial size?",
        ],
        "options": [r"$30 \times 30$", r"$32 \times 32$", r"$34 \times 34$", r"$16 \times 16$"],
        "code_options": False,
        "answer": 2,
    },
    # ---- Transformer / attention ----------------------------------------
    {
        "topic": "Multi-Head Attention: per-head dim",
        "body": [
            r"A multi-head attention block uses $d_{\mathrm{model}}=512$ and 8 heads.",
            r"What is $d_k$ (the per-head key dimension)?",
        ],
        "options": ["32", "64", "128", "512"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "MHA Parameter Count",
        "body": [
            r"How many parameters do the four projection matrices ($W_Q, W_K, W_V, W_O$)",
            r"in a multi-head attention layer have, with $d_{\mathrm{model}}=768$, no bias?",
        ],
        "options": ["589,824", "1,179,648", "2,359,296", "4,718,592"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "KV Cache Size",
        "body": [
            r"For one transformer decoder layer with sequence length 2048,",
            r"$d_{\mathrm{model}}=4096$, fp16, what is the KV cache size?",
        ],
        "options": ["8 MiB", "16 MiB", "32 MiB", "64 MiB"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "Attention Trainable Params",
        "body": [
            "Scaled dot-product attention is defined as",
            r"$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}(QK^\top / \sqrt{d_k})V$.",
            "",
            "How many trainable parameters does this op have?",
        ],
        "options": [r"$d_k$", r"$3 d_k$", r"$d_k^2$", "0"],
        "code_options": False,
        "answer": 4,
    },
    {
        "topic": "Rotary Position Embeddings",
        "body": [
            "Rotary positional embeddings (RoPE) are typically applied to which",
            "components of multi-head attention?",
        ],
        "options": [
            "Only the query Q",
            "Q and K, but not V",
            "Q, K, and V",
            "Only the value V",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Causal Mask Density",
        "body": [
            r"A causal (lower-triangular) attention mask for sequence length 4 allows",
            "queries to attend to keys at positions $\\leq$ their own.",
            "",
            "How many (query, key) pairs are unmasked?",
        ],
        "options": ["6", "8", "10", "16"],
        "code_options": False,
        "answer": 3,
    },
    # ---- PyTorch code tracing -------------------------------------------
    {
        "topic": "torch.cat Shape",
        "body": [
            "What is the shape of the output of",
            r"$\mathrm{torch.cat}([\mathrm{torch.zeros}(2,3), \mathrm{torch.ones}(2,3)], \mathrm{dim}=0)$?",
        ],
        "options": [r"$(2, 6)$", r"$(4, 3)$", r"$(2, 3, 2)$", r"$(6, 2)$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "torch.stack Shape",
        "body": [
            "What is the shape of the output of",
            r"$\mathrm{torch.stack}([\mathrm{torch.zeros}(3), \mathrm{torch.ones}(3)], \mathrm{dim}=1)$?",
        ],
        "options": [r"$(2, 3)$", r"$(3, 2)$", r"$(6,)$", r"$(3, 3)$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Linear Parameter Count",
        "body": [
            r"How many trainable parameters does $\mathrm{nn.Linear}(20, 30)$ have?",
        ],
        "options": ["600", "620", "630", "650"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "Backward of x**3",
        "body": [
            "Consider:",
            "",
            "    x = torch.tensor(2.0, requires_grad=True)",
            "    y = x ** 3",
            "    y.backward()",
            "",
            "What is x.grad after backward?",
        ],
        "options": ["3.0", "6.0", "8.0", "12.0"],
        "code_options": False,
        "answer": 4,
    },
    {
        "topic": "Broadcasting Shapes",
        "body": [
            "Two tensors with shapes $(3, 1, 5)$ and $(1, 4, 1)$ are added.",
            "What is the resulting shape?",
        ],
        "options": [r"$(3, 4, 5)$", r"$(1, 1, 1)$", r"$(3, 4, 1)$", "broadcast error"],
        "code_options": False,
        "answer": 1,
    },
    {
        "topic": "LayerNorm Normalization Axis",
        "body": [
            r"$\mathrm{nn.LayerNorm}(64)$ is applied to a tensor of shape $(B, T, 64)$.",
            "",
            "Over which dimension(s) is the mean and variance computed?",
        ],
        "options": [
            r"Dim 0 (batch B)",
            r"Dim 1 (sequence T)",
            r"Dim 2 (the last dim, size 64)",
            "All three dims jointly",
        ],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "Dropout Scaling",
        "body": [
            r"During training, $\mathrm{nn.Dropout}(p=0.5)$ scales the surviving",
            r"activations by what factor (so the expected value matches eval)?",
        ],
        "options": [r"$0.5$", r"$1.0$", r"$1.5$", r"$2.0$"],
        "code_options": False,
        "answer": 4,
    },
    {
        "topic": "torch.argmax Result",
        "body": [
            "What does the following return?",
            "",
            "    torch.argmax(torch.tensor([[1, 3, 2],",
            "                               [5, 0, 4]]), dim=1)",
        ],
        "options": [
            "tensor([0, 1])",
            "tensor([1, 0])",
            "tensor([3, 5])",
            "tensor([1, 1])",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "CrossEntropyLoss Targets",
        "body": [
            r"$\mathrm{nn.CrossEntropyLoss}$ in PyTorch (default) expects the target",
            "tensor to be in which form?",
        ],
        "options": [
            r"One-hot float tensor of shape $(N, C)$",
            r"Class indices as a long tensor of shape $(N,)$",
            r"Log-probabilities of shape $(N, C)$",
            r"Probabilities of shape $(N, C)$ summing to 1",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "MaxPool2d Output",
        "body": [
            r"$\mathrm{nn.MaxPool2d}(\mathrm{kernel\_size}=2)$ is applied to a tensor",
            r"of shape $(1, 16, 31, 31)$. What is the output spatial size?",
        ],
        "options": [r"$15 \times 15$", r"$16 \times 16$", r"$14 \times 14$", r"$31 \times 31$"],
        "code_options": False,
        "answer": 1,
    },
    # ---- Backprop / gradients -------------------------------------------
    {
        "topic": "Sigmoid Derivative",
        "body": [
            r"What is the derivative of $\sigma(x) = 1/(1+e^{-x})$ evaluated at $x=0$?",
        ],
        "options": ["0", "0.25", "0.5", "1"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "tanh Derivative at Zero",
        "body": [
            r"What is $\frac{d}{dx} \tanh(x)$ evaluated at $x=0$?",
        ],
        "options": ["0", "0.5", "1", "2"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "ReLU Derivative",
        "body": [
            r"What is $\frac{d}{dx} \mathrm{ReLU}(x)$ at $x = -1$?",
        ],
        "options": ["-1", "0", "1", "undefined"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Cross-Entropy Gradient",
        "body": [
            r"For softmax cross-entropy with logits $z$, predicted probabilities $p$,",
            r"and one-hot label $y$, what is $\partial L / \partial z_i$?",
        ],
        "options": [r"$p_i \cdot y_i$", r"$p_i - y_i$", r"$y_i - p_i$", r"$\log p_i - y_i$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Swish Derivative at Zero",
        "body": [
            r"Let $f(x) = x \cdot \sigma(x)$ (Swish/SiLU).",
            r"What is $f'(0)$?",
        ],
        "options": ["0", "0.25", "0.5", "1"],
        "code_options": False,
        "answer": 3,
    },
    # ---- Optimizers -----------------------------------------------------
    {
        "topic": "Adam Default Hyperparameters",
        "body": [
            r"What are the default values of $(\beta_1, \beta_2)$ in PyTorch's",
            r"$\mathrm{torch.optim.Adam}$?",
        ],
        "options": [
            r"$(0.5, 0.9)$",
            r"$(0.9, 0.99)$",
            r"$(0.9, 0.999)$",
            r"$(0.99, 0.999)$",
        ],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "Momentum Effective LR",
        "body": [
            r"Plain SGD with momentum coefficient $\mu = 0.9$ amplifies the",
            r"asymptotic effective step size relative to vanilla SGD by what factor?",
        ],
        "options": [r"$0.9 \times$", r"$1.1 \times$", r"$2 \times$", r"$10 \times$"],
        "code_options": False,
        "answer": 4,
    },
    {
        "topic": "AdamW vs Adam",
        "body": [
            "What is the principal difference between AdamW and Adam?",
        ],
        "options": [
            r"AdamW uses a different learning-rate schedule",
            r"AdamW decouples weight decay from the gradient update",
            r"AdamW uses second-order Hessian information",
            r"AdamW removes the bias-correction terms",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Weight Decay in SGD",
        "body": [
            r"In PyTorch's SGD, weight_decay coefficient $\lambda$ modifies the gradient",
            r"that is applied to parameter $w$ by adding which term?",
        ],
        "options": [
            r"$\lambda \cdot w$",
            r"$\lambda \cdot w^2$",
            r"$2\lambda \cdot w$",
            r"$\lambda \cdot \mathrm{sign}(w)$",
        ],
        "code_options": False,
        "answer": 1,
    },
    # ---- Loss functions -------------------------------------------------
    {
        "topic": "Convexity of BCE-with-Logits",
        "body": [
            "Binary cross-entropy with logits (sigmoid then negative log-likelihood)",
            r"is, as a function of the logit $z$:",
        ],
        "options": [
            "Convex",
            "Concave",
            "Neither convex nor concave",
            "Linear",
        ],
        "code_options": False,
        "answer": 1,
    },
    {
        "topic": "Focal Loss Modulation",
        "body": [
            r"Focal loss multiplies cross-entropy by a factor that down-weights",
            r"easy examples. With $p_t$ the predicted probability of the true class",
            r"and $\gamma \geq 0$, this factor is:",
        ],
        "options": [r"$p_t^\gamma$", r"$(1 - p_t)^\gamma$", r"$\gamma \cdot p_t$", r"$\log p_t / \gamma$"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Label Smoothing Target",
        "body": [
            r"With label smoothing $\varepsilon = 0.1$ over $K=10$ classes, what is",
            r"the smoothed target probability of the true class?",
        ],
        "options": ["0.81", "0.90", "0.91", "1.00"],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "Triplet Loss Constraint",
        "body": [
            r"Triplet loss with margin $\alpha$ on anchor $a$, positive $p$, negative $n$",
            r"is $\max(0,\ d(a,p) - d(a,n) + \alpha)$. The loss is zero iff:",
        ],
        "options": [
            r"$d(a,p) + \alpha \leq d(a,n)$",
            r"$d(a,n) + \alpha \leq d(a,p)$",
            r"$d(a,p) = d(a,n)$",
            r"$d(a,p) \leq \alpha$",
        ],
        "code_options": False,
        "answer": 1,
    },
    # ---- Activations ----------------------------------------------------
    {
        "topic": "SiLU / Swish Definition",
        "body": [
            r"The SiLU (a.k.a. Swish) activation is defined as:",
        ],
        "options": [
            r"$\mathrm{SiLU}(x) = \max(0, x)$",
            r"$\mathrm{SiLU}(x) = x \cdot \sigma(x)$",
            r"$\mathrm{SiLU}(x) = \tanh(x)$",
            r"$\mathrm{SiLU}(x) = \log(1 + e^x)$",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Leaky ReLU Output",
        "body": [
            r"What is the output of $\mathrm{LeakyReLU}(\alpha=0.01)$ at $x = -1$?",
        ],
        "options": ["-0.01", "0", "-1", "0.01"],
        "code_options": False,
        "answer": 1,
    },
    {
        "topic": "GELU at Zero",
        "body": [
            r"What is the value of $\mathrm{GELU}(x)$ at $x = 0$?",
        ],
        "options": ["-0.5", "0", "0.5", "1"],
        "code_options": False,
        "answer": 2,
    },
    # ---- BatchNorm ------------------------------------------------------
    {
        "topic": "BatchNorm Trainable Params",
        "body": [
            "How many trainable parameters does BatchNorm introduce per feature",
            r"channel (with $\mathrm{affine}=\mathrm{True}$)?",
        ],
        "options": ["1", "2", "3", "4"],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "BatchNorm at Eval Time",
        "body": [
            r"In evaluation mode, BatchNorm uses which statistics to normalize?",
        ],
        "options": [
            "The current batch's mean and variance",
            "The running mean and running variance",
            "Layer-norm-style statistics over features",
            "No normalization is applied at eval time",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Original BatchNorm Placement",
        "body": [
            "In the original BatchNorm paper (Ioffe & Szegedy, 2015), where is",
            "BatchNorm placed relative to the activation function?",
        ],
        "options": [
            "Before the linear / convolution layer",
            "After the activation function",
            "Between the linear and the activation",
            "Only at the output layer",
        ],
        "code_options": False,
        "answer": 3,
    },
    # ---- Initialization -------------------------------------------------
    {
        "topic": "Xavier Initialization Variance",
        "body": [
            r"Xavier (Glorot) initialization sets the weight variance to:",
        ],
        "options": [
            r"$1 / \mathrm{fan\_in}$",
            r"$2 / \mathrm{fan\_in}$",
            r"$2 / (\mathrm{fan\_in} + \mathrm{fan\_out})$",
            r"$1 / (\mathrm{fan\_in} \cdot \mathrm{fan\_out})$",
        ],
        "code_options": False,
        "answer": 3,
    },
    {
        "topic": "He / Kaiming Initialization",
        "body": [
            r"He (Kaiming) initialization sets the weight variance to:",
        ],
        "options": [
            r"$1 / \mathrm{fan\_in}$",
            r"$2 / \mathrm{fan\_in}$",
            r"$2 / (\mathrm{fan\_in} + \mathrm{fan\_out})$",
            r"$\sqrt{6 / \mathrm{fan\_in}}$",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Zero Initialization Failure",
        "body": [
            "Why does initializing all weights of an MLP to zero fail to train?",
        ],
        "options": [
            r"Gradients are exactly zero everywhere",
            r"The loss is undefined when weights are zero",
            r"All neurons in a layer compute identical outputs and gradients (no symmetry breaking)",
            r"PyTorch raises an exception on zero weights",
        ],
        "code_options": False,
        "answer": 3,
    },
    # ---- Regularization -------------------------------------------------
    {
        "topic": "Dropout Probability Interpretation",
        "body": [
            r"In $\mathrm{nn.Dropout}(p=0.2)$, the parameter $p$ is the probability of:",
        ],
        "options": [
            "Keeping a unit active",
            "Zeroing a unit",
            "Saturating a unit",
            "Re-initializing a unit",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "Mixup Augmentation",
        "body": [
            r"Mixup constructs a training example $(\tilde{x}, \tilde{y})$ from two",
            r"examples $(x_i, y_i)$ and $(x_j, y_j)$ as:",
        ],
        "options": [
            r"$\tilde{x} = x_i \odot x_j$, $\tilde{y} = y_i$",
            r"$\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$",
            r"$\tilde{x} = x_i + \mathrm{noise}$, $\tilde{y} = y_i$",
            r"$\tilde{x} = x_i$, $\tilde{y} = $ random class",
        ],
        "code_options": False,
        "answer": 2,
    },
    {
        "topic": "L1 vs L2 Regularization",
        "body": [
            r"Which statement about L1 vs L2 regularization is TRUE?",
        ],
        "options": [
            r"L2 tends to produce sparse weights; L1 does not",
            r"L1 tends to produce sparse weights; L2 shrinks toward zero but rarely exactly zero",
            r"L1 and L2 are mathematically equivalent",
            r"L1 has a closed-form gradient only when $w > 0$",
        ],
        "code_options": False,
        "answer": 2,
    },
    # ---- Architecture trivia --------------------------------------------
    {
        "topic": "Vision Transformer Patch Count",
        "body": [
            r"A standard ViT-B/16 splits a $224 \times 224$ image into",
            r"non-overlapping $16 \times 16$ patches.",
            "",
            "How many patch tokens are produced (excluding the CLS token)?",
        ],
        "options": ["49", "144", "196", "256"],
        "code_options": False,
        "answer": 3,
    },
]


# ---------- rendering ----------------------------------------------------

PAGE_W_IN = 8.5
PAGE_H_IN = 11.0
DPI = 150
LEFT_MARGIN = 0.10
RIGHT_MARGIN = 0.95
TOP = 0.95
BOTTOM = 0.05

TITLE_SIZE = 22
HEADER_SIZE = 16
BODY_SIZE = 13
OPTION_SIZE = 13
CODE_SIZE = 11
PAGE_NUM_SIZE = 11

LINE_HEIGHT = 0.030      # in axes-fraction units (used for body text)
TITLE_GAP = 0.055
SECTION_GAP = 0.030
OPTION_GAP = 0.022
CODE_LINE_HEIGHT = 0.022


def _draw_text_line(ax, x: float, y: float, text: str, *, size: int, weight: str = "normal", family: str = "serif") -> None:
    ax.text(x, y, text, fontsize=size, weight=weight, family=family,
            transform=ax.transAxes, verticalalignment="top")


def _draw_code_block(ax, x: float, y: float, code: str, *, width: float) -> float:
    """Render a multi-line code block in a light grey box. Returns the new y."""
    lines = code.splitlines() or [code]
    n = max(1, len(lines))
    pad_v = 0.008
    pad_h = 0.010
    height = n * CODE_LINE_HEIGHT + 2 * pad_v
    rect = patches.Rectangle(
        (x, y - height),
        width,
        height,
        transform=ax.transAxes,
        facecolor="#f3f3f3",
        edgecolor="#888888",
        linewidth=0.6,
    )
    ax.add_patch(rect)
    text_y = y - pad_v
    for line in lines:
        ax.text(
            x + pad_h, text_y, line,
            fontsize=CODE_SIZE, family="monospace",
            transform=ax.transAxes, verticalalignment="top",
        )
        text_y -= CODE_LINE_HEIGHT
    return y - height - 0.012


def render_question(q: dict, out_path: Path, page_num: int) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = LEFT_MARGIN
    width = RIGHT_MARGIN - LEFT_MARGIN
    y = TOP

    # Title
    _draw_text_line(ax, x, y, f"Ques: {q['topic']}", size=TITLE_SIZE, weight="bold")
    y -= TITLE_GAP

    # Body
    for line in q["body"]:
        if line == "":
            y -= LINE_HEIGHT * 0.6
            continue
        # Light wrapping for very long lines.
        wrapped = textwrap.wrap(line, width=92, break_long_words=False, break_on_hyphens=False) or [""]
        for w in wrapped:
            _draw_text_line(ax, x, y, w, size=BODY_SIZE)
            y -= LINE_HEIGHT

    y -= SECTION_GAP

    # Options header
    _draw_text_line(ax, x, y, "Options", size=HEADER_SIZE, weight="bold")
    y -= 0.04

    # Options A B C D
    for i, opt in enumerate(q["options"]):
        letter = "ABCD"[i]
        if q.get("code_options"):
            _draw_text_line(ax, x + 0.005, y, f"{letter}.", size=OPTION_SIZE)
            y = _draw_code_block(ax, x + 0.030, y + 0.005, opt, width=width - 0.030)
        else:
            # If option contains code-ish content with newlines, treat as code anyway.
            if "\n" in opt:
                _draw_text_line(ax, x + 0.005, y, f"{letter}.", size=OPTION_SIZE)
                y = _draw_code_block(ax, x + 0.030, y + 0.005, opt, width=width - 0.030)
            else:
                _draw_text_line(ax, x + 0.020, y, f"{letter}. {opt}", size=OPTION_SIZE)
                y -= LINE_HEIGHT + OPTION_GAP * 0.4

    # Footer page number
    ax.text(0.5, 0.025, str(page_num), ha="center", fontsize=PAGE_NUM_SIZE,
            family="serif", transform=ax.transAxes)

    fig.savefig(out_path, dpi=DPI, format="png")
    plt.close(fig)


def _balance_answer_positions(questions: list[dict], seed: int = 17) -> None:
    """Mutate questions in place so the correct answer is uniformly distributed
    over A/B/C/D (12-13 each across 50 questions). For each question we swap the
    correct option into the target position and re-index q['answer'] to match.
    A `B`-only guesser would otherwise score ~35% on this set.
    """
    import random
    n = len(questions)
    base = ([1, 2, 3, 4] * ((n // 4) + 1))[:n]  # 12-13 of each letter
    rng = random.Random(seed)
    rng.shuffle(base)
    for q, target in zip(questions, base):
        cur = q["answer"]
        if cur != target:
            opts = q["options"]
            opts[cur - 1], opts[target - 1] = opts[target - 1], opts[cur - 1]
            q["answer"] = target


def main() -> int:
    assert len(QUESTIONS) == 50, f"expected 50 questions, got {len(QUESTIONS)}"
    _balance_answer_positions(QUESTIONS)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    test_rows = []
    sample_rows = []
    answer_rows = []

    for i, q in enumerate(QUESTIONS, start=1):
        name = f"image_{i}"
        out_path = IMG_DIR / f"{name}.png"
        render_question(q, out_path, page_num=i)
        print(f"[gen] {out_path.name}  topic={q['topic']!r}  answer={q['answer']}")
        test_rows.append({"image_name": name})
        sample_rows.append({"id": name, "image_name": name, "option": 5})
        answer_rows.append({"id": name, "image_name": name, "option": q["answer"]})

    with (OUT_DIR / "test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name"])
        w.writeheader()
        w.writerows(test_rows)

    with (OUT_DIR / "sample_submission.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_name", "option"])
        w.writeheader()
        w.writerows(sample_rows)

    with (OUT_DIR / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_name", "option"])
        w.writeheader()
        w.writerows(answer_rows)

    print(f"[gen] wrote 50 images + test.csv + sample_submission.csv + answers.csv to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
