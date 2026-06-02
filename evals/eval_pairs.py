# 20 hardcoded Q&A pairs for RAG evaluation.
# These are written for a generic AI/ML paper (e.g. "Attention Is All You Need").
# Replace answers with ground-truth text from your actual ingested document.

EVAL_PAIRS = [
    {
        "id": "q01",
        "question": "What is the main architecture proposed in the paper?",
        "expected_keywords": ["transformer", "attention", "encoder", "decoder"],
    },
    {
        "id": "q02",
        "question": "What problem does self-attention solve compared to RNNs?",
        "expected_keywords": ["parallelization", "long-range", "sequential"],
    },
    {
        "id": "q03",
        "question": "How many attention heads are used in the base model?",
        "expected_keywords": ["8"],
    },
    {
        "id": "q04",
        "question": "What is the dimensionality of the model embeddings?",
        "expected_keywords": ["512"],
    },
    {
        "id": "q05",
        "question": "What optimizer is used during training?",
        "expected_keywords": ["adam"],
    },
    {
        "id": "q06",
        "question": "What dataset is used for English-to-German translation?",
        "expected_keywords": ["wmt", "2014"],
    },
    {
        "id": "q07",
        "question": "What BLEU score does the model achieve on English-to-German translation?",
        "expected_keywords": ["28", "bleu"],
    },
    {
        "id": "q08",
        "question": "What is dropout used for in the Transformer?",
        "expected_keywords": ["regularization", "overfitting"],
    },
    {
        "id": "q09",
        "question": "What is positional encoding and why is it needed?",
        "expected_keywords": ["position", "sequence", "order"],
    },
    {
        "id": "q10",
        "question": "How does multi-head attention differ from single-head attention?",
        "expected_keywords": ["multiple", "subspaces", "jointly"],
    },
    {
        "id": "q11",
        "question": "What is the feed-forward network dimension used in the model?",
        "expected_keywords": ["2048"],
    },
    {
        "id": "q12",
        "question": "What label smoothing value is used?",
        "expected_keywords": ["0.1"],
    },
    {
        "id": "q13",
        "question": "How many layers does the encoder have?",
        "expected_keywords": ["6"],
    },
    {
        "id": "q14",
        "question": "What is the training duration mentioned for the base model?",
        "expected_keywords": ["100000", "steps", "hours"],
    },
    {
        "id": "q15",
        "question": "What hardware is used for training?",
        "expected_keywords": ["gpu", "p100", "nvidia"],
    },
    {
        "id": "q16",
        "question": "What is the role of the decoder in the Transformer?",
        "expected_keywords": ["output", "autoregressive", "masked"],
    },
    {
        "id": "q17",
        "question": "What is the scaled dot-product attention formula?",
        "expected_keywords": ["softmax", "sqrt", "dk", "qk"],
    },
    {
        "id": "q18",
        "question": "What task does the big Transformer model set a new state of the art on?",
        "expected_keywords": ["translation", "bleu", "english"],
    },
    {
        "id": "q19",
        "question": "What is residual connection and where is it applied?",
        "expected_keywords": ["residual", "add", "layer norm"],
    },
    {
        "id": "q20",
        "question": "What future work do the authors suggest?",
        "expected_keywords": ["images", "audio", "video", "modalities"],
    },
]
