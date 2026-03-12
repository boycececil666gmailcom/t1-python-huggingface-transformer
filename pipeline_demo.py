"""
pipeline_demo.py

Quick-start demonstrations using Hugging Face pipeline() API.
Covers: sentiment analysis, text generation, summarization,
        translation, question-answering, and zero-shot classification.
"""

from transformers import pipeline


# ── 1. Sentiment Analysis ────────────────────────────────────────────────────
def demo_sentiment():
    classifier = pipeline("sentiment-analysis")
    texts = [
        "Transformers make NLP incredibly accessible!",
        "I really dislike bugs in production code.",
    ]
    results = classifier(texts)
    print("\n── Sentiment Analysis ──")
    for text, res in zip(texts, results):
        print(f"  {text!r}  →  {res['label']} ({res['score']:.4f})")


# ── 2. Text Generation ───────────────────────────────────────────────────────
def demo_text_generation():
    generator = pipeline("text-generation", model="gpt2")
    prompt = "The field of artificial intelligence has recently"
    outputs = generator(prompt, max_new_tokens=60, num_return_sequences=2, truncation=True)
    print("\n── Text Generation ──")
    for i, out in enumerate(outputs, 1):
        print(f"  [{i}] {out['generated_text']}\n")


# ── 3. Summarization ─────────────────────────────────────────────────────────
def demo_summarization():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    article = (
        "Hugging Face is a company that builds tools for machine learning. "
        "It is best known for its Transformers library, which provides thousands "
        "of pretrained models for NLP, vision, and audio tasks. "
        "The library supports PyTorch, TensorFlow, and JAX, and the Hugging Face Hub "
        "hosts more than 300,000 public models contributed by the community."
    )
    summary = summarizer(article, max_length=60, min_length=20, do_sample=False)
    print("\n── Summarization ──")
    print(f"  {summary[0]['summary_text']}")


# ── 4. Translation ───────────────────────────────────────────────────────────
def demo_translation():
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    text = "Hugging Face Transformers is the best NLP library ever created."
    result = translator(text)
    print("\n── Translation (EN → FR) ──")
    print(f"  {result[0]['translation_text']}")


# ── 5. Question Answering ────────────────────────────────────────────────────
def demo_qa():
    qa = pipeline("question-answering")
    context = (
        "Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, "
        "and Thomas Wolf. The company is headquartered in New York City."
    )
    question = "When was Hugging Face founded?"
    result = qa(question=question, context=context)
    print("\n── Question Answering ──")
    print(f"  Q: {question}")
    print(f"  A: {result['answer']}  (score: {result['score']:.4f})")


# ── 6. Zero-Shot Classification ──────────────────────────────────────────────
def demo_zero_shot():
    classifier = pipeline("zero-shot-classification")
    text = "This tutorial covers neural network training with PyTorch."
    labels = ["sports", "technology", "politics", "cooking"]
    result = classifier(text, candidate_labels=labels)
    print("\n── Zero-Shot Classification ──")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label:<12} {score:.4f}")


if __name__ == "__main__":
    demo_sentiment()
    demo_text_generation()
    demo_summarization()
    demo_translation()
    demo_qa()
    demo_zero_shot()
