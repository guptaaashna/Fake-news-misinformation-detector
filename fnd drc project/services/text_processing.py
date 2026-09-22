"""NLTK-based processing and descriptive statistics for extracted articles."""

import re
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer


FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "will", "with",
}


def _sentences(text: str) -> list[str]:
    try:
        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except LookupError:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _tokens(text: str) -> list[str]:
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return re.findall(r"\b[\w]+(?:['-][\w]+)*\b|[^\w\s]", text)


def _stop_words() -> set[str]:
    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOP_WORDS


def _lemmatize(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    result = []
    for token in tokens:
        try:
            result.append(lemmatizer.lemmatize(token))
        except LookupError:
            result.append(token)
    return result


def process_text(text: str) -> dict:
    """Process extracted natural-language text without changing claim input."""
    cleaned_text = re.sub(r"\s+", " ", text or "").strip()
    sentences = _sentences(cleaned_text) if cleaned_text else []
    tokens = _tokens(cleaned_text) if cleaned_text else []
    stop_words = _stop_words()
    filtered_tokens = [
        token.lower()
        for token in tokens
        if token.isalpha() and token.lower() not in stop_words
    ]
    lemmatized_tokens = _lemmatize(filtered_tokens)
    word_tokens = [token.lower() for token in tokens if token.isalpha()]
    word_frequency = dict(Counter(lemmatized_tokens).most_common())
    word_count = len(word_tokens)
    return {
        "cleaned_text": cleaned_text,
        "sentences": sentences,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "statistics": {
            "character_count": len(cleaned_text),
            "word_count": word_count,
            "sentence_count": len(sentences),
            "unique_word_count": len(set(word_tokens)),
            "average_sentence_length": word_count / len(sentences) if sentences else 0.0,
        },
        "word_frequency": word_frequency,
    }
