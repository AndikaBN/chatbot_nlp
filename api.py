import csv
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model-indobert-ticketing"
DATA_PATH = BASE_DIR / "data" / "dataset_chatbot_it_support_3_kategori.csv"

SIMILARITY_THRESHOLD = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.60
KEYWORD_BOOST = 0.20

ADWARE_QUERY_KEYWORDS = ["iklan", "popup", "pop up", "adware", "malware"]
ADWARE_DATA_KEYWORDS = ["iklan", "adware", "malware"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    debug: bool = False


class TopMatch(BaseModel):
    label: str
    question: str
    score: float


class ChatResponse(BaseModel):
    success: bool
    reply: str
    intent: str | None = None
    confidence: float | None = None
    matched_question: str | None = None
    retrieval_label: str | None = None
    similarity: float | None = None
    top_matches: list[TopMatch] = []
    should_ask_resolution: bool = False


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()

    replacements = {
        "nggak": "tidak",
        "ngga": "tidak",
        "gak": "tidak",
        "ga": "tidak",
        "ndak": "tidak",
        "tdk": "tidak",
        "nyala": "menyala",
    }

    for source, target in replacements.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def make_dataset_row(
    question: str,
    label: str,
    response: str,
    augmented: bool = False,
) -> dict[str, Any]:
    normalized_question = normalize_text(question)
    normalized_response = normalize_text(response)

    return {
        "text": question,
        "label": label,
        "response": response,
        "normalized_text": normalized_question,
        "normalized_search_text": f"{normalized_question} {normalized_response}".strip(),
        "augmented": augmented,
    }


def find_adware_reference_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if contains_any_keyword(row["text"], ["iklan", "popup", "pop up", "adware"]):
            return row

    for row in rows:
        if contains_any_keyword(row["response"], ["iklan", "adware", "malware"]):
            return row

    return None


def find_reference_row(
    rows: list[dict[str, Any]],
    text_keywords: list[str],
    response_keywords: list[str] | None = None,
) -> dict[str, Any] | None:
    response_keywords = response_keywords or text_keywords

    for row in rows:
        if contains_any_keyword(row["text"], text_keywords):
            return row

    for row in rows:
        if contains_any_keyword(row["response"], response_keywords):
            return row

    return None


def append_augmented_questions(
    rows: list[dict[str, Any]],
    questions: list[str],
    label: str,
    response: str,
    existing_questions: set[str],
) -> None:
    for question in questions:
        normalized_question = normalize_text(question)

        if normalized_question in existing_questions:
            continue

        rows.append(
            make_dataset_row(
                question=question,
                label=label,
                response=response,
                augmented=True,
            )
        )
        existing_questions.add(normalized_question)


def add_augmented_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adware_row = find_adware_reference_row(rows)
    password_row = find_reference_row(rows, ["password", "reset", "lupa"])
    printer_install_row = find_reference_row(rows, ["printer", "driver"], ["printer", "driver", "terinstal"])
    existing_questions = {row["normalized_text"] for row in rows}

    if adware_row is not None:
        append_augmented_questions(
            rows,
            [
                "aplikasi banyak iklan",
                "iklan muncul terus di komputer",
                "banyak pop up iklan muncul sendiri",
                "komputer muncul iklan terus",
                "browser tidak dibuka tapi muncul iklan",
            ],
            "software",
            adware_row["response"],
            existing_questions,
        )

    if password_row is not None:
        append_augmented_questions(
            rows,
            [
                "bagaimana cara reset password",
                "cara reset password windows",
                "saya lupa password windows",
                "password akun windows lupa",
                "minta bantuan reset password",
            ],
            "software",
            password_row["response"],
            existing_questions,
        )

    if printer_install_row is not None:
        append_augmented_questions(
            rows,
            [
                "bagaimana instalasi printer",
                "cara install printer",
                "cara instal printer",
                "cara pasang driver printer",
                "printer belum terinstal",
            ],
            "hardware",
            printer_install_row["response"],
            existing_questions,
        )

    return rows


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"text", "label", "response"}

        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Dataset harus memiliki kolom: text, label, response")

        for row in reader:
            question = row["text"].strip()
            label = row["label"].strip()
            response = row["response"].strip()

            if not question or not label or not response:
                continue

            rows.append(make_dataset_row(question, label, response))

    if not rows:
        raise ValueError("Dataset kosong atau tidak memiliki baris valid.")

    return add_augmented_rows(rows)


dataset_rows = load_dataset(DATA_PATH)
dataset_questions = [row["normalized_text"] for row in dataset_rows]

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    lowercase=True,
)
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=True,
)

char_dataset_vectors = char_vectorizer.fit_transform(dataset_questions)
word_dataset_vectors = word_vectorizer.fit_transform(dataset_questions)

classifier = pipeline(
    "text-classification",
    model=str(MODEL_PATH),
    tokenizer=str(MODEL_PATH),
)

app = FastAPI(
    title="IT Support Chatbot API",
    version="1.0.0",
)


def predict_intent(text: str) -> tuple[str, float]:
    result = classifier(text)[0]
    return result["label"], float(result["score"])


def apply_keyword_boost(query: str, scores):
    if not contains_any_keyword(query, ADWARE_QUERY_KEYWORDS):
        return scores

    boosted_scores = scores.copy()

    for index, row in enumerate(dataset_rows):
        if contains_any_keyword(row["normalized_search_text"], ADWARE_DATA_KEYWORDS):
            boosted_scores[index] = min(1.0, boosted_scores[index] + KEYWORD_BOOST)

    return boosted_scores


def build_top_matches(scores, limit: int = 3) -> list[TopMatch]:
    top_indices = scores.argsort()[::-1][:limit]
    matches = []

    for index in top_indices:
        row = dataset_rows[int(index)]
        matches.append(
            TopMatch(
                label=row["label"],
                question=row["text"],
                score=float(scores[int(index)]),
            )
        )

    return matches


def find_best_response(text: str) -> dict[str, Any]:
    normalized_query = normalize_text(text)

    if not normalized_query:
        return {
            "best_text": None,
            "best_label": None,
            "best_response": None,
            "similarity_score": 0.0,
            "top_matches": [],
        }

    char_query_vector = char_vectorizer.transform([normalized_query])
    word_query_vector = word_vectorizer.transform([normalized_query])

    char_scores = cosine_similarity(char_query_vector, char_dataset_vectors).flatten()
    word_scores = cosine_similarity(word_query_vector, word_dataset_vectors).flatten()
    final_scores = (0.6 * char_scores) + (0.4 * word_scores)
    final_scores = apply_keyword_boost(normalized_query, final_scores)

    best_index = int(final_scores.argmax())
    best_row = dataset_rows[best_index]

    return {
        "best_text": best_row["text"],
        "best_label": best_row["label"],
        "best_response": best_row["response"],
        "similarity_score": float(final_scores[best_index]),
        "top_matches": build_top_matches(final_scores),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "success": True,
        "status": "ok",
        "dataset_rows": len(dataset_rows),
        "model_path": str(MODEL_PATH),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    intent, confidence = predict_intent(message)
    retrieval = find_best_response(message)
    similarity = retrieval["similarity_score"]

    if similarity >= SIMILARITY_THRESHOLD:
        return ChatResponse(
            success=True,
            reply=retrieval["best_response"],
            intent=intent,
            confidence=confidence,
            matched_question=retrieval["best_text"],
            retrieval_label=retrieval["best_label"],
            similarity=similarity,
            top_matches=retrieval["top_matches"],
            should_ask_resolution=True,
        )

    return ChatResponse(
        success=True,
        reply="Maaf, saya belum menemukan jawaban yang sesuai. Bisa jelaskan masalahnya lebih detail?",
        intent=intent,
        confidence=confidence,
        matched_question=retrieval["best_text"],
        retrieval_label=retrieval["best_label"],
        similarity=similarity,
        top_matches=retrieval["top_matches"] if request.debug else [],
        should_ask_resolution=False,
    )
