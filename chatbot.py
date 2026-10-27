import csv
import re
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


MODEL_PATH = "model-indobert-ticketing"
DATA_PATH = "data/dataset_chatbot_it_support_3_kategori.csv"

SIMILARITY_THRESHOLD = 0.25
NEW_PROBLEM_THRESHOLD = 0.30
LOW_CONFIDENCE_THRESHOLD = 0.60
KEYWORD_BOOST = 0.20

TICKET_DETAIL_FORMAT_REPLY = (
    "Format data belum lengkap. Mohon isi dengan format: "
    "Nama, Perusahaan/Divisi, Lokasi, Detail masalah."
)

ADWARE_QUERY_KEYWORDS = ["iklan", "popup", "pop up", "adware", "malware"]
ADWARE_DATA_KEYWORDS = ["iklan", "adware", "malware"]
BAD_DETAIL_WORDS = {"tai", "kntl", "babi", "anjing", "bangsat", "qwerty", "asdf"}

pending_problem = None
pending_ticket = None
debug_enabled = True


def normalize_text(text):
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


def contains_any_keyword(text, keywords):
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def make_dataset_row(question, label, response, augmented=False):
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


def find_adware_reference_row(rows):
    for row in rows:
        if contains_any_keyword(row["text"], ["iklan", "popup", "pop up", "adware"]):
            return row

    for row in rows:
        if contains_any_keyword(row["response"], ["iklan", "adware", "malware"]):
            return row

    return None


def find_reference_row(rows, text_keywords, response_keywords=None):
    response_keywords = response_keywords or text_keywords

    for row in rows:
        if contains_any_keyword(row["text"], text_keywords):
            return row

    for row in rows:
        if contains_any_keyword(row["response"], response_keywords):
            return row

    return None


def append_augmented_questions(rows, questions, label, response, existing_questions):
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


def add_augmented_rows(rows):
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


def load_dataset(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        header = file.readline()
        file.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","

        reader = csv.DictReader(file, delimiter=delimiter, skipinitialspace=True)
        if reader.fieldnames is not None:
            reader.fieldnames = [column.strip().lower() for column in reader.fieldnames]
        required_columns = {"text", "label", "response"}

        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Dataset {path} harus memiliki kolom: text, label, response. "
                f"Kolom yang terbaca: {reader.fieldnames!r}"
            )

        for row in reader:
            question = (row["text"] or "").strip()
            label = (row["label"] or "").strip()
            response = (row["response"] or "").strip()

            if not question or not label or not response:
                continue

            rows.append(
                make_dataset_row(
                    question=question,
                    label=label,
                    response=response,
                )
            )

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
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)


def predict_intent(text):
    result = classifier(text)[0]
    label = result["label"]
    score = float(result["score"])
    return label, score


def apply_keyword_boost(query, scores):
    if not contains_any_keyword(query, ADWARE_QUERY_KEYWORDS):
        return scores

    boosted_scores = scores.copy()

    for index, row in enumerate(dataset_rows):
        if contains_any_keyword(row["normalized_search_text"], ADWARE_DATA_KEYWORDS):
            boosted_scores[index] = min(1.0, boosted_scores[index] + KEYWORD_BOOST)

    return boosted_scores


def build_top_matches(scores, limit=3):
    top_indices = scores.argsort()[::-1][:limit]
    top_matches = []

    for index in top_indices:
        row = dataset_rows[int(index)]
        top_matches.append(
            {
                "text": row["text"],
                "label": row["label"],
                "score": float(scores[int(index)]),
            }
        )

    return top_matches


def find_best_response(text):
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


def is_solved_message(text):
    text = normalize_text(text)

    solved_keywords = {
        "sudah",
        "sudah bisa",
        "berhasil",
        "aman",
        "selesai",
        "done",
        "teratasi",
        "sudah teratasi",
    }

    return text in solved_keywords


def is_unsolved_message(text):
    text = normalize_text(text)

    unsolved_keywords = {
        "belum",
        "masih",
        "masih error",
        "tidak bisa",
        "belum bisa",
        "tetap error",
        "masih bermasalah",
        "buat tiket",
        "buat ticket",
        "ticket",
        "tiket",
    }

    return text in unsolved_keywords


def is_reset_message(text):
    text = normalize_text(text)
    return text in {"reset", "batal semua", "mulai ulang"}


def is_debug_off_message(text):
    return normalize_text(text) == "debug off"


def is_debug_on_message(text):
    return normalize_text(text) == "debug on"


def is_cancel_message(text):
    text = normalize_text(text)
    return text in {"batal", "cancel", "tidak jadi"}


def is_confirmation_message(text):
    text = normalize_text(text)
    return text in {"ya", "iya", "yes", "buat", "buat tiket", "lanjut"}


def validate_ticket_detail(text):
    normalized = normalize_text(text)
    words = set(normalized.split())

    if words & BAD_DETAIL_WORDS:
        return False

    parts = [part.strip() for part in text.split(",") if part.strip()]

    if len(parts) < 4:
        return False

    problem_detail = parts[-1]

    if len(problem_detail) < 10:
        return False

    if len(normalize_text(problem_detail).split()) < 3:
        return False

    return True


def generate_ticket_number():
    now = datetime.now()
    return f"TKT-{now.strftime('%Y%m%d-%H%M%S')}"


def make_result(
    reply,
    intent=None,
    confidence=None,
    matched_question=None,
    retrieval_label=None,
    similarity=None,
    top_matches=None,
    show_debug=True,
):
    return {
        "reply": reply,
        "intent": intent,
        "confidence": confidence,
        "matched_question": matched_question,
        "retrieval_label": retrieval_label,
        "similarity": similarity,
        "top_matches": top_matches or [],
        "show_debug": show_debug,
    }


def result_from_state(reply, state):
    return make_result(
        reply=reply,
        intent=state["indobert_intent"],
        confidence=state["confidence"],
        matched_question=state["matched_question"],
        retrieval_label=state["retrieval_label"],
        similarity=state["similarity"],
        top_matches=state["top_matches"],
    )


def build_pending_problem(user_message, indobert_intent, confidence, retrieval_result):
    return {
        "indobert_intent": indobert_intent,
        "confidence": confidence,
        "retrieval_label": retrieval_result["best_label"],
        "similarity": retrieval_result["similarity_score"],
        "matched_question": retrieval_result["best_text"],
        "top_matches": retrieval_result["top_matches"],
        "keluhan": user_message.strip(),
    }


def answer_from_retrieval(user_message, indobert_intent, confidence, retrieval_result):
    global pending_problem

    pending_problem = build_pending_problem(
        user_message=user_message,
        indobert_intent=indobert_intent,
        confidence=confidence,
        retrieval_result=retrieval_result,
    )

    return make_result(
        reply=(
            f"{retrieval_result['best_response']}\n\n"
            "Apakah masalah sudah teratasi? Ketik 'sudah' jika selesai, "
            "atau 'belum' jika masih bermasalah dan ingin dibuatkan tiket."
        ),
        intent=indobert_intent,
        confidence=confidence,
        matched_question=retrieval_result["best_text"],
        retrieval_label=retrieval_result["best_label"],
        similarity=retrieval_result["similarity_score"],
        top_matches=retrieval_result["top_matches"],
    )


def create_ticket_reply():
    global pending_problem
    global pending_ticket

    ticket_number = generate_ticket_number()

    reply = (
        "Tiket berhasil dibuat.\n\n"
        f"Nomor tiket: {ticket_number}\n"
        f"Kategori: {pending_ticket['intent']}\n"
        f"Keluhan awal: {pending_ticket['keluhan']}\n"
        f"Detail: {pending_ticket['detail_tambahan']}\n"
        "Status: Open"
    )

    result = result_from_state(reply, pending_ticket)

    pending_problem = None
    pending_ticket = None

    return result


def start_ticket_from_pending_problem():
    global pending_ticket

    pending_ticket = {
        "intent": pending_problem["retrieval_label"],
        "indobert_intent": pending_problem["indobert_intent"],
        "confidence": pending_problem["confidence"],
        "keluhan": pending_problem["keluhan"],
        "detail_tambahan": None,
        "matched_question": pending_problem["matched_question"],
        "retrieval_label": pending_problem["retrieval_label"],
        "similarity": pending_problem["similarity"],
        "top_matches": pending_problem["top_matches"],
    }

    return result_from_state(
        reply=(
            "Baik, karena masalah belum teratasi, saya akan bantu buatkan tiket.\n\n"
            "Mohon lengkapi data berikut dalam satu pesan:\n"
            "Nama, Perusahaan/Divisi, Lokasi, Detail masalah."
        ),
        state=pending_ticket,
    )


def handle_pending_ticket(user_message):
    global pending_problem
    global pending_ticket

    if is_cancel_message(user_message):
        pending_problem = None
        pending_ticket = None
        return make_result("Baik, pembuatan tiket dibatalkan.")

    if is_confirmation_message(user_message):
        if pending_ticket["detail_tambahan"] is None:
            return result_from_state(TICKET_DETAIL_FORMAT_REPLY, pending_ticket)

        return create_ticket_reply()

    if not validate_ticket_detail(user_message):
        return result_from_state(TICKET_DETAIL_FORMAT_REPLY, pending_ticket)

    pending_ticket["detail_tambahan"] = user_message.strip()

    return result_from_state(
        reply=(
            "Data tiket sementara:\n\n"
            f"Kategori: {pending_ticket['intent']}\n"
            f"Keluhan awal: {pending_ticket['keluhan']}\n"
            f"Detail tambahan: {pending_ticket['detail_tambahan']}\n\n"
            "Apakah data sudah benar dan tiket ingin dibuat?\n"
            "Ketik 'ya' untuk membuat tiket atau 'batal' untuk membatalkan."
        ),
        state=pending_ticket,
    )


def handle_pending_problem(user_message):
    global pending_problem
    global pending_ticket

    if is_solved_message(user_message):
        result = result_from_state(
            "Baik, tiket tidak dibuat karena masalah sudah teratasi.",
            pending_problem,
        )
        pending_problem = None
        pending_ticket = None
        return result

    if is_unsolved_message(user_message):
        return start_ticket_from_pending_problem()

    indobert_intent, confidence = predict_intent(user_message)
    retrieval_result = find_best_response(user_message)
    matched_question = retrieval_result["best_text"]
    previous_question = pending_problem["matched_question"]

    if (
        retrieval_result["similarity_score"] >= NEW_PROBLEM_THRESHOLD
        and matched_question != previous_question
    ):
        result = answer_from_retrieval(
            user_message=user_message,
            indobert_intent=indobert_intent,
            confidence=confidence,
            retrieval_result=retrieval_result,
        )
        result["reply"] = (
            "Saya mendeteksi Anda menyampaikan keluhan baru. "
            "Saya akan memproses keluhan terbaru ini.\n\n"
            f"{result['reply']}"
        )
        return result

    return result_from_state(
        reply=(
            "Mohon konfirmasi apakah masalah sudah teratasi.\n\n"
            "Ketik 'sudah' jika selesai, atau 'belum' jika masih bermasalah "
            "dan ingin dibuatkan tiket.\n"
            "Jika ingin memulai keluhan baru, ketik 'reset'."
        ),
        state=pending_problem,
    )


def handle_new_problem(user_message):
    if is_unsolved_message(user_message):
        return make_result(
            "Silakan sampaikan keluhan terlebih dahulu agar chatbot dapat mencari jawaban dari dataset."
        )

    indobert_intent, confidence = predict_intent(user_message)
    retrieval_result = find_best_response(user_message)
    similarity = retrieval_result["similarity_score"]

    if similarity >= SIMILARITY_THRESHOLD:
        return answer_from_retrieval(
            user_message=user_message,
            indobert_intent=indobert_intent,
            confidence=confidence,
            retrieval_result=retrieval_result,
        )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        reply = (
            "Maaf, saya belum menemukan jawaban yang sesuai. "
            "Bisa jelaskan masalahnya lebih detail?"
        )
    else:
        reply = (
            "Saya mendeteksi kategori masalahnya, tetapi belum menemukan pertanyaan "
            "yang cukup mirip di dataset. Bisa jelaskan masalahnya lebih detail?"
        )

    return make_result(
        reply=reply,
        intent=indobert_intent,
        confidence=confidence,
        matched_question=retrieval_result["best_text"],
        retrieval_label=retrieval_result["best_label"],
        similarity=similarity,
        top_matches=retrieval_result["top_matches"],
    )


def chatbot_reply(user_message):
    global pending_problem
    global pending_ticket
    global debug_enabled

    if not normalize_text(user_message):
        return make_result(
            "Pesan masih kosong. Silakan tuliskan keluhan atau pertanyaan Anda.",
            show_debug=False,
        )

    if is_debug_off_message(user_message):
        debug_enabled = False
        return make_result("Debug disembunyikan.", show_debug=False)

    if is_debug_on_message(user_message):
        debug_enabled = True
        return make_result("Debug ditampilkan.", show_debug=False)

    if is_reset_message(user_message):
        pending_problem = None
        pending_ticket = None
        return make_result(
            "Baik, percakapan direset. Silakan sampaikan keluhan baru.",
            show_debug=False,
        )

    if pending_ticket is not None:
        return handle_pending_ticket(user_message)

    if pending_problem is not None:
        return handle_pending_problem(user_message)

    return handle_new_problem(user_message)


def print_debug(result):
    confidence = result["confidence"]
    similarity = result["similarity"]

    print("\nDebug:")
    print(f"Intent IndoBERT: {result['intent'] or '-'}")
    print(
        "Confidence IndoBERT: "
        f"{confidence:.2f}" if confidence is not None else "Confidence IndoBERT: -"
    )
    print(f"Matched Question: {result['matched_question'] or '-'}")
    print(f"Retrieval Label: {result['retrieval_label'] or '-'}")
    print(
        f"Similarity: {similarity:.2f}" if similarity is not None else "Similarity: -"
    )
    print("Top 3 Matches:")

    if not result["top_matches"]:
        print("-")
        return

    for index, match in enumerate(result["top_matches"], start=1):
        print(
            f"{index}. [{match['label']}] {match['text']} "
            f"(score: {match['score']:.2f})"
        )


def main():
    print("=== Chatbot IT Support Hybrid IndoBERT + Retrieval ===")
    print("Ketik 'keluar' untuk berhenti.")
    print("Ketik 'reset' untuk memulai keluhan baru.")
    print("Ketik 'debug off' untuk menyembunyikan debug.")
    print("Ketik 'debug on' untuk menampilkan debug.\n")

    while True:
        user_input = input("User: ")

        if normalize_text(user_input) in {"exit", "quit", "keluar"}:
            print("Bot: Terima kasih. Sesi chatbot selesai.")
            break

        result = chatbot_reply(user_input)

        print("\nBot:")
        print(result["reply"])

        if debug_enabled and result["show_debug"]:
            print_debug(result)

        print("-" * 50)


if __name__ == "__main__":
    main()
