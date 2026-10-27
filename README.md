# Chatbot NLP API

Project ini berisi model IndoBERT dan retrieval CSV untuk chatbot IT Support.

## Menjalankan API

```bash
cd /Users/andikabintang/Documents/Developer/Project/chatbot_nlp
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8001 --reload
```

## Endpoint

```bash
curl http://127.0.0.1:8001/health
```

```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"printer saya bergaris","session_id":"demo","debug":true}'
```

API ini stateless. State percakapan, tombol sudah/belum, dan pembuatan tiket diatur oleh Laravel.
