# Image Lab — Interactive Web-Based Image Processing Platform

CENG401 graduation project. Supervisor: Prof. Dr. Muhammed Fatih Demirci.

| Folder | Contents |
|--------|----------|
| `frontend/` | React (Vite) user interface |
| `backend/` | FastAPI image processing service |

## Run locally

Two terminals:

```bash
# terminal 1
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload

# terminal 2
cd frontend
npm run dev
```

Open http://localhost:5173. First-time setup steps are in each folder's README.
