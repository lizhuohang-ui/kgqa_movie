# Repository Guidelines

## Project Structure & Module Organization

This is a flat Python project for a movie KGQA system using Neo4j, BERT, FastAPI, and Streamlit. Core logic lives in `core_modules.py`; `main_api.py` exposes the FastAPI service; `app.py` is the Streamlit UI. Data and graph setup scripts are `movies_data.csv`, `data_preprocess.py`, and `neo4j_import.py`. Model training is in `intent_model_train.py`, with `intent_model.pth` expected at the repository root. Deployment notes are in `DEPLOYMENT.md`; package pins are in `requirements.txt`.

## Build, Test, and Development Commands

This repository uses a `uv`-managed `.venv`. Do not recreate it unless it is missing. Install or refresh dependencies with:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run commands without activation by prefixing them with `uv run`. Prepare graph data with `uv run python data_preprocess.py`, then import it with `uv run python neo4j_import.py` after Neo4j 5.x is running on `localhost:7687`. Start the API with `uv run uvicorn main_api:app --reload --host 0.0.0.0 --port 8000`. Start the UI with `uv run streamlit run app.py --server.port 8501`. Use `uv run python intent_model_train.py` only for retraining, and `uv run python test_llm_api.py` to check LLM connectivity after setting credentials.

## Coding Style & Naming Conventions

Use Python 3.8-3.10 and follow PEP 8: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants such as `NEO4J_URI`. Keep comments concise and preserve existing Chinese user-facing labels. Prefer parameterized Cypher queries through existing helpers.

## Testing Guidelines

There is no formal pytest suite yet. Add new tests as root-level `test_*.py` files unless a test package is introduced. For API changes, verify `GET /health` and relevant endpoints through Swagger at `http://localhost:8000/docs`. For UI changes, run Streamlit locally and check the affected path.

## Commit & Pull Request Guidelines

Existing commits use short imperative summaries, for example `update README.md` and `improve the front-end page`. Keep future commits similarly focused. Pull requests should include a concise description, commands run, any Neo4j/data/model setup required, linked issues if applicable, and screenshots or short recordings for Streamlit UI changes.

## Security & Configuration Tips

Do not commit real API keys, `.env` files, virtual environments, generated data, or large model artifacts. Set `LLM_API_KEY` and `LLM_URL` in the environment for LLM-backed answers. If local Neo4j credentials differ from `core_modules.py`, document the setup change in the PR.
