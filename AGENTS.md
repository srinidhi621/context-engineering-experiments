# Repository Guidelines

## Project Structure & Module Organization
- Core Python packages live in `src/`, split into `context_engineering/` (assemblers), `corpus/` (loaders and padding pipelines), `models/` (Gemini client + embeddings), `utils/` (monitoring, logging, tokenization), and `evaluation/`.
- Command-line helpers sit in `scripts/`; use them for setup, feasibility checks, monitoring, and reporting.
- Keep raw inputs in `data/`, generated artifacts or monitor snapshots in `results/` (e.g., `results/.monitor_state.json`), exploratory work in `notebooks/`, and reference corpora in `texts/`.
- Pytest suites belong in `tests/` and should mirror the layout under `src/` for quick discovery.

## Build, Test, and Development Commands
- `bash scripts/setup_environment.sh` provisions the virtualenv, installs dependencies, and validates prerequisites.
- After activation (`source venv/bin/activate`), run `pip install -e .` so local packages import cleanly as `src.*`.
- Use `python scripts/estimate_feasibility.py` before large runs to confirm the plan fits rate/budget limits.
- `python scripts/check_rate_limits.py` surfaces the current state tracked by the unified monitor and stored in `results/`.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, and PEP 8 alignment. Match existing modules by providing docstrings and type hints for public APIs.
- Stick to `snake_case` for functions/modules, `CamelCase` for classes, and `UPPER_SNAKE_CASE` for constants; reuse dataclasses for configs (`src/config.py`) and structured payloads.
- When extending monitoring or client code, log via `src.utils.logging.get_logger` and keep pure helpers isolated so they can be unit-tested without network access.
- Place new assemblers or retrieval strategies under `src/context_engineering/` and expose them through the package `__init__.py` for easy importing.

## Testing Guidelines
- Run suites with `python -m pytest`; add focused cases under `tests/test_<feature>.py` that mirror the source module name.
- Cover happy-path and failure scenarios for rate limiting, monitoring persistence, and corpus loaders; mock Gemini responses rather than calling the live API.
- Update or add fixtures alongside tests, and avoid coupling tests to large artifacts in `data/`.

## Temporary Files & Cleanup
- If you create temporary test scripts or helper files for investigation, delete them immediately after the investigation is complete.
- NEVER create markdown files unless explicitly requested by the user. If something needs the user's attention, communicate it directly in the chat window.
- The only circumstance to create and leave a new markdown file is if the user explicitly asks for it.

## Commit & Pull Request Guidelines
- Follow the existing imperative, sentence-case style (`Clean up documentation: fix inconsistencies and duplications`). Keep subjects ≤72 characters and add a short clarifying clause when helpful.
- PRs should explain experiment impact, link to the relevant issue or roadmap phase, and include proof of validation (pytest output, feasibility check results, or monitor snapshots). Add screenshots only when notebook visuals change.
- Confirm tests pass, generated files are ignored via `.gitignore`, and large datasets remain out of Git history before requesting review.

## Task Execution & Alignment
- **Mandatory Review:** Before starting any new task or phase, you MUST list your proposed ToDos and get explicit confirmation from the user.
- **Alignment First:** Do not proceed with implementation until the user has reviewed and approved the plan.
- **Updates:** If the plan changes significantly during execution, pause and re-align with the user.
