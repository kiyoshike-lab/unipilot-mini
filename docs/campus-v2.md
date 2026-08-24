# Campus v2 (opt-in candidate)

Campus v2 changes only the orchestration layer. It does not change or train the v0.4 model.

Install the development-only Router dependency:

```powershell
python -m pip install -r requirements-campus-v2.txt
```

Generate the fixed data files and run evaluation:

```powershell
python -m scripts.build_campus_v2_data
python -m evaluation.analyze_campus_v1_router
python -m evaluation.evaluate_campus_v2_router --split dev
python -m evaluation.evaluate_campus_v2
```

Campus v2 is disabled unless explicitly selected:

```powershell
$env:UNIPILOT_PIPELINE_VERSION = "campus-v2"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`requirements-prod.txt` and the production environment remain unchanged. Do not promote Campus v2 while `evaluation/campus-v2-production-gate.json` is failed.
