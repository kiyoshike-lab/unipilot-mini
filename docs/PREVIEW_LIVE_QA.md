# Authorized Preview QA — PHASE51

No Render or Vercel Production deployment/configuration change is authorized in PHASE51. Vercel Preview protection stays enabled. Local Demo, read-only Live readiness and manual authenticated Preview workflow results must be reported separately.

## Bounded CORS preparation (code only)

`api/cors.py` preserves the existing localhost/127.0.0.1:3000 and Production `https://unipilot-mini-pjgy.vercel.app` defaults. Optional `UNIPILOT_CORS_ALLOWED_ORIGINS` adds comma-separated **exact** origins (maximum20 additions /8192 characters). It rejects wildcards, paths, query/fragment, credentials, malformed ports, noncanonical origins and non-local HTTP. Invalid configuration fails closed. Credentials remain disabled; GET/POST methods and existing header policy are unchanged.

Deployment owner procedure, for a separately authorized deployment/configuration change:

1. Identify the intended protected Preview and copy its exact origin (scheme + host + optional port; no trailing slash).
2. Review and set that origin in `UNIPILOT_CORS_ALLOWED_ORIGINS` on the approved API environment. Do not paste a blanket `*.vercel.app` rule or disable CORS.
3. Build the approved Preview with `NEXT_PUBLIC_API_URL` equal to the actual API origin. This is a build-time public value, not a secret. Runtime environment changes cannot correct an already compiled client bundle.
4. Keep Vercel authentication/protection. Remove expired exact Preview entries in a later approved maintenance action.

These are setup instructions, not evidence that live infrastructure has changed. Pushing foundation-research does not authorize a Production rollout.

## Read-only helper

From the repository, provide URLs explicitly; no unique Preview URL is hardcoded:

```powershell
$env:PREVIEW_URL = 'https://YOUR-APPROVED-PREVIEW.vercel.app'
$env:API_URL = 'https://unipilot-mini.onrender.com'
.venv-gpu/Scripts/python.exe web/tests/live-preview-qa.py --output web/qa/phase51/live-api.json
```

The helper sends only GET (Preview, same-origin compiled scripts, API health) and OPTIONS (`/chat/stream`). It records status/CORS, observable compiled API references and login protection. It does not send chat, cookies, passwords, bypass tokens or deploy requests. Output creation is exclusive: use a new output name for a later run. A login redirect or timeout means NOT_TESTED, never a fabricated PASS. Scanning JS for the intended API plus a localhost reference is a conservative readiness check, not execution proof; inspect runtime Network requests in the authorized browser as well.

## Manual authenticated Preview workflow

An authorized user opens the exact Preview, signs in through the normal Vercel flow, and completes this checklist. Do not share credentials, disable protection, or implement an auth bypass. The agent must report NOT_TESTED if no authenticated session is available.

- Record Preview URL, deployment SHA, UTC test time, browser and viewport. Verify the intended deployment contains Stage3.
- In DevTools Network, confirm `/health` and app requests target the intended API, not localhost. Online must reflect a real `{status:ok, loaded:true}` response.
- Inspect a chat preflight: exact Access-Control-Allow-Origin equals this Preview; POST/content-type allowed; no credential wildcard. Check the actual console for CORS errors.
- With explicit approval for the test message, send a non-sensitive Tutor query. Observe an intermediate stream update and the final reply. Test Materials/Exam within their small context budget. This step is an actual model workflow, unlike GET/OPTIONS readiness; do not report it performed merely because health succeeded.
- On Report/Research, add user-supplied evidence, check a matching and nonmatching span, link/unlink a claim, save/reload and Clear. Confirm these operations send no content to an API. Keep source summaries separate from original evidence.
- Test360/390/768/1024/1440 widths, keyboard focus,44px controls and reduced motion. Record failures and uncertainty; no quality claims from synthetic Demo outputs.

Record the three outcomes independently: DEMO; LIVE read-only readiness; MANUAL_AUTHENTICATED_PREVIEW (with actual workflow evidence). Local exact-origin middleware tests are not evidence of deployed Render CORS. Production and external AI APIs remain unchanged/OFF in this phase.
