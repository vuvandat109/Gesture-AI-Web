# Deploy Gesture AI Web on Vercel

The site is a **Flask** application, not a static site. Its entry point is
`app.py`; `vercel.json` enables Flask detection.

## Prerequisites

1. Sign in to Vercel, and connect GitHub as a login provider.
2. **Install the Vercel GitHub App** on GitHub (this is separate from
   "Authorize Vercel" for login). Open https://github.com/apps/vercel
   and choose Configure/Install, then select `vuvandat109/Gesture-AI-Web`.
3. Import `vuvandat109/Gesture-AI-Web` into Vercel. Use project root `.`
   and framework `Flask`.
4. Set environment variables in Vercel before relying on accounts:
   - `DATABASE_URL`: the **external** PostgreSQL connection URL from
     Render Postgres, or another durable database reachable from Vercel.
     Render's *internal* database URL is not valid from Vercel.
   - `SECRET_KEY`: a stable random secret across deployments.
   - `COOKIE_SECURE=1`: HTTPS-only session cookie.
   Never commit, share, log, or paste these secrets.

## Verify the deployment

Open `/api/health` and require `model_ready=true` and
`account_storage=postgresql`. Then test registration, return to login,
authentication, a real Sensor Logger CSV upload and prediction, logout,
and a second login after a redeployment.

This repo contains the real trained model
`models/gesture_real_v1.joblib`, but running under Vercel's Python
Functions is different from Render's always-running Flask/Gunicorn web
service; verify model loading, package build size, cold starts and
function timeouts. **Do not shut down Render until the Vercel deployment
and PostgreSQL persistence are actually verified.**

The existing Render PostgreSQL free instance expires on 2026-11-26
according to the service metadata observed on 2026-10-28. Migrating
the web host alone does not extend the database's lifetime.

This preparation does **not** migrate old accounts from SQLite, copy
secrets, create a Vercel project or establish a production URL.
