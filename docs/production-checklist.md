# Production Review Checklist

## Runtime prerequisites

- [ ] Python runtime available and pinned in deployment image
- [ ] `solve-field` installed and executable
- [ ] Astrometry indexes `4107-4119` present
- [ ] Required reference catalogs present
- [ ] Health endpoints wired into orchestrator probes

## API safety

- [ ] Request body size capped
- [ ] Upload size capped
- [ ] Unsupported file types rejected
- [ ] Busy server returns `429` instead of queueing unbounded work
- [ ] Worker jobs have a hard timeout

## Process management

- [ ] Worker preloaded at startup
- [ ] Worker crash triggers clean rejection of in-flight jobs
- [ ] Graceful shutdown stops HTTP server and worker process
- [ ] Old generated files are cleaned up on an interval

## Security and privacy

- [ ] Security headers applied to all responses
- [ ] Uploaded inputs are not exposed publicly unless explicitly enabled
- [ ] Static file serving prevents path traversal
- [ ] Runtime downloads are disabled in production deploy path

## Observability

- [ ] Request IDs added to API responses
- [ ] Startup failures are explicit and fail fast
- [ ] Health response includes worker/job state
- [ ] Error paths are logged without leaking stack traces to clients

## Verification

- [ ] `bun run typecheck`
- [ ] `bun run test`
- [ ] `bun run sample:orion`
- [ ] Manual smoke check of `/healthz`, `/readyz`, `/api/analyze`
