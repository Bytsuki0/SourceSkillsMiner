/**
 * frontend/server.js
 *
 * Node.js / Express server that:
 *   - Serves the static dashboard (public/)
 *   - Proxies /api/* requests to the Flask backend
 *
 * Start with:
 *   node server.js
 *   (or: npm start)
 */

const express  = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path     = require('path');

const app         = express();
const PORT        = process.env.PORT        || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

// ── Proxy all /api/* to Flask ────────────────────────────────────────────────
// IMPORTANT: mount at root '/' and use pathFilter, NOT app.use('/api', ...).
// When Express mounts middleware with app.use('/api', ...) it strips the '/api'
// segment before the proxy sees the path, so Flask receives '/analyze' instead
// of '/api/analyze'.  Using pathFilter at root preserves the full path.
app.use(createProxyMiddleware({
    target:       BACKEND_URL,
    changeOrigin: true,
    pathFilter:   '/api',           // only forward requests whose path starts with /api
    on: {
        proxyReq: (proxyReq) => {
            proxyReq.setHeader('Accept', 'text/event-stream');
        },
    },
}));

// ── Serve static frontend ────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

// SPA fallback
app.get(/.*/, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`\n  SourceSkillsMiner UI`);
    console.log(`  ─────────────────────────────`);
    console.log(`  Frontend : http://localhost:${PORT}`);
    console.log(`  Backend  : ${BACKEND_URL}`);
    console.log();
});