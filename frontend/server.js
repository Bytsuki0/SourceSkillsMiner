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
app.use('/api', createProxyMiddleware({
    target:       BACKEND_URL,
    changeOrigin: true,
    // SSE needs these so Express doesn't buffer the stream
    on: {
        proxyReq: (proxyReq) => {
            proxyReq.setHeader('Accept', 'text/event-stream');
        },
    },
    selfHandleResponse: false,
}));

// ── Serve static frontend ────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

// SPA fallback
app.get(/.*/, (_req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
app.listen(PORT, () => {
    console.log(`\n  SourceSkillsMiner UI`);
    console.log(`  ─────────────────────────────`);
    console.log(`  Frontend : http://localhost:${PORT}`);
    console.log(`  Backend  : ${BACKEND_URL}`);
    console.log();
});
