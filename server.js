// LR KrishiPrep - by LR AgroSense
// Mock tests + AI doubt solver (RAG) + Telegram bot
require('dotenv').config();
const express = require('express');
const path = require('path');
const crypto = require('crypto');
const { MongoClient, ObjectId } = require('mongodb');

const {
  MONGODB_URI,
  GEMINI_API_KEY,
  GROQ_API_KEY,
  ADMIN_PASSWORD,
  TELEGRAM_BOT_TOKEN,
  TELEGRAM_CHANNEL_ID, // example: @lrkrishiprep
  CRON_KEY,
  PUBLIC_URL, // example: https://lr-krishiprep.onrender.com
  RAZORPAY_KEY_ID,
  RAZORPAY_KEY_SECRET,
  PREMIUM_PRICE = '149', // rupees per 30 days
  FREE_DAILY_LIMIT = '5', // free AI doubts per day in the Telegram bot
  GROQ_MODEL = 'llama-3.3-70b-versatile',
  GEMINI_MODEL = 'gemini-2.5-flash',
  EMBED_MODEL = 'gemini-embedding-001',
  PORT = 3000
} = process.env;

const app = express();
app.set('trust proxy', 1);
app.use(express.json({ limit: '5mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/api', (req, res, next) => {
  if (!dbReady) return res.status(503).json({ error: 'The site is starting up or the database is not connected yet. Please try again in a minute.' });
  next();
});

// ---------- Database ----------
let Questions, Chunks, Users;
let chunkCache = null;
let dbReady = false;

async function connectDb() {
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  const db = client.db('krishiprep');
  Questions = db.collection('questions');
  Chunks = db.collection('chunks');
  Users = db.collection('users');
  await Questions.createIndex({ exam: 1, subject: 1 });
  await Chunks.createIndex({ source: 1 });
  await Users.createIndex({ tgId: 1 }, { unique: true });
}

// ---------- Helpers ----------
const wrap = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch((e) => {
    console.error(e);
    res.status(500).json({ error: e.message || 'Server error' });
  });

const hits = new Map();
function rateLimit(key, max = 30, windowMs = 60 * 60 * 1000) {
  const now = Date.now();
  const list = (hits.get(key) || []).filter((t) => now - t < windowMs);
  if (list.length >= max) return false;
  list.push(now);
  hits.set(key, list);
  return true;
}

function adminOnly(req, res, next) {
  if (!ADMIN_PASSWORD || req.get('x-admin-password') !== ADMIN_PASSWORD) {
    return res.status(401).json({ error: 'Wrong admin password' });
  }
  next();
}

function channelLink() {
  if (TELEGRAM_CHANNEL_ID && TELEGRAM_CHANNEL_ID.startsWith('@')) {
    return 'https://t.me/' + TELEGRAM_CHANNEL_ID.slice(1);
  }
  return '';
}

// ---------- RAG: embeddings + search ----------
function embedRequest(text, taskType) {
  return {
    model: `models/${EMBED_MODEL}`,
    content: { parts: [{ text }] },
    taskType,
    outputDimensionality: 768
  };
}

async function embedMany(texts, taskType = 'RETRIEVAL_DOCUMENT') {
  const all = [];
  for (let i = 0; i < texts.length; i += 50) {
    const batch = texts.slice(i, i + 50);
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:batchEmbedContents`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY },
        body: JSON.stringify({ requests: batch.map((t) => embedRequest(t, taskType)) })
      }
    );
    const data = await res.json();
    if (!res.ok) throw new Error('Embedding failed: ' + (data.error?.message || res.status));
    data.embeddings.forEach((e) => all.push(e.values));
  }
  return all;
}

async function embedOne(text, taskType) {
  const [v] = await embedMany([text], taskType);
  return v;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function chunkText(text, size = 1000, overlap = 150) {
  const clean = text.replace(/\r/g, '').replace(/\n{3,}/g, '\n\n').trim();
  const chunks = [];
  let i = 0;
  while (i < clean.length) {
    let end = Math.min(i + size, clean.length);
    if (end < clean.length) {
      const cut = Math.max(clean.lastIndexOf('\n', end), clean.lastIndexOf('. ', end));
      if (cut > i + size * 0.5) end = cut + 1;
    }
    chunks.push(clean.slice(i, end).trim());
    if (end >= clean.length) break;
    i = end - overlap;
  }
  return chunks.filter((c) => c.length > 30);
}

async function getChunks() {
  if (!chunkCache) {
    chunkCache = await Chunks.find({}, { projection: { text: 1, source: 1, embedding: 1 } }).toArray();
  }
  return chunkCache;
}

async function retrieve(question, k = 5) {
  const all = await getChunks();
  if (!all.length) return [];
  const qv = await embedOne(question, 'RETRIEVAL_QUERY');
  return all
    .map((c) => ({ text: c.text, source: c.source, score: cosine(qv, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .filter((c) => c.score > 0.3);
}

// ---------- AI answer ----------
const SYSTEM_PROMPT = `You are LR KrishiPrep AI, a tutor for agriculture students (BSc Agriculture, MSc Agriculture, MBA Agri-business) and for all agriculture exams (ICAR PG/JRF/SRF, IBPS AFO, NABARD, state Agriculture Officer/AEO, FCI, university and other exams). You were developed by LR AgroSense.
Rules:
- Explain clearly and simply in short points, with the key facts students need for exams (scientific names, numbers, definitions).
- Reply in the same language the student used (English, Telugu or Hindi).
- Students often use local names for crop diseases, pests and practices. Always read the question in its agricultural meaning. For example, in Telugu 'అగ్గి తెగులు' means rice blast disease (caused by the fungus Magnaporthe oryzae), not a fire; 'తెగులు' means a plant disease; 'పురుగు' means an insect pest. If a term is unclear, state your interpretation in one line first.
- Keep answers short and clear. Never repeat the same word or line.
- Only answer agriculture, allied subjects and exam-preparation questions. Politely decline anything else.`;

let groqModel = GROQ_MODEL;
let groqChecked = false;

// Pick a Groq model this API key can actually use
async function pickGroqModel() {
  if (groqChecked) return groqModel;
  groqChecked = true;
  try {
    const res = await fetch('https://api.groq.com/openai/v1/models', { headers: { Authorization: `Bearer ${GROQ_API_KEY}` } });
    const data = await res.json();
    const ids = (data.data || []).map((m) => m.id);
    if (ids.includes(groqModel)) return groqModel;
    const preferred = ['openai/gpt-oss-120b', 'llama-3.3-70b-versatile', 'meta-llama/llama-4-maverick-17b-128e-instruct', 'qwen/qwen3-32b', 'openai/gpt-oss-20b', 'llama-3.1-8b-instant'];
    const pick = preferred.find((p) => ids.includes(p)) || ids.find((id) => !/whisper|tts|guard|embed|orpheus|playai|compound/i.test(id));
    if (pick) {
      groqModel = pick;
      console.log('Using Groq model:', pick);
    }
  } catch (e) {
    console.error('Groq model check failed:', e.message);
  }
  return groqModel;
}

async function generate(system, user) {
  if (GROQ_API_KEY) {
    try {
      const model = await pickGroqModel();
      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${GROQ_API_KEY}` },
        body: JSON.stringify({
          model,
          temperature: 0.3,
          max_tokens: 1200,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user }
          ]
        })
      });
      const data = await res.json();
      if (res.ok) return data.choices[0].message.content;
      console.error('Groq error:', data.error?.message);
    } catch (e) {
      console.error('Groq failed:', e.message);
    }
  }
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: 'user', parts: [{ text: user }] }],
        generationConfig: { temperature: 0.3 }
      })
    }
  );
  const data = await res.json();
  if (!res.ok) throw new Error('AI failed: ' + (data.error?.message || res.status));
  return (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('') || 'Sorry, I could not answer that.';
}

async function askAI(question) {
  let ctx = [];
  try {
    ctx = await retrieve(question);
  } catch (e) {
    console.error('Search failed:', e.message);
  }
  const material = ctx.length
    ? ctx.map((c, i) => `[${i + 1}] (${c.source})\n${c.text}`).join('\n\n')
    : 'No study material found for this question.';
  const user = `Study material:\n${material}\n\nStudent question: ${question}\n\nAnswer using the study material when it is relevant. If the study material does not cover the question, answer from your general agriculture knowledge and add this line at the end: "(Not from our study notes - please verify with your textbook.)"`;
  const answer = await generate(SYSTEM_PROMPT, user);
  return { answer, sources: [...new Set(ctx.map((c) => c.source))] };
}

// ---------- Question parser ----------
// Format:
// Q: question text
// A) option  B) option  C) option  D) option (each on its own line)
// Answer: B
// Explanation: short reason
function parseQuestions(text) {
  const good = [];
  const bad = [];
  let cur = null;
  const finish = () => {
    if (!cur) return;
    const ok = cur.question && cur.options.every(Boolean) && cur.answer >= 0;
    if (ok) good.push({ question: cur.question.trim(), options: cur.options, answer: cur.answer, explanation: cur.explanation.trim() });
    else bad.push(cur.question || '(empty question)');
  };
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    let m;
    if ((m = line.match(/^Q\s*\d*\s*[:.)]\s*(.+)$/i))) {
      finish();
      cur = { question: m[1], options: [null, null, null, null], answer: -1, explanation: '', inExp: false };
    } else if (!cur) {
      continue;
    } else if ((m = line.match(/^\(?([A-D])\s*[).:]\s*(.+)$/i)) && !cur.inExp) {
      cur.options['ABCD'.indexOf(m[1].toUpperCase())] = m[2].trim();
    } else if ((m = line.match(/^(Answer|Ans|Correct answer)\s*[:.-]\s*\(?([A-D])\b/i))) {
      cur.answer = 'ABCD'.indexOf(m[2].toUpperCase());
    } else if ((m = line.match(/^Explanation\s*[:.-]\s*(.*)$/i))) {
      cur.explanation = m[1];
      cur.inExp = true;
    } else if (cur.inExp) {
      cur.explanation += ' ' + line;
    } else if (!cur.options[0]) {
      cur.question += ' ' + line;
    }
  }
  finish();
  return { good, bad };
}

// ---------- Public API ----------
app.get('/api/meta', wrap(async (req, res) => {
  const rows = await Questions.aggregate([
    { $group: { _id: { exam: '$exam', subject: '$subject' }, count: { $sum: 1 } } }
  ]).toArray();
  const exams = {};
  for (const r of rows) {
    const e = r._id.exam || 'General';
    const s = r._id.subject || 'General';
    exams[e] = exams[e] || { total: 0, subjects: {} };
    exams[e].total += r.count;
    exams[e].subjects[s] = r.count;
  }
  res.json({ exams, channelLink: channelLink() });
}));

app.get('/api/test', wrap(async (req, res) => {
  const match = {};
  if (req.query.exam) match.exam = String(req.query.exam);
  if (req.query.subject) match.subject = String(req.query.subject);
  const count = Math.min(Math.max(parseInt(req.query.count, 10) || 20, 5), 100);
  const questions = await Questions.aggregate([
    { $match: match },
    { $sample: { size: count } },
    { $project: { question: 1, options: 1, exam: 1, subject: 1 } }
  ]).toArray();
  res.json({ questions });
}));

app.post('/api/test/submit', wrap(async (req, res) => {
  const list = Array.isArray(req.body.answers) ? req.body.answers.slice(0, 100) : [];
  const ids = list.filter((a) => ObjectId.isValid(a.id)).map((a) => new ObjectId(a.id));
  const docs = await Questions.find({ _id: { $in: ids } }).toArray();
  const byId = new Map(docs.map((d) => [String(d._id), d]));
  let correct = 0;
  const results = [];
  for (const a of list) {
    const d = byId.get(String(a.id));
    if (!d) continue;
    const choice = Number.isInteger(a.choice) ? a.choice : null;
    if (choice === d.answer) correct++;
    results.push({ id: a.id, choice, answer: d.answer, explanation: d.explanation || '' });
  }
  res.json({ correct, total: results.length, results });
}));

app.post('/api/ask', wrap(async (req, res) => {
  const question = String(req.body.question || '').trim();
  if (!question) return res.status(400).json({ error: 'Please type a question.' });
  if (question.length > 1000) return res.status(400).json({ error: 'Question is too long (max 1000 characters).' });
  if (!rateLimit('web:' + req.ip, 30)) return res.status(429).json({ error: 'Hourly limit reached. Please try again later.' });
  res.json(await askAI(question));
}));

// ---------- Admin API ----------
app.get('/api/admin/stats', adminOnly, wrap(async (req, res) => {
  const questions = await Questions.countDocuments();
  const sources = await Chunks.aggregate([{ $group: { _id: '$source', chunks: { $sum: 1 } } }, { $sort: { _id: 1 } }]).toArray();
  res.json({ questions, sources: sources.map((s) => ({ source: s._id, chunks: s.chunks })) });
}));

app.post('/api/admin/questions', adminOnly, wrap(async (req, res) => {
  const exam = String(req.body.exam || '').trim();
  const subject = String(req.body.subject || '').trim();
  if (!exam || !subject) return res.status(400).json({ error: 'Exam and subject are required.' });
  const { good, bad } = parseQuestions(String(req.body.text || ''));
  if (good.length) {
    const now = new Date();
    await Questions.insertMany(good.map((q) => ({ ...q, exam, subject, createdAt: now })));
  }
  res.json({ added: good.length, skipped: bad });
}));

app.delete('/api/admin/questions', adminOnly, wrap(async (req, res) => {
  const exam = String(req.query.exam || '').trim();
  const subject = String(req.query.subject || '').trim();
  if (!exam) return res.status(400).json({ error: 'Exam is required.' });
  const filter = subject ? { exam, subject } : { exam };
  const r = await Questions.deleteMany(filter);
  res.json({ deleted: r.deletedCount });
}));

app.post('/api/admin/notes', adminOnly, wrap(async (req, res) => {
  const source = String(req.body.source || '').trim();
  const text = String(req.body.text || '');
  if (!source || text.trim().length < 50) return res.status(400).json({ error: 'Add a source name and at least 50 characters of notes.' });
  if (text.length > 300000) return res.status(400).json({ error: 'Too long. Paste at most 300,000 characters at a time.' });
  const pieces = chunkText(text);
  const vectors = await embedMany(pieces);
  const now = new Date();
  await Chunks.insertMany(pieces.map((t, i) => ({ source, text: t, embedding: vectors[i], createdAt: now })));
  chunkCache = null;
  res.json({ added: pieces.length });
}));

app.delete('/api/admin/notes', adminOnly, wrap(async (req, res) => {
  const source = String(req.query.source || '');
  const r = await Chunks.deleteMany({ source });
  chunkCache = null;
  res.json({ deleted: r.deletedCount });
}));

app.post('/api/admin/post-to-channel', adminOnly, wrap(async (req, res) => {
  const r = await postDailyQuestion(req.body.exam);
  res.json(r);
}));

// For automatic daily posting (use cron-job.org to open this link once a day)
app.get('/api/cron/daily-question', wrap(async (req, res) => {
  if (!CRON_KEY || req.query.key !== CRON_KEY) return res.status(401).json({ error: 'Wrong key' });
  res.json(await postDailyQuestion());
}));

// ---------- Telegram ----------
async function tg(method, body) {
  const res = await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!data.ok) console.error('Telegram', method, data.description);
  return data;
}

function sendQuizPoll(chatId, q) {
  const body = {
    chat_id: chatId,
    question: q.question.slice(0, 300),
    options: q.options.map((o) => ({ text: o.slice(0, 100) })),
    type: 'quiz',
    correct_option_id: q.answer,
    is_anonymous: true
  };
  if (q.explanation) body.explanation = q.explanation.slice(0, 200);
  return tg('sendPoll', body);
}

async function postDailyQuestion(exam) {
  if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHANNEL_ID) return { ok: false, error: 'Telegram bot token or channel is not set.' };
  const match = exam ? { exam: String(exam) } : {};
  const [q] = await Questions.aggregate([{ $match: match }, { $sample: { size: 1 } }]).toArray();
  if (!q) return { ok: false, error: 'No questions found.' };
  const r = await sendQuizPoll(TELEGRAM_CHANNEL_ID, q);
  return r.ok ? { ok: true, question: q.question } : { ok: false, error: r.description };
}

// ---------- Premium (Razorpay payment link) ----------
const DAY = 24 * 60 * 60 * 1000;
const todayIST = () => new Date(Date.now() + 5.5 * 60 * 60 * 1000).toISOString().slice(0, 10);
const fmtDate = (d) => new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'Asia/Kolkata' });

async function getUser(tgId) {
  return (await Users.findOne({ tgId: String(tgId) })) || { tgId: String(tgId) };
}
const isPremium = (u) => u.premiumUntil && new Date(u.premiumUntil) > new Date();

// Counts one AI doubt for a free user. Returns false if today's free limit is used up.
async function useFreeDoubt(tgId) {
  const today = todayIST();
  const u = await getUser(tgId);
  const used = u.usageDate === today ? u.usageCount || 0 : 0;
  if (used >= Number(FREE_DAILY_LIMIT)) return false;
  await Users.updateOne({ tgId: String(tgId) }, { $set: { usageDate: today, usageCount: used + 1 } }, { upsert: true });
  return true;
}

async function razorpay(method, urlPath, body) {
  const auth = Buffer.from(RAZORPAY_KEY_ID + ':' + RAZORPAY_KEY_SECRET).toString('base64');
  const res = await fetch('https://api.razorpay.com/v1' + urlPath, {
    method,
    headers: { Authorization: 'Basic ' + auth, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  });
  const data = await res.json();
  if (!res.ok) throw new Error((data.error && data.error.description) || 'Razorpay error');
  return data;
}

async function createPaymentLink(tgId, name) {
  return razorpay('POST', '/payment_links', {
    amount: Number(PREMIUM_PRICE) * 100,
    currency: 'INR',
    description: 'LR KrishiPrep Premium - 30 days',
    customer: { name: String(name || 'Telegram user').slice(0, 50) },
    notes: { tg_id: String(tgId) },
    callback_url: PUBLIC_URL.replace(/\/$/, '') + '/payment/done',
    callback_method: 'get',
    expire_by: Math.floor(Date.now() / 1000) + 24 * 60 * 60
  });
}

// Gives 30 days of premium for a paid link (only once per link).
async function activatePremium(linkId) {
  const link = await razorpay('GET', '/payment_links/' + encodeURIComponent(linkId));
  if (link.status !== 'paid') return { ok: false, reason: 'Payment not completed yet.' };
  const tgId = link.notes && link.notes.tg_id;
  if (!tgId) return { ok: false, reason: 'Payment is not linked to a Telegram user.' };
  const u = await getUser(tgId);
  if ((u.paidLinks || []).includes(link.id)) return { ok: true, tgId, until: u.premiumUntil, already: true };
  const start = isPremium(u) ? new Date(u.premiumUntil).getTime() : Date.now();
  const until = new Date(start + 30 * DAY);
  await Users.updateOne(
    { tgId: String(tgId) },
    { $set: { premiumUntil: until }, $push: { paidLinks: link.id } },
    { upsert: true }
  );
  if (TELEGRAM_BOT_TOKEN) {
    await tg('sendMessage', {
      chat_id: tgId,
      text: `Payment received. Thank you!\n\nYou are now a Premium member until ${fmtDate(until)}.\nAsk unlimited doubts any time.`
    });
  }
  return { ok: true, tgId, until };
}

function resultPage(title, message, ok) {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title></head><body style="font-family:system-ui,sans-serif;background:#eef6f0;margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center;padding:16px">
<div style="background:#fff;max-width:420px;width:100%;border-radius:16px;padding:28px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.08)">
<div style="font-size:48px">${ok ? '✅' : '⚠️'}</div><h2 style="color:#1f7a3a;margin:8px 0">${title}</h2>
<p style="color:#333;line-height:1.5">${message}</p>
<p style="color:#777;font-size:14px">You can close this page and go back to Telegram.</p></div></body></html>`;
}

// Razorpay sends the buyer here after payment.
app.get('/payment/done', wrap(async (req, res) => {
  const linkId = String(req.query.razorpay_payment_link_id || '');
  if (!dbReady || !RAZORPAY_KEY_ID || !linkId) {
    return res.status(400).send(resultPage('Something went wrong', 'Payment details are missing. If money was taken, send /status to the bot or contact us.', false));
  }
  const r = await activatePremium(linkId);
  if (!r.ok) return res.send(resultPage('Payment not confirmed', r.reason + ' If money was taken, it will be checked soon.', false));
  res.send(resultPage('Premium activated', `Your LR KrishiPrep Premium is active until <b>${fmtDate(r.until)}</b>.`, true));
}));

function cleanForTelegram(text) {
  return text.replace(/\*\*/g, '').replace(/^#+\s*/gm, '');
}

async function handleTelegram(update) {
  const msg = update.message;
  if (!msg || !msg.text || msg.chat.type !== 'private') return;
  const chatId = msg.chat.id;
  const text = msg.text.trim();

  if (text.startsWith('/start') || text.startsWith('/help')) {
    return tg('sendMessage', {
      chat_id: chatId,
      text:
        'Welcome to LR KrishiPrep by LR AgroSense!\n\n' +
        '• Send me any agriculture question (English, Telugu or Hindi).\n' +
        '• /quiz - get a practice question\n' +
        `• Free: ${FREE_DAILY_LIMIT} AI doubts per day\n` +
        `• /premium - unlimited AI doubts for ₹${PREMIUM_PRICE}/month\n` +
        '• /status - see your plan\n' +
        (PUBLIC_URL ? `• Full mock tests: ${PUBLIC_URL}` : '')
    });
  }

  if (text.startsWith('/quiz')) {
    const [q] = await Questions.aggregate([{ $sample: { size: 1 } }]).toArray();
    if (!q) return tg('sendMessage', { chat_id: chatId, text: 'No questions yet. Please check back soon!' });
    return sendQuizPoll(chatId, q);
  }

  if (text.startsWith('/premium')) {
    const u = await getUser(chatId);
    if (!RAZORPAY_KEY_ID || !RAZORPAY_KEY_SECRET || !PUBLIC_URL) {
      return tg('sendMessage', { chat_id: chatId, text: 'Premium payments will open soon. Please check back later!' });
    }
    const link = await createPaymentLink(chatId, msg.from && msg.from.first_name);
    return tg('sendMessage', {
      chat_id: chatId,
      text:
        'LR KrishiPrep Premium - ₹' + PREMIUM_PRICE + ' for 30 days\n\n' +
        '✔ Unlimited AI doubts (English, Telugu, Hindi)\n' +
        '✔ Priority answers for all agriculture exams\n\n' +
        (isPremium(u) ? `You are already premium until ${fmtDate(u.premiumUntil)}. Paying again adds 30 more days.\n\n` : '') +
        'Pay with UPI, card or net banking (link valid for 24 hours). Premium starts automatically after payment.',
      reply_markup: { inline_keyboard: [[{ text: `Pay ₹${PREMIUM_PRICE}`, url: link.short_url }]] }
    });
  }

  if (text.startsWith('/status')) {
    const u = await getUser(chatId);
    const used = u.usageDate === todayIST() ? u.usageCount || 0 : 0;
    return tg('sendMessage', {
      chat_id: chatId,
      text: isPremium(u)
        ? `Plan: Premium ✅\nValid until: ${fmtDate(u.premiumUntil)}\nAI doubts: unlimited`
        : `Plan: Free\nAI doubts used today: ${used}/${FREE_DAILY_LIMIT}\n\nSend /premium for unlimited doubts at ₹${PREMIUM_PRICE}/month.`
    });
  }

  const user = await getUser(chatId);
  if (!isPremium(user) && !(await useFreeDoubt(chatId))) {
    return tg('sendMessage', {
      chat_id: chatId,
      text: `You have used your ${FREE_DAILY_LIMIT} free AI doubts for today. They reset at midnight.\n\nGet unlimited doubts for ₹${PREMIUM_PRICE}/month - send /premium`
    });
  }

  if (!rateLimit('tg:' + chatId, isPremium(user) ? 60 : 20)) {
    return tg('sendMessage', { chat_id: chatId, text: 'You have reached the hourly limit. Please try again later.' });
  }
  await tg('sendChatAction', { chat_id: chatId, action: 'typing' });
  const { answer } = await askAI(text.slice(0, 1000));
  const clean = cleanForTelegram(answer);
  for (let i = 0; i < clean.length; i += 4000) {
    await tg('sendMessage', { chat_id: chatId, text: clean.slice(i, i + 4000) });
  }
}

const TG_PATH = TELEGRAM_BOT_TOKEN
  ? '/telegram/' + crypto.createHash('sha256').update(TELEGRAM_BOT_TOKEN).digest('hex').slice(0, 32)
  : null;

if (TG_PATH) {
  app.post(TG_PATH, (req, res) => {
    res.sendStatus(200);
    handleTelegram(req.body).catch((e) => console.error('Telegram handler:', e.message));
  });
}

// ---------- Start ----------
// The website opens even if the database is not ready yet; it keeps retrying.
app.listen(PORT, () => console.log('LR KrishiPrep running on port ' + PORT));

async function startDb() {
  if (!MONGODB_URI) {
    console.error('MONGODB_URI is not set. Add it in Render > Environment.');
    return;
  }
  try {
    await connectDb();
    dbReady = true;
    console.log('Database connected');
  } catch (e) {
    console.error('Database connection failed:', e.message, '- retrying in 30 seconds');
    setTimeout(startDb, 30000);
  }
}

async function startTelegram() {
  if (!TG_PATH || !PUBLIC_URL) return;
  try {
    await tg('setWebhook', { url: PUBLIC_URL.replace(/\/$/, '') + TG_PATH, allowed_updates: ['message'] });
    await tg('setMyCommands', {
      commands: [
        { command: 'quiz', description: 'Get a practice question' },
        { command: 'premium', description: 'Unlimited AI doubts - ₹' + PREMIUM_PRICE + '/month' },
        { command: 'status', description: 'See your plan and free doubts left' },
        { command: 'help', description: 'How to use this bot' }
      ]
    });
    console.log('Telegram bot connected');
  } catch (e) {
    console.error('Telegram setup failed:', e.message);
  }
}

startDb();
startTelegram();
