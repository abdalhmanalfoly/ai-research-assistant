import { mountShell } from '../components/sidebar.js';
import { el, clear, qs } from '../utils/dom.js';
import { icon } from '../components/icons.js';
import { toast } from '../components/toast.js';
import { uploadPaper, getPaper } from '../api/papers.js';
import { getJob, isTerminal, trackJob } from '../api/jobs.js';
import { getHealth, describeHealth } from '../api/health.js';
import { describeError } from '../api/client.js';
import { validatePdf, validateMetadata } from '../utils/validation.js';
import { formatBytes, statusMeta } from '../utils/format.js';
import { LIMITS, TIMING } from '../config.js';

mountShell({ page: 'upload', title: 'Upload Papers' });
const root = qs('#page-content');
let maxMb = LIMITS.defaultMaxPdfMb;
let file = null, submitting = false, pollTimer = null;

getHealth({ timeoutMs: 8000 }).then((h) => { const m = describeHealth(h).maxPdfMb; if (m) { maxMb = m; hint.textContent = hintText(); } }).catch(() => {});
const hintText = () => `PDF only, up to ${maxMb} MB. The server validates the file again.`;

const input = el('input', { type: 'file', class: 'sr-only', attrs: { accept: 'application/pdf,.pdf', id: 'file' } });
const hint = el('div', { class: 'hint', text: hintText() });
const dropLabel = el('label', { class: 'dropzone', attrs: { for: 'file' } }, icon('upload'),
  el('strong', { text: 'Drop a PDF here or click to choose' }), hint);
const fileSlot = el('div');
const fields = {
  title: el('input', { class: 'input', attrs: { id: 'title', placeholder: 'Defaults to the file name' } }),
  authors: el('input', { class: 'input', attrs: { id: 'authors', placeholder: 'Comma-separated' } }),
  year: el('input', { class: 'input', type: 'number', attrs: { id: 'year', min: 1000, max: 3000 } }),
  doi: el('input', { class: 'input', attrs: { id: 'doi', placeholder: '10.1234/abcd' } }),
  sourceUrl: el('input', { class: 'input', type: 'url', attrs: { id: 'src', placeholder: 'https://…' } }),
};
const errs = Object.fromEntries(Object.keys(fields).map((k) => [k, el('div', { class: 'field-error' })]));
const field = (label, key, id) => el('div', { class: 'field' }, el('label', { text: label, attrs: { for: id } }), fields[key], errs[key]);
const submitBtn = el('button', { class: 'btn btn--primary', type: 'submit', disabled: true }, icon('upload'), 'Upload and index');
const form = el('form', { class: 'stack', attrs: { novalidate: true } },
  el('div', { class: 'form-grid' }, field('Title (optional)', 'title', 'title'), field('Authors (optional)', 'authors', 'authors'),
    field('Year (optional)', 'year', 'year'), field('DOI (optional)', 'doi', 'doi'), field('Source URL (optional)', 'sourceUrl', 'src')),
  el('div', { class: 'row' }, submitBtn, el('span', { class: 'hint', text: 'Metadata is stored as you enter it; nothing is extracted or invented for you.' })));
const result = el('div', { class: 'stack', attrs: { 'aria-live': 'polite' } });
root.replaceChildren(el('div', { class: 'stack page-narrow' }, dropLabel, input, fileSlot, form, result));

function setFile(f) {
  const v = validatePdf(f, maxMb);
  clear(fileSlot);
  if (!v.ok) { file = null; fileSlot.append(el('div', { class: 'callout callout--bad', attrs: { role: 'alert' } }, icon('alert'), el('div', { text: v.error }))); }
  else {
    file = f;
    fileSlot.append(el('div', { class: 'file-chip' }, icon('file'), el('div', { attrs: { style: 'flex:1;min-width:0' } }, el('div', { text: f.name, attrs: { style: 'font-weight:600;overflow-wrap:anywhere' } }), el('div', { class: 'small muted', text: formatBytes(f.size) })),
      el('button', { class: 'btn btn--ghost btn--sm', type: 'button', attrs: { 'aria-label': 'Remove selected file' }, on: { click: () => { file = null; input.value = ''; clear(fileSlot); sync(); } } }, icon('close'))));
  }
  sync();
}
const sync = () => { submitBtn.disabled = !file || submitting; };

input.addEventListener('change', () => { if (input.files?.[0]) setFile(input.files[0]); });
for (const evt of ['dragenter', 'dragover']) dropLabel.addEventListener(evt, (e) => { e.preventDefault(); dropLabel.classList.add('is-over'); });
for (const evt of ['dragleave', 'drop']) dropLabel.addEventListener(evt, (e) => { e.preventDefault(); dropLabel.classList.remove('is-over'); });
dropLabel.addEventListener('drop', (e) => { const f = e.dataTransfer?.files?.[0]; if (f) setFile(f); });

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!file || submitting) return;
  const meta = { title: fields.title.value, authors: fields.authors.value, year: fields.year.value, doi: fields.doi.value, sourceUrl: fields.sourceUrl.value };
  const bad = validateMetadata(meta);
  for (const k of Object.keys(errs)) errs[k].textContent = bad[k] ?? '';
  if (Object.keys(bad).length) return;
  submitting = true; sync(); submitBtn.replaceChildren(el('span', { class: 'spinner' }), 'Uploading…');
  clear(result).append(el('div', { class: 'callout callout--info', attrs: { role: 'status' } }, el('span', { class: 'spinner' }), el('div', { text: 'Sending the file to the server… (upload progress is not measurable here)' })));
  try {
    const r = await uploadPaper({ file, ...meta });
    if (r.alreadyRegistered) {
      clear(result).append(el('div', { class: 'callout callout--info' }, icon('info'), el('div', {}, el('div', { text: 'This exact file is already registered, so no new job was created. It is either indexed or still being processed.' }),
        el('a', { href: `papers.html?paper=${encodeURIComponent(r.paperId)}`, text: 'View the paper and its status' }))));
    } else { trackJob(r.jobId); followJob(r); }
    file = null; input.value = ''; clear(fileSlot);
  } catch (err) {
    clear(result).append(el('div', { class: 'callout callout--bad', attrs: { role: 'alert' } }, icon('alert'), el('div', { text: describeError(err) })));
  } finally { submitting = false; submitBtn.replaceChildren(icon('upload'), 'Upload and index'); sync(); }
});

/* Follow the real job until the backend reports a terminal state. "Indexed" is only claimed when the paper says so. */
function followJob({ paperId, jobId }) {
  clearTimeout(pollTimer);
  const card = el('div', { class: 'card stack', attrs: { 'aria-live': 'polite' } });
  clear(result).append(card);
  const paint = (job, paper, note) => {
    const jm = statusMeta('job', job?.status);
    card.replaceChildren(
      el('div', { class: 'row row--between' }, el('strong', { text: 'Processing' }), el('span', { class: `badge badge--${jm.tone}`, text: jm.label })),
      el('div', { class: 'small muted mono', text: `job ${jobId}` }),
      job && !isTerminal(job.status) ? el('div', { class: 'row' }, el('span', { class: 'spinner' }), el('span', { class: 'small muted', text: 'Waiting for the server (extracting text, embedding, indexing). Time depends on the paper length and hardware.' })) : null,
      note,
      el('div', { class: 'row' }, el('a', { class: 'btn btn--sm', href: `papers.html?paper=${encodeURIComponent(paperId)}`, text: 'Open paper' }), el('a', { class: 'btn btn--sm', href: 'jobs.html', text: 'All jobs' }),
        job && isTerminal(job.status) ? el('button', { class: 'btn btn--sm', type: 'button', text: 'Upload another', on: { click: () => { clear(result); window.scrollTo({ top: 0 }); } } }) : null));
  };
  paint(null, null, el('div', { class: 'small muted', text: 'Uploaded. Waiting for the first status from the server…' }));
  const tick = async () => {
    if (!card.isConnected) return;
    try {
      const job = await getJob(jobId);
      if (!isTerminal(job.status)) { paint(job); pollTimer = setTimeout(tick, TIMING.jobPollMs); return; }
      const paper = await getPaper(paperId).catch(() => null);
      if (job.status === 'succeeded' && paper?.status === 'indexed') { paint(job, paper, el('div', { class: 'callout callout--ok' }, icon('check'), el('div', { text: 'Indexed: this paper is now searchable in Research Chat.' }))); toast('Paper indexed', { type: 'ok' }); }
      else if (job.status === 'succeeded') paint(job, paper, el('div', { class: 'callout callout--warn' }, icon('alert'), el('div', { text: `The job succeeded but the paper status is “${paper?.status ?? 'unknown'}”. Check the library before relying on it.` })));
      else paint(job, paper, el('div', { class: `callout callout--${paper?.status === 'needs_ocr' ? 'warn' : 'bad'}` }, icon('alert'), el('div', { text: `${job.errorMessage ?? 'Processing failed.'}${paper?.status === 'needs_ocr' ? ' OCR runs automatically on scanned pages; if it found nothing, try a higher-quality scan (300 DPI) or OCR the file externally.' : ''}` })));
    } catch (err) { card.append(el('div', { class: 'field-error', text: `Could not read job status: ${describeError(err)}` })); pollTimer = setTimeout(tick, TIMING.jobPollMs * 3); }
  };
  tick();
}
globalThis.addEventListener?.('pagehide', () => clearTimeout(pollTimer));
