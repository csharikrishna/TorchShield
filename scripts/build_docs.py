"""
build_docs.py
Compiles all markdown documentation files in docs/ into beautiful,
light-themed HTML pages and embeds them into docs/index.html with an interactive modal reader.
"""

import os
import re
import json
import markdown

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

DOC_METADATA = [
    {
        "id": "INSTALLATION",
        "file": "INSTALLATION.md",
        "html_file": "INSTALLATION.html",
        "title": "Installation Guide",
        "category": "Getting Started",
        "desc": "Standalone wheel, source distribution, CUDA & CPU setup.",
        "icon": "download"
    },
    {
        "id": "USAGE",
        "file": "USAGE.md",
        "html_file": "USAGE.html",
        "title": "Usage & API Guide",
        "category": "Core Tutorials",
        "desc": "Training checkpoint recipes, metadata inspection, and CLI.",
        "icon": "book-open"
    },
    {
        "id": "HUGGINGFACE_INTEGRATION",
        "file": "HUGGINGFACE_INTEGRATION.md",
        "html_file": "HUGGINGFACE_INTEGRATION.html",
        "title": "Hugging Face Integration",
        "category": "Core Tutorials",
        "desc": "TorchShieldTrainerCallback, Trainer subclass, and pickle purging.",
        "icon": "cpu"
    },
    {
        "id": "THREAT_MODEL",
        "file": "THREAT_MODEL.md",
        "html_file": "THREAT_MODEL.html",
        "title": "Security Threat Model",
        "category": "Security & Architecture",
        "desc": "Formal adversary capabilities, trust boundaries, and defense limits.",
        "icon": "shield"
    },
    {
        "id": "TRAINING_CHECKPOINT_SCHEMA",
        "file": "TRAINING_CHECKPOINT_SCHEMA.md",
        "html_file": "TRAINING_CHECKPOINT_SCHEMA.html",
        "title": "Checkpoint Schema (v5)",
        "category": "Security & Architecture",
        "desc": "Byte-level layout, AAD serialization, and nonces.",
        "icon": "code"
    },
    {
        "id": "KMS_INTEGRATION",
        "file": "KMS_INTEGRATION.md",
        "html_file": "KMS_INTEGRATION.html",
        "title": "KMS Key Management",
        "category": "Security & Architecture",
        "desc": "Envelope encryption with AWS KMS, local keys, and rotation.",
        "icon": "key"
    },
    {
        "id": "ROLLBACK_PROTECTION",
        "file": "ROLLBACK_PROTECTION.md",
        "html_file": "ROLLBACK_PROTECTION.html",
        "title": "Rollback & Freshness Ledger",
        "category": "Security & Architecture",
        "desc": "Monotonic anti-replay ledger and race condition prevention.",
        "icon": "refresh-cw"
    },
    {
        "id": "DISTRIBUTED_CHECKPOINTS",
        "file": "DISTRIBUTED_CHECKPOINTS.md",
        "html_file": "DISTRIBUTED_CHECKPOINTS.html",
        "title": "Distributed Checkpointing",
        "category": "Advanced Systems",
        "desc": "Multi-GPU and multi-node training state synchronization.",
        "icon": "layers"
    },
    {
        "id": "ARTIFACT_PROVENANCE",
        "file": "ARTIFACT_PROVENANCE.md",
        "html_file": "ARTIFACT_PROVENANCE.html",
        "title": "Release Artifact Provenance",
        "category": "Compliance & Release",
        "desc": "Deterministic reproducible builds, cryptographic wheel hashes.",
        "icon": "check-circle"
    }
]

HTML_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__ — TorchShield Documentation</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    body { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    code, pre, .mono { font-family: 'JetBrains Mono', monospace; }
    
    /* Styled Markdown Typography */
    .prose-custom h1 { font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.75rem; letter-spacing: -0.02em; }
    .prose-custom h2 { font-size: 1.4rem; font-weight: 700; color: #1e293b; margin-top: 2rem; margin-bottom: 0.75rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; letter-spacing: -0.01em; }
    .prose-custom h3 { font-size: 1.15rem; font-weight: 700; color: #334155; margin-top: 1.5rem; margin-bottom: 0.5rem; }
    .prose-custom p { margin-top: 0.75rem; margin-bottom: 0.75rem; line-height: 1.7; color: #475569; font-size: 0.95rem; }
    .prose-custom ul { list-style-type: disc; padding-left: 1.5rem; margin-top: 0.75rem; margin-bottom: 0.75rem; color: #475569; font-size: 0.95rem; }
    .prose-custom ol { list-style-type: decimal; padding-left: 1.5rem; margin-top: 0.75rem; margin-bottom: 0.75rem; color: #475569; font-size: 0.95rem; }
    .prose-custom li { margin-top: 0.35rem; margin-bottom: 0.35rem; line-height: 1.6; }
    .prose-custom strong { color: #0f172a; font-weight: 600; }
    .prose-custom code:not(pre code) { background-color: #f1f5f9; color: #4338ca; padding: 0.15rem 0.4rem; border-radius: 0.375rem; font-size: 0.85em; border: 1px solid #e2e8f0; font-family: 'JetBrains Mono', monospace; }
    .prose-custom pre { background-color: #0f172a; border-radius: 0.75rem; padding: 1.25rem; overflow-x: auto; margin-top: 1rem; margin-bottom: 1rem; border: 1px solid #1e293b; position: relative; }
    .prose-custom pre code { color: #f8fafc; font-size: 0.85rem; line-height: 1.6; font-family: 'JetBrains Mono', monospace; background: transparent; padding: 0; }
    .prose-custom table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; margin-bottom: 1.5rem; font-size: 0.875rem; }
    .prose-custom th { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1rem; font-weight: 700; text-align: left; color: #1e293b; }
    .prose-custom td { border: 1px solid #e2e8f0; padding: 0.75rem 1rem; color: #475569; vertical-align: top; }
    .prose-custom tr:nth-child(even) { background-color: #fbfcfe; }
    .prose-custom blockquote { border-left: 4px solid #6366f1; background-color: #f5f7ff; padding: 0.75rem 1.25rem; border-radius: 0 0.5rem 0.5rem 0; margin-top: 1rem; margin-bottom: 1rem; color: #334155; }
    .prose-custom hr { border: 0; border-top: 1px solid #e2e8f0; margin-top: 2rem; margin-bottom: 2rem; }
    .prose-custom a { color: #4f46e5; font-weight: 600; text-decoration: underline; text-underline-offset: 2px; }
    .prose-custom a:hover { color: #3730a3; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen antialiased selection:bg-indigo-500 selection:text-white flex flex-col">

  <!-- TOP NAV -->
  <header class="border-b border-slate-200 bg-white/90 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <a href="index.html" class="flex items-center gap-3 group">
          <div class="h-9 w-9 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-white shadow-sm shadow-indigo-200 group-hover:bg-indigo-700 transition">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg>
          </div>
          <div class="flex items-baseline gap-2">
            <span class="font-bold text-lg tracking-tight text-slate-900">TorchShield</span>
            <span class="text-xs text-slate-500 font-medium">Docs</span>
          </div>
        </a>
        <span class="px-2 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 ml-1">v0.7.1</span>
      </div>

      <div class="flex items-center gap-4">
        <a href="index.html" class="text-xs font-semibold text-slate-600 hover:text-indigo-600 transition flex items-center gap-1.5">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
          Back to Showcase
        </a>
        <a href="https://github.com/csharikrishna/TorchShield" target="_blank" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-300 transition flex items-center gap-2">
          <svg class="w-3.5 h-3.5 fill-slate-800" viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
          GitHub
        </a>
      </div>
    </div>
  </header>

  <!-- MAIN LAYOUT -->
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full flex flex-col md:flex-row gap-8">
    
    <!-- LEFT SIDEBAR -->
    <aside class="w-full md:w-64 shrink-0">
      <div class="sticky top-24 bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Documentation</div>
        <nav class="space-y-1">
          __SIDEBAR_LINKS__
        </nav>
      </div>
    </aside>

    <!-- MAIN ARTICLE -->
    <main class="flex-1 min-w-0 bg-white rounded-2xl border border-slate-200 p-6 sm:p-10 shadow-sm">
      <!-- Breadcrumbs -->
      <div class="flex items-center gap-2 text-xs font-medium text-slate-400 mb-6">
        <a href="index.html" class="hover:text-indigo-600">Home</a>
        <span>/</span>
        <span>Docs</span>
        <span>/</span>
        <span class="text-slate-700 font-semibold">__TITLE__</span>
      </div>

      <!-- Article Content -->
      <article class="prose-custom">
        __RENDERED_CONTENT__
      </article>

      <!-- Pagination Footer -->
      <div class="mt-12 pt-6 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
        <div>
          __PREV_BUTTON__
        </div>
        <div>
          __NEXT_BUTTON__
        </div>
      </div>
    </main>

  </div>

  <!-- FOOTER -->
  <footer class="py-6 bg-white border-t border-slate-200 text-center text-xs text-slate-400">
    <p>TorchShield &bull; Authenticated, Pickle-Free AI Checkpoint Platform &bull; v0.7.1</p>
  </footer>

  <script>
    hljs.highlightAll();

    // Add copy button to all pre blocks
    document.querySelectorAll('pre').forEach(pre => {
      const button = document.createElement('button');
      button.className = 'absolute top-3 right-3 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono border border-slate-700 transition';
      button.innerText = 'Copy';
      button.addEventListener('click', () => {
        const code = pre.querySelector('code').innerText;
        navigator.clipboard.writeText(code).then(() => {
          button.innerText = 'Copied!';
          setTimeout(() => button.innerText = 'Copy', 2000);
        });
      });
      pre.appendChild(button);
    });
  </script>
</body>
</html>
"""

def render_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Preprocess markdown: replace local relative links from .md to .html
    for doc in DOC_METADATA:
        text = text.replace(f"({doc['file']})", f"({doc['html_file']})")
        text = text.replace(f"[`{doc['file']}`]({doc['file']})", f"[`{doc['file']}`]({doc['html_file']})")

    html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc', 'nl2br'])
    return html, text

def main():
    print(f"Building documentation from {DOCS_DIR}...")
    
    compiled_docs = {}

    for idx, doc in enumerate(DOC_METADATA):
        src_path = os.path.join(DOCS_DIR, doc["file"])
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found!")
            continue

        rendered_content, raw_text = render_md(src_path)
        compiled_docs[doc["id"]] = {
            "title": doc["title"],
            "desc": doc["desc"],
            "category": doc["category"],
            "html_file": doc["html_file"],
            "html_content": rendered_content
        }

        # Build sidebar links
        sidebar_html = []
        for other_doc in DOC_METADATA:
            is_active = (other_doc["id"] == doc["id"])
            if is_active:
                sidebar_html.append(f"""
                <a href="{other_doc['html_file']}" class="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                  <span class="w-1.5 h-1.5 rounded-full bg-indigo-600"></span>
                  <span>{other_doc['title']}</span>
                </a>
                """)
            else:
                sidebar_html.append(f"""
                <a href="{other_doc['html_file']}" class="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition">
                  <span class="w-1.5 h-1.5 rounded-full bg-transparent"></span>
                  <span>{other_doc['title']}</span>
                </a>
                """)

        # Prev / Next navigation
        prev_doc = DOC_METADATA[idx - 1] if idx > 0 else None
        next_doc = DOC_METADATA[idx + 1] if idx < len(DOC_METADATA) - 1 else None

        prev_btn = f"""<a href="{prev_doc['html_file']}" class="text-indigo-600 font-semibold hover:underline flex items-center gap-1">&larr; {prev_doc['title']}</a>""" if prev_doc else ""
        next_btn = f"""<a href="{next_doc['html_file']}" class="text-indigo-600 font-semibold hover:underline flex items-center gap-1">{next_doc['title']} &rarr;</a>""" if next_doc else ""

        full_html = HTML_PAGE_TEMPLATE.replace("__TITLE__", doc["title"])\
                                      .replace("__SIDEBAR_LINKS__", "\n".join(sidebar_html))\
                                      .replace("__RENDERED_CONTENT__", rendered_content)\
                                      .replace("__PREV_BUTTON__", prev_btn)\
                                      .replace("__NEXT_BUTTON__", next_btn)

        dst_path = os.path.join(DOCS_DIR, doc["html_file"])
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"Generated: {doc['html_file']}")

    # Write docs_data.js for instant offline/file:/// reader modal access
    js_data_path = os.path.join(DOCS_DIR, "docs_data.js")
    with open(js_data_path, "w", encoding="utf-8") as f:
        f.write(f"window.DOCS_DATA = {json.dumps(compiled_docs, indent=2)};\n")
    print("Generated docs_data.js for instant modal reader!")

if __name__ == "__main__":
    main()
