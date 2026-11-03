"""Serve the completed experiment's evidence locally, without editing any artifacts."""
import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
KINDS = {
    'inputs': ('Inputs', 'data/processed/researched_forecaster/prompts', '.txt',
               'The exact text sent to the frozen forecasting encoder, including the booster probability and every evidence chunk.'),
    'research': ('Research', 'data/processed/game_note_research/reports', '.md',
                 'Source passages, page citations, and accepted or rejected claims. Model-proposed summaries are audit material and were not forecast inputs.'),
    'tables': ('Tables', 'data/processed/researched_forecaster/table_evidence', '.md',
               'Additional player-table facts calculated by code. Some games have no additional facts in this file; their other evidence is in Inputs and Research.'),
}
STYLE = """
:root{color-scheme:light;--ink:#1d2935;--muted:#536575;--line:#dce3e8;--accent:#176653}
*{box-sizing:border-box}body{margin:0;background:#f4f6f7;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
.wrap{max-width:1160px;margin:auto;padding:32px 28px 70px}h1{font-size:29px;line-height:1.3;margin:12px 0}
h2{font-size:21px}h3{font-size:17px}p{margin:10px 0}.muted{color:var(--muted)}
a{color:var(--accent);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}
nav a{background:white;border:1px solid var(--line);border-radius:7px;padding:8px 18px;text-decoration:none}
nav a[aria-current=page]{background:var(--accent);color:white;border-color:var(--accent)}
.document{background:white;border:1px solid var(--line);border-radius:10px;padding:24px 30px;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0}
blockquote{border-left:3px solid #bdcfc9;margin:16px 0;padding:2px 18px;background:#f7faf9}
input{width:100%;font:inherit;border:1px solid #b8c6cd;border-radius:7px;padding:11px 14px;margin:12px 0}
table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:11px 14px;border-bottom:1px solid var(--line)}
th{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}.links a{margin-right:14px;white-space:nowrap}
@media(max-width:650px){.wrap{padding:18px 12px}.document{padding:18px}th,td{padding:9px 7px}.split{display:none}}
"""


def page(title, content):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{STYLE}</style></head>'
            f'<body><main class="wrap">{content}</main></body></html>').encode()


def load_games():
    rows = [json.loads(line) for line in
            (ROOT/'data/processed/researched_forecaster/inputs.jsonl').read_text().splitlines()]
    games = {}
    for row in rows:
        slug = row['game_id'].replace(':', '-')
        paths = {kind: ROOT/folder/(slug+suffix) for kind, (_, folder, suffix, _) in KINDS.items()}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError('Missing evidence file for '+slug)
        games[slug] = {'label':f"{row['away_abbr']} at {row['home_abbr']}",
                       'date':row['game_date_et'], 'split':row['split'], 'paths':paths}
    return games


def index(games):
    body = ('<p class="muted">NBA forecasting · evidence library</p><h1>Read the game-note evidence</h1>'
            '<p>Choose a game, then open <strong>Inputs</strong> to see exactly what the model received. '
            '<strong>Research</strong> shows the source audit; <strong>Tables</strong> shows additional calculated player statistics.</p>'
            '<label for="filter">Find a team, date, or split</label><input id="filter" type="search" placeholder="e.g. CLE, 2026-02, test">'
            f'<p id="count" class="muted">{len(games)} games</p><table><thead><tr><th>Date</th><th>Game</th>'
            '<th class="split">Split</th><th>Read</th></tr></thead><tbody>')
    for slug, game in games.items():
        body += (f'<tr><td>{game["date"]}</td><td>{html.escape(game["label"])}</td>'
                 f'<td class="split">{game["split"]}</td><td class="links">'+
                 ''.join(f'<a href="/view/{slug}/{kind}">{label}</a>' for kind,(label,*_) in KINDS.items())+'</td></tr>')
    body += """</tbody></table><script>
document.querySelector('#filter').addEventListener('input', event => {
  const q=event.target.value.trim().toLowerCase();let visible=0;
  document.querySelectorAll('tbody tr').forEach(row=>{
    row.hidden=!row.textContent.toLowerCase().includes(q);if(!row.hidden)visible++;
  });document.querySelector('#count').textContent=visible+' games';
});</script>"""
    return page('Game-note evidence', body)


def document(slug, kind, game):
    label, _, _, description = KINDS[kind]
    text = game['paths'][kind].read_text()
    if kind == 'inputs':
        rendered = '<pre id="exact-input">'+html.escape(text)+'</pre>'
    else:
        try:
            from markdown_it import MarkdownIt
            rendered = MarkdownIt('commonmark', {'html':False}).disable('image').render(text)
        except ImportError:
            rendered = '<pre>'+html.escape(text)+'</pre>'
    title = f'{game["label"]} · {game["date"]}'
    body = (f'<a href="/">← All 107 games</a><h1>{html.escape(title)}</h1>'
            f'<p class="muted">{game["split"].capitalize()} set</p><nav>')
    for tab, (name,*_) in KINDS.items():
        active = ' aria-current="page"' if tab == kind else ''
        body += f'<a href="/view/{slug}/{tab}"{active}>{name}</a>'
    body += (f'</nav><p>{html.escape(description)}</p><p><a href="/raw/{slug}/{kind}">Download original {label.lower()} file</a></p>'
             f'<article class="document" data-kind="{kind}">{rendered}</article>')
    return page(label+' · '+title,body)


def handler_for(games):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                content, mime, download = index(games), 'text/html; charset=utf-8', None
            elif path == '/favicon.ico':
                self.send_response(204);self.end_headers();return
            else:
                parts = path.strip('/').split('/')
                if len(parts)!=3 or parts[0] not in ('view','raw') or parts[1] not in games or parts[2] not in KINDS:
                    self.send_error(404);return
                route, slug, kind = parts
                file = games[slug]['paths'][kind]
                content = document(slug,kind,games[slug]) if route=='view' else file.read_bytes()
                mime = 'text/html; charset=utf-8' if route=='view' else 'text/plain; charset=utf-8'
                download = None if route=='view' else f'{slug}-{kind}{file.suffix}'
            self.send_response(200)
            self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'")
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers();self.wfile.write(content)
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();games=load_games()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(games))
    print(f'Evidence viewer: http://127.0.0.1:{args.port} ({len(games)} games)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
