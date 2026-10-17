"""Generate notebook/smoke/autosearch_smoke.ipynb (Phase-3 bounded smoke)."""
import ast
import json
from pathlib import Path


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": list(src)}


def code(text):
    ast.parse(text)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


CELL1 = '''# 1. Clone project + deps (fail fast on git/pip problems).
import os, subprocess, sys
from pathlib import Path
WORK = Path('/kaggle/working')
PROJ = WORK / 'autosearch_kaggle'
os.chdir(WORK)  # never rmtree the directory the kernel is currently sitting in
if PROJ.exists():
    import shutil; shutil.rmtree(PROJ)
subprocess.run(['git', 'clone', '-q',
                'https://github.com/MokshonWork/autosearch_kaggle.git',
                str(PROJ)], check=True)
os.chdir(PROJ)
sys.path.insert(0, str(PROJ))
print(subprocess.run(['git', 'log', '--oneline', '-2'],
                     capture_output=True, text=True).stdout)
print(subprocess.run(['nvidia-smi', '--list-gpus'],
                     capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'rustbpe>=0.1.0', 'tiktoken>=0.11.0'], check=True)
# Informational only: the kernel's own torch is not used for training
# (subprocesses import torch themselves); never block on it.
try:
    import torch
    print('torch', torch.__version__, 'devices', torch.cuda.device_count())
except Exception as exc:
    print('kernel-process torch check failed (non-fatal):',
          type(exc).__name__, str(exc)[:200])
    import importlib.util
    print('torch spec:', importlib.util.find_spec('torch'))
'''

CELL2 = '''# 2. Secret — HARD requirement for the smoke (Gemini calls happen). Fail fast.
# Retries a few times: the Kaggle Secrets backend has been flaky (HTTP 400).
import os, time
from kaggle_secrets import UserSecretsClient
key = None
last_exc = None
for attempt in range(4):
    try:
        key = UserSecretsClient().get_secret('GEMINI_API_KEY')
        break
    except Exception as exc:
        last_exc = exc
        print(f'secret fetch attempt {attempt + 1} failed:',
              type(exc).__name__, str(exc)[:160])
        time.sleep(15)
if key is None:
    print('GEMINI_API_KEY unfetchable after retries; cannot run the smoke')
    raise last_exc
assert key, 'GEMINI_API_KEY secret missing or empty'
os.environ['GEMINI_API_KEY'] = key
del key
print('GEMINI_API_KEY loaded:', bool(os.environ.get('GEMINI_API_KEY')))
'''

CELL2B = '''# 2b. Verify every configured Gemini model id responds (assumption A6:
# "verify current IDs on run day"). Fail fast BEFORE any expensive step.
import yaml, requests as rq
from pathlib import Path
cfg = yaml.safe_load(Path('config.smoke.yaml').read_text())
ids = sorted({v for v in cfg['llm'].values()
              if isinstance(v, str) and v.startswith('gemini-')})
print('probing configured model ids:', ids)
key_hdr = {'x-goog-api-key': os.environ['GEMINI_API_KEY']}
for mid in ids:
    r = rq.post(
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{mid}:generateContent',
        json={'contents': [{'parts': [{'text': 'Reply with the word: ok'}]}]},
        headers=key_hdr, timeout=30)
    print(f'  {mid} -> HTTP {r.status_code}')
    assert r.status_code == 200, (
        f'model id {mid} unusable (HTTP {r.status_code}); '
        'update llm model ids in config, re-push, re-run')
print('all configured model ids OK')
'''

CELL3 = '''# 3. Nested repo/.git + data prep (skips if cache already present).
import subprocess, sys
from pathlib import Path
repo = Path('/kaggle/working/autosearch_kaggle/repo')
if not (repo / '.git').exists():
    for a in (['init', '-q'], ['add', '-A'],
              ['-c', 'user.email=k@kaggle', '-c', 'user.name=k',
               'commit', '-q', '-m', 't4-port baseline']):
        subprocess.run(['git', '-C', str(repo), *a], check=True)
    print('repo/.git re-initialized')
r = subprocess.run([sys.executable, 'prepare.py', '--num-shards', '4'],
                   cwd=str(repo), capture_output=True, text=True, timeout=3600)
print(r.stdout[-1200:])
print('prepare exit:', r.returncode)
assert r.returncode == 0, 'dataset/tokenizer prep failed'
'''


CELL4 = '''# 4. The bounded smoke run itself. Full output saved to log files.
import subprocess, sys
from pathlib import Path
out = Path('/kaggle/working/smoke_results')
out.mkdir(parents=True, exist_ok=True)
p = subprocess.run(
    [sys.executable, 'evals/run_ab.py',
     '--config', 'config.smoke.yaml',
     '--repeats', '1', '--arms', 'multi_agent',
     '--out', str(out)],
    cwd='/kaggle/working/autosearch_kaggle',
    capture_output=True, text=True, timeout=4500)
(out / 'run_ab_stdout.log').write_text(p.stdout)
(out / 'run_ab_stderr.log').write_text(p.stderr)
print('run_ab exit:', p.returncode)
print('--- stdout tail ---')
print(p.stdout[-3000:])
print('--- stderr tail ---')
print(p.stderr[-2000:])
assert p.returncode == 0, 'run_ab failed — see logs above'
'''

CELL5 = '''# 5. Validate wiring evidence: state, experiments, budgets, metrics.
import json, math
from pathlib import Path
out = Path('/kaggle/working/smoke_results')
mem = out / 'multi_agent' / 'run0' / 'memory'
state = json.loads((mem / 'state.json').read_text())
exps = [json.loads(l) for l in
        (mem / 'experiments.jsonl').read_text().splitlines() if l.strip()]
print('state:', json.dumps({k: state.get(k) for k in (
    'round', 'experiments_done', 'best_val_bpb', 'baseline_bpb',
    'epsilon', 'stopped_reason')}, indent=2))
for e in exps:
    r = e['result']
    hid = e['hypothesis']['id']
    print(f"{hid} gpu{e['worker']} {r['status']} "
          f"val_bpb={r['val_bpb']} steps={r['steps']} wall={r['wall_seconds']}s")
checks = {
    'epsilon_set': state.get('epsilon') is not None,
    'baseline_finite': (state.get('baseline_bpb') is not None
                        and math.isfinite(state['baseline_bpb'])
                        and state['baseline_bpb'] > 0),
    'budget_stop': state.get('stopped_reason') in
                   ('max_rounds', 'max_experiments', 'no_improvement'),
    'api_calls_made': state.get('api_calls', 0) > 0,
    'api_budget_respected': state.get('api_calls', 0) <= 30,
    'at_least_one_ok_run': any(
        e['result']['status'] == 'ok'
        and e['result']['val_bpb'] is not None
        and math.isfinite(e['result']['val_bpb'])
        and e['result']['val_bpb'] > 0
        and e['result']['steps'] > 0 for e in exps),
    'gpu_ids_valid': all(e['worker'] in (0, 1) for e in exps),
    'ab_report_exists': (out / 'ab_report.md').exists(),
}
summary = {
    'checks': checks,
    'state': {k: state.get(k) for k in (
        'round', 'experiments_done', 'baseline_bpb', 'best_val_bpb',
        'epsilon', 'calibration_values', 'stopped_reason', 'api_calls',
        'api_tokens')},
    'experiments': [f"{e['hypothesis']['id']}:{e['result']['status']}"
                    for e in exps],
}
(out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(checks, indent=2))
failed = [k for k, v in checks.items() if not v]
assert not failed, f'smoke checks failed: {failed}'
print('SMOKE_COMPLETE_ALL_CHECKS_PASS')
'''

cells = [
    md("# autosearch smoke — bounded Phase-3 smoke test (GPU T4 x2)\n",
       "\n",
       "One arm (multi_agent), one repeat, config.smoke.yaml: calibration + up to\n",
       "2 experiments with a 10s training budget. Validates WIRING, not science.\n",
       "Expected wall time ~25-40 min (incl. data prep). Requires the\n",
       "GEMINI_API_KEY secret (checked BEFORE any expensive step)."),
    code(CELL1), code(CELL2), code(CELL2B), code(CELL3), code(CELL4), code(CELL5),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.13"}},
      "nbformat": 4, "nbformat_minor": 5}
outp = Path(__file__).resolve().parent / "autosearch_smoke.ipynb"
outp.write_text(json.dumps(nb, indent=1))
json.load(open(outp))
print("wrote", outp, outp.stat().st_size)
