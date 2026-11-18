import json
import subprocess

fixed = {'ml_thr': 0.105, 'ml_stub': 3, 'ml_tight_only': 1, 'park_max': 5, 'park_size': 20}
grid = {'fresh_min_level': [0, 50, 150, 300], 'fresh_stub': [3, 5]}
cmd = ['uv', 'run', 'python', '-X', 'utf8', '-m', 'research.run', '--strategy', 'strategies/exp_overcap.py', '--data', 'data/round1.csv', '--set', json.dumps(fixed), '--grid', json.dumps(grid), '--workers', '8', '--tag', 'r1-fresh-gate']
subprocess.run(cmd)
