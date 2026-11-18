import json
import subprocess

grid = {'uncap_unwind': [0, 20, 40, 60, 80]}
cmd = ['uv', 'run', 'python', '-X', 'utf8', '-m', 'research.run', '--strategy', 'strategies/exp_turnover.py', '--data', 'data/round1.csv', '--grid', json.dumps(grid), '--workers', '8', '--tag', 'r1-turnover']
subprocess.run(cmd)
