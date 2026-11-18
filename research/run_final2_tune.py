import json
import subprocess

grid = {'quote_size': [120, 150, 180, 200], 'park_max': [6, 8, 10, 12]}
cmd = ['uv', 'run', 'python', '-X', 'utf8', '-m', 'research.run', '--strategy', 'strategies/mm_r1_final2.py', '--data', 'data/round1.csv', '--grid', json.dumps(grid), '--workers', '8', '--tag', 'r1-final2-grid']
subprocess.run(cmd)
