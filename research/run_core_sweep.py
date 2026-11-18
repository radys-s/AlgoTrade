import json
import subprocess

grid = {'skew_per_lot': [0.0, 0.005, 0.01, 0.02], 'quote_size': [100, 150, 200]}
cmd = ['uv', 'run', 'python', '-X', 'utf8', '-m', 'research.run', '--strategy', 'strategies/mm_r1_final.py', '--data', 'data/round1.csv', '--grid', json.dumps(grid), '--workers', '8', '--tag', 'r1-core-skew']
subprocess.run(cmd)
