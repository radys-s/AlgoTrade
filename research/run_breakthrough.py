import json
import subprocess

grid = {'favor_boost': [1.0, 1.2, 1.5, 2.0], 'layer_size': [0, 20, 50]}
cmd = ['uv', 'run', 'python', '-X', 'utf8', '-m', 'research.run', '--strategy', 'strategies/exp_breakthrough.py', '--data', 'data/round1.csv', '--grid', json.dumps(grid), '--workers', '8', '--tag', 'r1-breakthrough']
subprocess.run(cmd)
