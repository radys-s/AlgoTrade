# COPILOT gap audit

Status: the server-side artifact is still absent, so the local-vs-server gap audit remains blocked until the server result file appears.

| Check | Local evidence | Server evidence | Interpretation |
| --- | --- | --- | --- |
| Artifact presence | the server directory is still missing | absent | no server-side cross-check yet |
| Base strategy | `mm_r2_fq2.py` is the uploaded model and the local base reproduces 349,077 on round2 | no server result file found | local ranking cannot be validated yet |
| Sweep evidence | best local R2 candidate so far is [20261107-222544_radys-s_r2-radys-fq5-spread.csv](../research/results/20261107-222544_radys-s_r2-radys-fq5-spread.csv) at 360,791 | absent | the local best remains directional only |
| Fill audit | local fill logs and ledger metrics are available | not available | no server-side fill comparison can be made |
| Risk | local sweeps are useful for tuning but not rank selection | unavailable | a server-confirmed result is required before any final claim |

Conclusion: the server artifact is still missing, so there is no trustworthy cross-check of the observed local gap. 
Conclusion: the strongest local run is 360,791 on the candidate file above, but that value is not server-backed and cannot be used for ranking. 
Conclusion: the C2 sweep is still running inside the allowed bounds, and the outputs are stored under research/results for later comparison. 
Conclusion: until the server result arrives, local results should be treated as directional only and not as proof of a champion. 
Conclusion: once the server artifact appears, I will compare the same run settings directly and update the ranking before any claim is made.
