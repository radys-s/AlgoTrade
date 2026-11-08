"""Write the mirrored tape of an arena CSV: sides B<->A, price 20000 - p (a stress test for one-sided tuning).

    uv run python -m research.diag.mirror_tape data/round3_public.csv data/round3_mirror.csv
"""

from __future__ import annotations

import sys


def main(src: str, dst: str) -> None:
    with open(src, encoding='utf-8') as fi, open(dst, 'w', encoding='utf-8', newline='') as fo:
        fo.write(fi.readline())
        for line in fi:
            t, op, oid, side, vol, price, life = line.rstrip('\r\n').split(',')
            if op == 'Insert':
                side = 'A' if side == 'B' else 'B'
                price = str(20000 - int(price))
            fo.write(f'{t},{op},{oid},{side},{vol},{price},{life}\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
