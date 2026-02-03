import sys
from collections import Counter

def startable_values(a):
    n = len(a)
    ok = set()
    for i in range(n):
        x = a[i]
        y = a[(i + 1) % n]
        z = a[(i + 2) % n]
        ok.add(min(x, y, z))
        ok.add(max(x, y, z))
    return ok

def solve():
    n_line = sys.stdin.readline().strip()
    n = int(n_line)
    a = list(map(int, sys.stdin.readline().split()))

    freq = Counter(a)
    ok = startable_values(a)

    res = []
    for x in a:
        if freq[x] == n:
            res.append("0")
        else:
            extra = 0 if x in ok else 1
            res.append(str((n - freq[x]) + extra))

    sys.stdout.write(" ".join(res))

solve()