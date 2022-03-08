import pstats
from pstats import SortKey
p = pstats.Stats('out2.prof')
p.sort_stats(SortKey.TIME).print_stats(100)
