"""Merge GT-row bands without fragmenting full-image tissue components."""
from collections import Counter

import numpy as np

from .retrieval_metrics import score_domain, summarize


COUNT_KEYS = ('tp', 'fp', 'fn', 'positive', 'negative', 'valid')


class RetrievalAccumulator:
    def __init__(self):
        self.counts = [Counter() for _ in range(5)]
        self.areas = Counter()
        self.hits = Counter()
        self.censored = 0
        self.bands = 0

    def update(self, *args, **kwargs):
        result = score_domain(*args, **kwargs)
        for count, row in zip(self.counts, [result['standard'], result['unprompted']] + result['distance']):
            count.update({k: row[k] for k in COUNT_KEYS})
        comp = result['components']
        self.areas.update(dict(zip(comp['ids'], comp['areas'])))
        self.hits.update(dict(zip(comp['ids'], comp['hits'])))
        self.censored += result['censored_target_pixels']
        self.bands += 1

    def finish(self):
        if not self.bands:
            raise ValueError('cannot report an unobserved domain')
        ids = np.array(sorted(self.areas), dtype=np.uint32)
        areas = np.array([self.areas[int(i)] for i in ids], dtype=np.int64)
        hits = np.array([self.hits[int(i)] for i in ids], dtype=np.int64)
        if np.any(hits > areas) or np.any(areas <= 0):
            raise ValueError('invalid merged component areas/hits')
        rows = [summarize({k: c[k] for k in COUNT_KEYS}) for c in self.counts]
        for k in COUNT_KEYS:
            if sum(row[k] for row in rows[2:]) != rows[1][k]:
                raise ValueError('distance bins must partition the entire censored domain')
        if int(areas.sum()) != rows[1]['positive'] or int(hits.sum()) != rows[1]['tp']:
            raise ValueError('full-component counters must conserve target pixels')
        return dict(standard=rows[0], unprompted=rows[1],
                    censored_target_pixels=self.censored,
                    components=dict(ids=ids.tolist(), areas=areas.tolist(), hits=hits.tolist(),
                        count=len(ids), mean_area_recall=float(np.mean(hits/areas)) if len(ids) else None,
                        detected_at_50=int(np.count_nonzero(2*hits >= areas)),
                        detected_at_10=int(np.count_nonzero(10*hits >= areas))),
                    distance=[dict(bin=name, **row) for name, row in zip(('<1mm', '1-5mm', '>=5mm'), rows[2:])])
