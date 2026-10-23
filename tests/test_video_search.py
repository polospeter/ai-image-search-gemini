import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from index_videos import index_video
from video_search import sample_frames, search_video_frames


class SearchTests(unittest.TestCase):
    def test_sampling_covers_middle_and_end(self):
        self.assertEqual(sample_frames(151, 30, 2), [0, 60, 120, 150])
        self.assertEqual(sample_frames(1, 30, 2), [0])
        self.assertEqual(sample_frames(5, 30, .001), [0, 1, 2, 3, 4])
        for fps, interval in [(0, 2), (float('nan'), 2), (30, -1), (30, float('inf'))]:
            with self.assertRaises(ValueError):
                sample_frames(100, fps, interval)

    def test_expands_past_duplicate_video_frames(self):
        rows = [('a.mp4', i, .01 + i / 1000) for i in range(12)] + [('b.mp4', 4, .1)]
        calls = []

        def query(**kwargs):
            calls.append(kwargs['n_results'])
            selected = rows[:kwargs['n_results']]
            return {'ids': [[str(i) for i in range(len(selected))]],
                    'metadatas': [[{'filename': fn, 'timestamp': t} for fn, t, d in selected]],
                    'distances': [[d for fn, t, d in selected]]}

        collection = NS(count=lambda: len(rows), query=query)
        result = search_video_frames(collection, [1], 2, 35)
        self.assertEqual([r['filename'] for r in result], ['a.mp4', 'b.mp4'])
        self.assertEqual(result[1]['timestamp'], 4)
        self.assertEqual(calls, [6, 12, 13])
        self.assertEqual([r['filename'] for r in search_video_frames(collection, [1], 2, 35, {'a.mp4'})], ['b.mp4'])
        self.assertEqual(len(search_video_frames(collection, [1], 2, 95)), 1)

    def test_empty_and_legacy_indexes(self):
        self.assertEqual(search_video_frames(NS(count=lambda: 0), [1], 2, 35), [])
        collection = NS(count=lambda: 1, query=lambda **kw: {
            'ids': [['old.mp4']], 'metadatas': [[{'filename': 'old.mp4'}]], 'distances': [[.1]],
        })
        self.assertEqual(search_video_frames(collection, [1], 2, 35)[0]['timestamp'], 0)


class IndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_migration_resume_and_interval_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.mp4'
            path.write_bytes(b'fake video')
            rows = {'clip.mp4': {'filename': 'clip.mp4'}}
            calls, failures = [], set()
            class Capture:
                def __init__(self, path): self.index = 0
                def isOpened(self): return True
                def get(self, key): return {1: 2, 2: 10}[key]
                def set(self, key, value): self.index = value
                def read(self): return True, self.index
                def release(self): pass
            def write_frame(path, frame, options):
                Path(path).write_bytes(str(frame).encode())
                return True
            cv2 = NS(VideoCapture=Capture, CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2,
                     CAP_PROP_POS_FRAMES=3, IMWRITE_JPEG_QUALITY=4, imwrite=write_frame)
            types = NS(Part=NS(from_bytes=lambda **kw: kw['data']))
            async def embed_content(**kwargs):
                frame = int(kwargs['contents'][1])
                calls.append(frame)
                if frame in failures: raise RuntimeError('failed embedding')
                return NS(embeddings=[NS(values=[1., 0.])])
            class Collection:
                def get(self, **kw): return {'ids': list(rows), 'metadatas': list(rows.values())}
                def upsert(self, ids, embeddings, metadatas): rows.update(zip(ids, metadatas))
                def delete(self, ids):
                    for key in ids: del rows[key]
            client = NS(aio=NS(models=NS(embed_content=embed_content)))
            async def run(interval):
                return await index_video(path, interval, Path(tmp), Collection(), client,
                                         asyncio.Semaphore(2), NS(write=lambda message: None))
            with patch.dict('sys.modules', {'cv2': cv2, 'google': NS(genai=NS(types=types)),
                                           'google.genai': NS(types=types)}):
                failures.add(4)
                self.assertFalse(await run(2.0))
                self.assertIn('clip.mp4', rows)  # old index survives failed upgrade
                self.assertEqual(len(rows), 4)  # three successful new frames are resumable
                failures.clear()
                calls.clear()
                self.assertTrue(await run(2.0))
                self.assertEqual(calls, [4])  # retry only failed frame
                self.assertNotIn('clip.mp4', rows)
                self.assertEqual(sorted(m['timestamp'] for m in rows.values()), [0, 2, 4, 4.5])
                calls.clear()
                self.assertTrue(await run(2.0))
                self.assertEqual(calls, [])
                self.assertTrue(await run(1.0))
                self.assertEqual(len(rows), 6)  # changed interval replaces old generation


if __name__ == '__main__':
    unittest.main()
