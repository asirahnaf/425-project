"""Build T1-top50 dataset: MusicCaps 5521 captions -> MTT top-50 keyword proxy.
ADDITIVE: writes mtt50_clips.csv + mtt_top50_kept.txt. Does NOT touch mtt_clips.csv (20-tag, Task4 dep).
"""
import ast, csv
from collections import Counter
from pathlib import Path

# tag -> caption keyword variants (from real MTT top-50)
VARIANTS = {
 "guitar": ["guitar"],
 "classical": ["classical", "orchestral", "orchestra", "baroque"],
 "slow": ["slow"],
 "techno": ["techno"],
 "strings": ["strings", "string section", "string ensemble"],
 "drums": ["drum", "drums", "drumming", "percussion"],
 "electronic": ["electronic", "electronica", "electro", "trance", "house music"],
 "rock": ["rock"],
 "fast": ["fast", "uptempo", "up-tempo", "energetic", "upbeat tempo"],
 "piano": ["piano"],
 "ambient": ["ambient"],
 "beat": ["beat"],
 "violin": ["violin", "fiddle"],
 "vocal": ["vocal"],
 "synth": ["synth"],
 "female": ["female", "woman", "girl"],
 "indian": ["indian", "india", "sitar", "raga"],
 "opera": ["opera", "operatic"],
 "male": ["male", "man ", "men "],
 "singing": ["singing", "sings", "singer"],
 "vocals": ["vocals"],
 "no vocals": ["no vocal", "no singing", "instrumental"],
 "harpsichord": ["harpsichord", "harpsicord", "cembalo"],
 "loud": ["loud", "noisy", "aggressive"],
 "quiet": ["quiet"],
 "flute": ["flute", "flutes"],
 "woman": ["woman"],
 "male vocal": ["male vocal", "male singer", "man singing"],
 "pop": ["pop"],
 "no vocal": ["no vocal"],
 "soft": ["soft", "mellow", "gentle", "calm"],
 "sitar": ["sitar"],
 "solo": ["solo"],
 "man": ["man "],
 "classic": ["classic"],
 "choir": ["choir", "choral", "chorus"],
 "voice": ["voice"],
 "new age": ["new age"],
 "dance": ["dance", "danceable"],
 "male voice": ["male voice"],
 "female vocal": ["female vocal"],
 "beats": ["beats"],
 "harp": ["harp"],
 "cello": ["cello"],
 "no voice": ["no voice"],
 "weird": ["weird", "strange", "odd"],
 "country": ["country"],
 "female voice": ["female voice"],
 "metal": ["metal"],
 "choral": ["choral"],
}

def parse_aspect(s):
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).lower() for x in v]
    except Exception:
        pass
    return [s.lower()]

def main():
    root = Path(r'C:\Users\asira\Downloads\425_assignment\gnn-bert-music-context')
    rows = list(csv.DictReader(open(root/'data'/'raw'/'musiccaps-public.csv', encoding='utf-8')))
    print('MusicCaps rows:', len(rows))
    cnt = Counter(); parsed = []
    for r in rows:
        blob = (r['caption'] + ' ' + ' '.join(parse_aspect(r['aspect_list']))).lower()
        hits = {t: any(v in blob for v in vs) for t, vs in VARIANTS.items()}
        parsed.append((r['ytid'], r['caption'], hits))
        for t, h in hits.items():
            if h: cnt[t] += 1
    print('support (top50 order):')
    for t in VARIANTS: print(f'  {t}: {cnt[t]}')
    kept = [t for t in VARIANTS if cnt[t] >= 50]
    print(f'kept {len(kept)}/50 at support>=50')
    dropped = [t for t in VARIANTS if cnt[t] < 50]
    print('dropped:', dropped)
    dst = root/'data'/'raw'/'mtt50_clips.csv'
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['clip_id', 'text'] + [f'tag_{t}' for t in VARIANTS])
        w.writeheader()
        for ytid, cap, hits in parsed:
            w.writerow({'clip_id': ytid, 'text': cap, **{f'tag_{t}': int(bool(hits[t])) for t in VARIANTS}})
    print(f'wrote {dst} ({len(parsed)} clips x {len(VARIANTS)} tags, all 50 kept regardless of support)')
    open(root/'data'/'raw'/'mtt_top50_kept.txt', 'w', encoding='utf-8').write('\n'.join(VARIANTS))

if __name__ == '__main__':
    main()
