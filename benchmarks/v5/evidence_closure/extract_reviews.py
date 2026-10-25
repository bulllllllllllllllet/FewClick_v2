"""Extract the supplied local OpenReview snapshot without accessing an account."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup

REVIEWERS={'HE9aa44qX3':('ZVeE','human',3,4),'3JnoqyoImO':('wHd8','human',5,4),
           'tE8ABfcLZ6':('HVPZ','human',7,3),'HSWqn6ONHh':('AI Review','AI',None,None)}
HEADINGS=['Paper Summary:','Strengths:','Weaknesses:','Justification of Recommendation:',
          'Specific Points of Feedback for Rebuttal:','Rating:','Confidence:','Review:']


def extract(source,output):
    output.mkdir(parents=True,exist_ok=False)
    raw=source.read_bytes();soup=BeautifulSoup(raw.decode('utf-8'),'html.parser');notes=[]
    for identity,(reviewer,kind,rating,confidence) in REVIEWERS.items():
        matches=soup.select(f'.note[data-id="{identity}"]')
        if len(matches)!=1:raise ValueError('missing/duplicate original review '+identity)
        text=matches[0].get_text('\n',strip=True);lines=text.splitlines();markers=[]
        for index,line in enumerate(lines):
            if line in HEADINGS:markers.append((index,line[:-1]))
        sections={}
        for number,(index,title) in enumerate(markers):
            end=markers[number+1][0] if number+1<len(markers) else len(lines)
            sections[title]='\n'.join(lines[index+1:end])
        if kind=='human':
            actual_rating=int(re.match(r'\d+',sections['Rating']).group())
            actual_confidence=int(re.match(r'\d+',sections['Confidence']).group())
            if (actual_rating,actual_confidence)!=(rating,confidence):raise ValueError('source rating changed')
            if not all(sections.get(name) for name in ('Weaknesses','Specific Points of Feedback for Rebuttal')):
                raise ValueError('review challenges missing')
        target=output/f'{reviewer.replace(" ","_")}_{identity}.txt';target.write_text(text+'\n')
        notes.append(dict(note_id=identity,reviewer=reviewer,authority=kind,rating=rating,confidence=confidence,
            saved_text=str(target),text_sha256=hashlib.sha256(text.encode()).hexdigest(),sections=sections,
            original_note_url=f'https://openreview.net/forum?id=sjms3c2fJZ&noteId={identity}'))
    if len(notes)!=4 or len({n['note_id'] for n in notes})!=4:raise ValueError('review extraction incomplete')
    payload=dict(status='complete_extraction',source=str(source.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
        notes=notes,excluded_administrative_note='SvM4Cdf0Cw: Phase1 update, not a scientific review',
        scope='supplied saved HTML; no live OpenReview login, submission, or paper-version binding',
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'reviews.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    # Disk replay validates verbatim preservation, distinct from interpretation.
    reread=json.loads((output/'reviews.json').read_text())
    for note in reread['notes']:
        text=Path(note['saved_text']).read_text().removesuffix('\n')
        if hashlib.sha256(text.encode()).hexdigest()!=note['text_sha256']:raise ValueError('review text reload mismatch')
    print(json.dumps(dict(status='complete_extraction',human_reviews=3,AI_reviews=1,source_sha256=payload['source_sha256'],output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--html',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();extract(a.html,a.output)
