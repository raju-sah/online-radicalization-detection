#!/usr/bin/env python
"""Build allowlisted code and manuscript-source archives for author review.

A complete build requires the PDF exported from the current LaTeX editor. This
script never compiles another document, uploads anything, or includes data/logs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
ROOT_FILES = ["AGENTS.md", "README.md", "PREREGISTRATION.md", "DEVIATIONS.md", "ROADMAP.md",
              "PROGRESS.md", "LICENSE", "CITATION.cff", "requirements.txt", "requirements-legacy.txt",
              "requirements-release-lock.txt", "pytest.ini", ".env.example", ".gitignore", ".zenodo.json"]
MANUSCRIPT_FILES = ["paper/main.tex", "paper/LICENSE.md", "paper/generated/numbers.tex",
                    "paper/generated/tab_release.tex", "paper/generated/tab_balance.tex",
                    "paper/generated/tab_release_robust.tex", "paper/generated/tab_streams.tex",
                    "paper/figures/release_reliability.png", "paper/figures/release_importance.png"]


def sha256(path: Path) -> str:
    """Compute an artifact's checksum without capturing its contents in logs."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def archive(destination: Path, names: list[str]) -> None:
    """Write a deterministic archive from a fixed list; reject symlinks."""
    pending = destination.with_suffix('.tmp')
    try:
        with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED) as out:
            for name in sorted(set(names)):
                path=ROOT/name
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f'Invalid allowlisted release path: {name}')
                item=zipfile.ZipInfo(name,date_time=(2026,10,24,0,0,0))
                item.compress_type=zipfile.ZIP_DEFLATED
                item.external_attr=0o644<<16
                out.writestr(item,path.read_bytes())
        pending.replace(destination)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def main() -> None:
    """Package reviewed sources without expanding a whole-workspace archive."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-only',action='store_true',help='Mark this package incomplete; omit PDF.')
    args=ap.parse_args()
    pdf=ROOT/'paper/main.pdf'
    if not args.source_only and not pdf.is_file():
        raise SystemExit('STOP: export the compiled, visually checked PDF from the current editor first. Use --source-only only for an incomplete review package.')
    # Refuse old generated inputs after a manuscript revision.
    for name in MANUSCRIPT_FILES:
        if not (ROOT/name).is_file():
            raise SystemExit(f'STOP: missing current paper input {name}; run scripts/make_paper.py.')
    code_names=list(ROOT_FILES)
    for folder,pattern in [('src','*.py'),('scripts','*.py'),('tests','*.py'),('configs','*.yaml')]:
        code_names += [str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob(pattern)]
    code_names += ['release/RELEASE.md','release/environment.json','paper/LICENSE.md']
    output=ROOT/'output/release'
    output.mkdir(parents=True,exist_ok=True)
    code=output/f'radpath-code-{VERSION}.zip'
    source=output/f'radpath-manuscript-source-{VERSION}.zip'
    archive(code,code_names)
    archive(source,MANUSCRIPT_FILES)
    metadata=output/'zenodo-metadata.json'
    shutil.copyfile(ROOT/'.zenodo.json',metadata)
    products=[code,source,metadata]
    if not args.source_only:
        published_pdf=output/f'radpath-preprint-{VERSION}.pdf'
        shutil.copyfile(pdf,published_pdf)
        products.append(published_pdf)
    elif (output/f'radpath-preprint-{VERSION}.pdf').exists():
        raise SystemExit('STOP: source-only output contains a stale PDF; choose a clean output directory.')
    info={'version':VERSION,'status':'incomplete-source-review' if args.source_only else 'requires-author-review',
          'pdf_included':not args.source_only,
          'excluded':['raw Reddit data','user-level tables','hashes','record IDs','text','embeddings','predictions','logs','secrets','Git history'],
          'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)} for p in products],
          'archive_members':{code.name:sorted(code_names),source.name:sorted(MANUSCRIPT_FILES)}}
    manifest=output/'manifest.json'
    manifest.write_text(json.dumps(info,indent=2)+'\n')
    products.append(manifest)
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in products))
    print(f'prepared {info["status"]}: {len(products)} files plus checksum list')


if __name__=='__main__':
    main()
