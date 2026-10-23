"""
List (do NOT download) the files in DF40's public Google Drive folders, so
you can pick only the methods NOEFS needs and check sizes before downloading.

Usage:  python repro/df40_list.py [json|train|test|all] > df40_files.tsv

Folder links are the ones in the DF40 README (github.com/YZY-stack/DF40,
section "Download DF40 data"). Written for gdown 6.x (no `remaining_ok`).
If a folder listing looks incomplete, open the link in a browser.
"""
import sys

import gdown

FOLDERS = {
    'json': 'https://drive.google.com/drive/folders/19VhAL4aDJOKvhl9stEq_ymFeHiXo6_j-',
    'train': 'https://drive.google.com/drive/folders/1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD',
    'test': 'https://drive.google.com/drive/folders/1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_',
}
# FF++ real frames (single file, train + test)
FFPP_REAL_ID = '1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb'

which = sys.argv[1] if len(sys.argv) > 1 else 'all'
names = list(FOLDERS) if which == 'all' else [which]
print('folder\tpath\tfile_id')
for name in names:
    try:
        files = gdown.download_folder(url=FOLDERS[name], skip_download=True,
                                      quiet=True)
    except Exception as e:  # Drive rate limits, permission changes, ...
        print(f'{name}\tERROR\t{e}', file=sys.stderr)
        continue
    for f in files or []:
        print(f'{name}\t{f.path}\t{f.id}')
print(f'real\tFF++ real frames (single file)\t{FFPP_REAL_ID}')
