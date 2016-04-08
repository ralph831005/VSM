import sys
import glob

with open(sys.argv[1]) as fp:
    lines = ''.join([x.strip() for x in fp.readlines()])
    title = lines.split('<title>')[1].split('</title>')[0].strip()
    p = ''.join([x.split('</p>')[0] for x in lines.split('<p>')[1:]])
print(len(title+p))

