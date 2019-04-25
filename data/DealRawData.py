import re
sourcefile = open("train2019.txt","r",encoding="UTF-8")
targetfile = open("origin.txt","w",encoding='UTF-8')
lines = sourcefile.readlines()
for item in lines:
    if not re.search("q[0-9]*:",item) is None:
        originstr = item
        endpos = re.search("q[0-9]*:", originstr).end()
        targetstr = " ".join(originstr[endpos:].rstrip("？?")).rstrip(" ")
        targetfile.write(targetstr+"\n")



