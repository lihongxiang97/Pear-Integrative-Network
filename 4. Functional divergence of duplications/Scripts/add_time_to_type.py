import sys
d = {}
with open(sys.argv[1]) as f:
    for lines in f:
        line = lines.strip().split()
        if line[2] != 'NA':
            d[line[0]+'\t'+line[1]] = line[2]
with open(sys.argv[2]) as f:
    for lines in f:
        line = lines.strip().split()
        pair = line[0]+'\t'+line[1]
        if len(line) >= 4:
            if pair in d:
                if 0< float(d[pair]) < 0.16:
                    print(pair,line[4],'After single WGD',sep='\t')
                if 0.16<= float(d[pair]) <1.5:
                    print(pair,line[4],'After gamma WGD',sep='\t')
                if float(d[pair]) >=1.5:
                    print(pair,line[4],'Before gamma WGD',sep='\t')
