import sys

# Function to count common neighbors
def common_neighbors(a, b, edge_file):
    ha = set()
    hb = set()
    i = 0
    with open(edge_file, 'r') as f:
        for line in f:
            x, y = line.strip().split('\t')
            if x == a:
                ha.add(y)
            elif y == a:
                ha.add(x)
            if x == b:
                hb.add(y)
            elif y == b:
                hb.add(x)
    return len(ha.intersection(hb))

# Read degrees from nodeinfo file
degrees = {}
with open(sys.argv[2], 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split('\t')
        degrees[parts[0]] = int(parts[3])

# Process pairs from the third argument file
with open(sys.argv[3], 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        overlap = common_neighbors(parts[0], parts[1], sys.argv[1])
        print(f"{parts[0]}\t{parts[1]}\t", end='')
        if parts[0] in degrees:
            gene1c = overlap / degrees[parts[0]]
            print(f"{degrees[parts[0]]}\t", end='')
        else:
            print("na\t", end='')
            gene1c = "na"
        if parts[1] in degrees:
            gene2c = overlap / degrees[parts[1]]
            print(f"{degrees[parts[1]]}\t", end='')
        else:
            print("na\t", end='')
            gene2c = "na"
        print(f"{overlap}\t{gene1c}\t{gene2c}")

