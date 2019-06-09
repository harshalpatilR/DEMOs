import pandas as pd
import gzip
import sys

# no of reviews to extract
get_records = 30000
picklef = "amazon_reviews_limited.pickle"

def parse(path): 
	with gzip.open(path, 'r') as g:
		for l in g: 
			yield eval(l) 

def getDF(path): 

	i = 0 
	df = {} 
	for d in parse(path): 
		df[i] = d 
		#print sys.getsizeof(d)
		i += 1 
		if(i==get_records):
			break
	
	return pd.DataFrame.from_dict(df, orient='index')


df = getDF('aggressive_dedup.json.gz')

print "\n RAW"
print df.head()
print df.tail()

#code text and label field
df["textvec"] = df["summary"]+" "+df["reviewText"]
df["labelvec"] = [1 if x<3.0 else 0 for x in df["overall"]]

print "\n Labeled"
print df.head()
print df.tail()

df.to_pickle(picklef,compression=None)

