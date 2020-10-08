#this code crawl the WebQuestion datasets github repo
import requests
import json
import urllib
import unidecode # convert unicode string to ascii string

url = "https://raw.githubusercontent.com/ysu1989/GraphQuestions/master/freebase13/graphquestions.testing.json"
req = requests.get(url)
print(req)
data = json.loads(req.content)


result = {}
key = str(data[0]['question'])
qid = data[0]['graph_query']['nodes'][0]['friendly_name']

result[key] = []

for t in data[1:]:
    if t['graph_query']['nodes'][0]['friendly_name'] == qid:
        result[key].append(unidecode.unidecode(t['question']))
    else:
        key = unidecode.unidecode(t['question'])
        qid = t['graph_query']['nodes'][0]['friendly_name']
        result[key] = []

f = open("./web_Question_crawled_dataset.csv","a")

f.write("question\tnumber of paraphrases\tparaphrases\n")
for k,v in result.items():
    f.write(k+"\t"+str(len(v)))
    for e in v:
        f.write("\t"+e)
    f.write("\n")

f.close()