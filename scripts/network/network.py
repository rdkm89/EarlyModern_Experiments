import urllib.request
url = "http://www.gutenberg.org/cache/epub/1661/pg1661.txt"
string = urllib.request.urlopen(url).read().decode()
len(string)

startText = string.find("THE ADVENTURES OF SHERLOCK HOLMES")
endText = string.find("End of the Project Gutenberg EBook of The Adventures of Sherlock Holmes")
filteredText = string[startText:endText]
len(filteredText)

import re
paragraphs = re.split(r'\r\n\r\n+', filteredText)

import nltk
nltk.download('words')

def findPeople(tree, people):
    if type(tree) is nltk.tree.Tree and tree.label() == "PERSON":
        people.append(" ".join([word for word, pos in tree]))
    elif (type(tree) is nltk.tree.Tree) or (type(tree) is list):
        [findPeople(branch, people) for branch in tree]

multi_people = []
for paragraph in paragraphs:
    sentences = nltk.sent_tokenize(paragraph)
    tokenizedSentences = [nltk.word_tokenize(sent) for sent in sentences]
    taggedSentences = [nltk.pos_tag(sent) for sent in tokenizedSentences]
    chunkedSentences = [nltk.ne_chunk(sent) for sent in taggedSentences]
    people = []
    findPeople(chunkedSentences, people)
    people = ["Holmes" if ("Holmes" in person or "Sherlock" in person) else person for person in people]
    people = [re.sub("(Mr\.|Miss)", "", person) for person in people]
    people = [person.strip() for person in people if person.strip()]
    people = set(people)
    if len(people) > 1:
        multi_people.append(people)

from collections import defaultdict
edgesDictionary = defaultdict(int)
for people in multi_people:
    for personA in people:
        for personB in people:
            if personA < personB:
                edgesDictionary[personA + " -- " + personB] += 1

edgesFreqs = nltk.FreqDist(edgesDictionary)
edgesFreqs.most_common(5) # have a peek

import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

G = nx.Graph()
plt.figure(figsize=(20,20))

# create graph edges (node pairs) and keep track of edges for each count
edges = defaultdict(list)
for names, count in edgesFreqs.most_common():
    if count > 1:
        parts = names.split(" -- ")
        G.add_edge(parts[0], parts[1], width=count)
        edges[count].append((parts[0], parts[1]))
    else:
        break

# draw labels (nx.draw(G) doesn't really work)
pos = nx.spring_layout(G)
nx.draw_networkx_labels(G, pos)

# draw edges with different widths
for count, edgelist in edges.items():
    g = G.subgraph(pos)
    nx.draw_networkx_edges(g, pos, edgelist=edgelist, width=count, alpha=0.1)

plt.axis("equal")
plt.xticks([], []), plt.yticks([], [])
plt.show()
