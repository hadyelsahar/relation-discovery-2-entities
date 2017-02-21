# relation-discovery-2-entities
unsupervised relation discovery between two entities




##Loading tutorial

### Loading the CoreNLP Server and get it running 

- make sure you have java-8 installed
- download and unzip `stanford-corenlp-2015-12-09`
- for more infor check [CoreNLP Server Webpage](http://stanfordnlp.github.io/CoreNLP/corenlp-server.html)
 
``` 
mkdir ./utils/corenlp

wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip -O ./utils/corenlp/stanford-corenlp-full-2015-12-09.zip

unzip ./utils/corenlp/stanford-corenlp-full-2015-12-09.zip -d ./utils/corenlp/

# Set up your classpath. For example, to add all jars in the current directory tree:
cd ./utils/corenlp/stanford-corenlp-full-2015-12-09
export CLASSPATH="`find . -name '*.jar'`"

# Run the server on port 9000
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000
```


### loading DBpedia Spotlight

```
wget http://www.dbpedia-spotlight.com/dbpedia-spotlight-latest.jar
wget http://www.dbpedia-spotlight.com/latest_models/en.tar.gz
tar xzf en.tar.gz
java -Xmx6g -jar dbpedia-spotlight-latest.jar en http://localhost:2222/rest
```
