#!/bin/bash

rm -f tunes.json
wget https://github.com/adactio/TheSession-data/raw/master/json/tunes.json

mkdir ../lib
cd ../lib
wget http://central.maven.org/maven2/org/json/json/20171018/json-20171018.jar
wget http://central.maven.org/maven2/org/parboiled/parboiled-java/0.10.0/parboiled-java-0.10.0.jar
wget http://central.maven.org/maven2/org/parboiled/parboiled-core/0.10.0/parboiled-core-0.10.0.jar
wget http://central.maven.org/maven2/org/xerial/sqlite-jdbc/3.21.0/sqlite-jdbc-3.21.0.jar
cd -

echo 'Compiling abc4j library...'
javac -cp ../lib/*:. \
      abc/instructions/*.java \
      abc/audio/*.java \
      abc/notation/*.java \
      abc/parser/*.java

echo '\nCompiling DBbuilder...'
javac -cp abc/*:. ABCSpeller.java
javac MattABCTools.java
javac -cp ../lib/*:. DBBuilder.java

echo '\nBuilding DB'
java -cp ../lib/*:. DBBuilder > output.log 2>&1

mv corpus.db ../..
echo 'Done!'
