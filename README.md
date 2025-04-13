## Dolos
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @dodona/dolos
zip -r duplication1.zip duplication1
dolos run -f web -l python duplication1.zip
```

## JPlag
1. Download java version 21
2. Download JPlag `jplag-6.0.0-jar-with-dependencies.jar` from `https://github.com/jplag/jplag/releases`
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 6 -r results-jplag-type1 jplag/type

```