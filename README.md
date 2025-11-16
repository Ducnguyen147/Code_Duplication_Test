
## Dolos
Link: https://dolos.ugent.be/docs/running.html
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @dodona/dolos
zip -r duplication1.zip duplication1
dolos run -f web -l python duplication1.zip
dolos run -f csv -l python duplication1.zip # Parse to csv
dolos run -l python duplication1.zip

```

## JPlag
1. Download java version 21
2. Download JPlag `jplag-6.0.0-jar-with-dependencies.jar` from `https://github.com/jplag/jplag/releases`

## PMD-CPD
1. sudo apt install -y unzip
2. wget https://github.com/pmd/pmd/releases/download/pmd_releases%2F7.13.0/pmd-dist-7.13.0-bin.zip
3. unzip pmd-dist-7.13.0-bin.zip
4. cd pmd-bin-7.13.0
5. export PATH="$PATH:$(pwd)/bin"
6. Test version: `pmd --version``


## JSCPD
1. sudo npm install -g jscpd
2. jscpd --version

## Semantic comparison

<!-- # 1) Clean out conflicting bits
pip uninstall -y tree_sitter_languages tree-sitter-language-pack tree-sitter

# 2) Install a modern, compatible stack
pip install "tree-sitter>=0.25,<0.26" "tree-sitter-language-pack>=0.7"
# (py-tree-sitter 0.25.x docs show Parser(language, ...))  ← verified. :contentReference[oaicite:3]{index=3} -->


```bash
python3 detect_clone_cli_v8.py --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/negatives --extensions .py --min-tokens 5 --mode hybrid --prefilter-topM 50 --mutual-nearest
```