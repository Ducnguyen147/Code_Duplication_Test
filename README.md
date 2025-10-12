
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
JPlag: use -t 9 as a robust default for typical coursework; the original JPlag evaluation found 7–11 to be the most reliable range and set the default to 9. For your small artificial snippets, go lower (-t 5–7) to avoid missing short matches (but expect more false positives).

1. Download java version 21
2. Download JPlag `jplag-6.0.0-jar-with-dependencies.jar` from `https://github.com/jplag/jplag/releases`
3. Commands
- Type 1:
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 7 -r jplag/type1/results-jplag-type1 jplag/type1
```
- Type 2:
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 10 -r jplag/type2/results-jplag-type2 jplag/type2
```
- Type 3:
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 4 -r jplag/type3/results-jplag-type3 jplag/type3
```
- Type 4;
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 3 -r jplag/type4/results-jplag-type4 jplag/type4
```
- Control flow
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 5 -r jplag/control_flow/results-jplag-cf jplag/control_flow
```
- Hybrid
```bash
java -jar jplag-6.0.0-jar-with-dependencies.jar -l python3 -t 5-r jplag/hybrid/results-jplag-hybrid jplag/hybrid
```

## PMD-CPD
PMD‑CPD: there’s no built‑in default in the CLI (you must supply a value), but common guidance and examples use ~100 tokens for normal projects (and the Maven plugin’s default is 100). For your short snippets, start at 10–30 tokens per language subset (Python near 10–20, Java/C++ 20–30), then sweep.

1. sudo apt install -y unzip
2. wget https://github.com/pmd/pmd/releases/download/pmd_releases%2F7.13.0/pmd-dist-7.13.0-bin.zip
3. unzip pmd-dist-7.13.0-bin.zip
4. cd pmd-bin-7.13.0
5. export PATH="$PATH:$(pwd)/bin"
6. Test version: `pmd --version``
7. 
- Type 1:
```bash
pmd cpd   --minimum-tokens 20   --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type1   --language python   --format xml > /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type1/cpd-type1.xml
```

- Type 2:
```bash
--ignore-identifiers    # anonymize all identifier names
--ignore-literals       # anonymize all literal values
```

 The official docs specify these flags are only implemented for Java and C++ lexers: https://pmd.github.io/pmd/pmd_userdocs_cpd.html?utm_source=chatgpt.com

```bash
pmd cpd   --minimum-tokens 5   --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type2 --ignore-identifiers --ignore-literals  --language python   --format xml > /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type2/cpd-type2.xml
```

- Type 3:
```bash
pmd cpd   --minimum-tokens 5 --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type3  --language python   --format xml > /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type3/cpd-type3.xml
```

- Type 4:
```bash
pmd cpd   --minimum-tokens 5 --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type4  --language python   --format xml > /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type4/cpd-type4.xml
```

- Control Flow:
```bash
pmd cpd   --minimum-tokens 5 --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/control_flow  --language python   --format xml > /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/control_flow/cpd-cf.xml
```

- Hybrid
```bash

```

## JSCPD
jscpd: default is --min-tokens 50 and --min-lines 5; that’s good for real‑world repos, but too high for shortcode snippets. Start at --min-tokens 10–20 and --min-lines 1–2, then sweep.

1. sudo npm install -g jscpd
2. jscpd --version
3.
- Type 1:
```bash
jscpd /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type1/type1a.py /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type1/type1b.py --format python --min-tokens 10 --min-lines 1 --mode weak --reporters console,html,json --output /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type1/reports
```

- Type 2:
```bash
jscpd /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type2/type2a.py /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type2/type2b.py --format python --min-tokens 3 --min-lines 1 --mode weak --reporters console,html,json --output /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type2/report
```
a-b: Token = 3
a-c: Token = 5
b-c: Token = 3

- Type 3:
```bash
jscpd /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type3/type3a.py /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type3/type3b.py --format python --min-tokens 6 --min-lines 1 --mode weak --reporters console,html,json --output /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type3/reports
```
a-b: Token = 5
a-c: Token = 4
b-c: Token = 5

- Type 4:
```bash
jscpd /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type4/type4a.py /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type4/type4b.py --format python --min-tokens 3 --min-lines 1 --mode weak --reporters console,html,json --output /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/type4/reports
```
a-b: Token = 4
a-c: token = 4
b-c: Token = 4

- Control flow:
```bash
jscpd /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/control_flow/cf_b.py /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/control_flow/cf_a.py --format python --min-tokens 4 --min-lines 1 --mode weak --reporters console,html,json --output /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/control_flow/reports
```

- Hybrid:
```bash
jscpd --skipLocal /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/hybrid/a /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/hybrid/b --format python --min-tokens 4 --min-lines 1 --mode weak --reporters console,html,json --output /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/jscpd/hybrid/reports
```

## Semantic comparison
```bash
python3 semantic_comparison.py --files dolos/type4/dolos-report-20250406T191928512Z-type4/files.csv --pairs dolos/type4/dolos-report-20250406T191928512Z-type4/pairs.csv --output semantic_pairs.csv --threshold 0.5 --device cpu
```

```bash
python3 semantic_clone_detector.py --file1 path1 --file2 path2 --threshold 0.7
```

```bash
python3 semantic_clone_binary.py --file1 path1 --file2 path2 --threshold 0.7
```

```bash
python detect_clones_cli.py --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/non_similar --extensions .py .java --model mchochlov/codebert-base-cd-ft --threshold 0.85
```

```bash
python detect_clone_cli_v3.py --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/non_similar --extensions .py .java --fusion concat --w-embed 1.0 --w-ast 0.6 --w-lex 0.2 --threshold 0.2
```

```bash
python detect_clone_cli_v3.py --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type1 --extensions .py .java --fusion late --prefilter-threshold 0.75 --w-embed 1.0 --w-ast 0.7 --w-lex 0.2 --threshold 0.2
```
<!-- # 1) Clean out conflicting bits
pip uninstall -y tree_sitter_languages tree-sitter-language-pack tree-sitter

# 2) Install a modern, compatible stack
pip install "tree-sitter>=0.25,<0.26" "tree-sitter-language-pack>=0.7"
# (py-tree-sitter 0.25.x docs show Parser(language, ...))  ← verified. :contentReference[oaicite:3]{index=3} -->

```bash
python detect_clone_cli_v4.py --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/dolos/type4 --extensions .py --fusion late --prefilter-threshold 0.80 --threshold 0.20 --w-embed 1.0 --w-ast 0.35 --embed-superpass 0.7 --w-lex 0.1 --no-center --ast-no-center --ast-tfidf --lex-mode py-token --lex-n 3 --fp-k 5 --fp-w 4 --min-fp-sim 0.05 --min-fp-total 2 --min-fp-longest 1 --min-ast-sim 0.00 --min-lex-sim 0.00 --debug-components --topk 20
``````

```bash
python3 detect_clone_cli_v7.py --dir /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/hybrid --extensions .py --min-tokens 5 --mode semantic --model BAAI/bge-code-v1
```