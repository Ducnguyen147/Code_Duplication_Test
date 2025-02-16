## Dolos
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @dodona/dolos
zip -r duplication1.zip duplication1
dolos run -f web -l python duplication1.zip
```