# Setup GitHub Actions - Guida Completa

## ✅ Cosa è stato creato

Ho configurato un sistema CI/CD completo e professionale con 5 workflow GitHub Actions:

### 1. **CI Workflow** (`.github/workflows/ci.yml`)
Il workflow principale che si attiva ad ogni push e PR:
- ✅ **Lint**: Controlla stile codice con Ruff
- ✅ **Test Matrix**: Testa su 9 combinazioni (3 OS × 3 versioni Python)
- ✅ **Security**: Scansiona vulnerabilità con Safety e Bandit
- ✅ **Build**: Valida che il package si compili correttamente
- ✅ **Coverage**: Traccia copertura test (preparato per Codecov)

### 2. **Release Workflow** (`.github/workflows/release.yml`)
Gestisce pubblicazione automatica:
- 📦 Build distribuzione Python (wheel + source)
- 🚀 Pubblicazione automatica su PyPI al rilascio
- 🐳 Build e push immagine Docker multi-architettura
- 📝 Generazione note di rilascio automatiche

### 3. **CodeQL Workflow** (`.github/workflows/codeql.yml`)
Security scanning avanzato GitHub:
- 🔍 Analisi settimanale automatica
- 🛡️ Rileva vulnerabilità e problemi di qualità
- 📊 Report nella tab Security

### 4. **Dependency Review** (`.github/workflows/dependency-review.yml`)
Controllo dipendenze nelle PR:
- ⚠️ Blocca PR con vulnerabilità moderate+
- 💬 Commenta summary nella PR

### 5. **Stale Issues** (`.github/workflows/stale.yml`)
Gestione automatica issue/PR inattive:
- 🏷️ Marca come "stale" dopo inattività
- 🗑️ Chiude automaticamente se nessuna risposta

### 6. **Dependabot** (`.github/dependabot.yml`)
Aggiornamenti automatici dipendenze:
- 📅 Controllo settimanale (lunedì 3 AM)
- 🤖 PR automatiche per aggiornamenti sicuri
- 👥 Assegnate automaticamente a te

### 7. **Issue Templates**
Templates strutturati per:
- 🐛 Bug Report
- ✨ Feature Request  
- 📚 Documentation Issues

### 8. **Pull Request Template**
Template standardizzato per PR con checklist completa

---

## 🚀 Passi Successivi per Attivare Tutto

### Passo 1: Push dei file su GitHub
```bash
cd /Users/filippostanghellini/GitHub/DocFinder
git add .github/
git add .gitignore
git add pyproject.toml
git add README.md
git commit -m "ci: add comprehensive GitHub Actions workflows and templates"
git push origin main
```

### Passo 2: Configura i Secrets (Opzionali ma consigliati)

#### A. Per Coverage Report (Codecov)
1. Vai su https://codecov.io
2. Accedi con GitHub
3. Aggiungi il repository DocFinder
4. Copia il token
5. Vai su: `https://github.com/filippostanghellini/DocFinder/settings/secrets/actions`
6. Clicca "New repository secret"
7. Nome: `CODECOV_TOKEN`, Valore: [il token copiato]

#### B. Per Pubblicazione PyPI (quando sarai pronto)

**Opzione 1: Trusted Publishing (RACCOMANDATO - Nessun secret!)**
1. Vai su https://pypi.org/manage/account/publishing/
2. Aggiungi trusted publisher:
   - Owner: `filippostanghellini`
   - Repository: `DocFinder`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Fatto! Nessun token necessario

**Opzione 2: API Token (Alternativa)**
1. Vai su https://pypi.org/manage/account/token/
2. Crea token con scope "Entire account" o specifico per DocFinder
3. Aggiungi secret `PYPI_API_TOKEN`
4. Ripeti per TestPyPI: https://test.pypi.org → `TESTPYPI_API_TOKEN`

#### C. Per Docker Hub (opzionale)
1. Crea account su https://hub.docker.com
2. Settings → Security → New Access Token
3. Aggiungi secrets:
   - `DOCKER_USERNAME`: il tuo username Docker Hub
   - `DOCKER_PASSWORD`: il token (NON la password!)

### Passo 3: Abilita GitHub Actions
1. Vai su: `https://github.com/filippostanghellini/DocFinder/actions`
2. Se richiesto, clicca "I understand my workflows, go ahead and enable them"

### Passo 4: Configura Branch Protection (IMPORTANTE!)
1. Vai su: `https://github.com/filippostanghellini/DocFinder/settings/branches`
2. Clicca "Add branch protection rule"
3. Branch name pattern: `main`
4. Abilita:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
     - Cerca e seleziona: `Lint`, `Test`, `Build`
   - ✅ Require conversation resolution before merging
   - ✅ Do not allow bypassing the above settings
5. Salva

Ora **nessuno può pushare direttamente su main** senza che i test passino! 🎉

---

## 🧪 Test del Setup

### Test 1: CI Workflow
```bash
# Crea una branch di test
git checkout -b test/ci-setup
echo "# Test" >> test_file.md
git add test_file.md
git commit -m "test: verify CI workflow"
git push origin test/ci-setup

# Crea una PR su GitHub
# Dovresti vedere i workflow partire automaticamente!
```

### Test 2: Linting Locale
```bash
# Installa le dipendenze dev
pip install -e ".[dev]"

# Esegui ruff
ruff check src/ tests/
ruff format --check src/ tests/

# Fix automatico
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Test 3: Test Locale
```bash
# Esegui i test
pytest -v

# Con coverage
pytest --cov=docfinder --cov-report=html
# Apri htmlcov/index.html nel browser
```

### Test 4: Security Scan Locale
```bash
# Safety check
pip install safety
safety check

# Bandit check
pip install bandit[toml]
bandit -r src/
```

---

## 📊 Monitoraggio Workflow

### Dashboard Actions
- URL: `https://github.com/filippostanghellini/DocFinder/actions`
- Qui vedi tutti i workflow in esecuzione
- Puoi re-eseguire workflow falliti
- Scarica logs per debugging

### Status Badge
I badge nel README mostrano status in tempo reale:
- 🟢 Verde = tutto ok
- 🔴 Rosso = qualcosa è fallito
- 🟡 Giallo = in esecuzione

### Email Notifications
Riceverai email quando:
- Un workflow fallisce
- Dependabot crea una PR
- Qualcuno apre issue/PR

---

## 🎯 Workflow di Sviluppo Consigliato

### Per Feature/Bug Fix
```bash
# 1. Crea branch
git checkout -b feature/nome-feature

# 2. Sviluppa e testa localmente
ruff check src/ tests/
pytest

# 3. Commit
git add .
git commit -m "feat: add new feature"

# 4. Push
git push origin feature/nome-feature

# 5. Crea PR su GitHub
# I workflow CI partono automaticamente

# 6. Aspetta review e CI verde
# 7. Merge (se protezione branch attiva, serve approvazione)
```

### Per Release
```bash
# 1. Aggiorna versione in pyproject.toml
# version = "0.2.0"

# 2. Aggiorna CHANGELOG.md (quando lo creerai)

# 3. Commit e push
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git push origin main

# 4. Crea release su GitHub
git tag v0.2.0
git push origin v0.2.0

# Via UI GitHub:
# - Vai su Releases → Draft new release
# - Scegli tag: v0.2.0
# - Auto-generate release notes
# - Publish

# 5. Il workflow Release parte automaticamente e:
#    - Compila il package
#    - Pubblica su PyPI
#    - Builda Docker image
#    - Allega artifacts alla release
```

---

## 🔧 Customizzazione

### Modificare Python Versions
In `.github/workflows/ci.yml`:
```yaml
matrix:
  python-version: ["3.10", "3.11", "3.12", "3.13"]  # Aggiungi 3.13
```

### Modificare OS Testati
```yaml
matrix:
  os: [ubuntu-latest, macos-latest]  # Rimuovi windows se non serve
```

### Disabilitare Coverage Upload
Rimuovi o commenta in `ci.yml`:
```yaml
# - name: Upload coverage to Codecov
#   uses: codecov/codecov-action@v4
#   ...
```

### Cambiare Schedule Dependabot
In `.github/dependabot.yml`:
```yaml
schedule:
  interval: "daily"  # Invece di weekly
```

---

## 📚 Risorse Utili

- **GitHub Actions Docs**: https://docs.github.com/actions
- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **PyPI Trusted Publishing**: https://docs.pypi.org/trusted-publishers/
- **Codecov**: https://docs.codecov.com/docs
- **Act (test locale)**: https://github.com/nektos/act

---

## ❓ Troubleshooting Comuni

### "Workflow not running"
- Verifica che GitHub Actions sia abilitato nel repo
- Controlla sintassi YAML (usa yamllint)
- Guarda tab Actions per errori

### "Tests fail on CI but pass locally"
- Controlla differenze OS (path separators)
- Verifica dipendenze in pyproject.toml
- Controlla variabili ambiente

### "PyPI publish fails: version already exists"
- Incrementa versione in pyproject.toml
- Non puoi ri-pubblicare stessa versione

### "Coverage badge shows 'unknown'"
- Aspetta che primo upload codecov completi
- Verifica CODECOV_TOKEN sia settato
- Controlla che repo sia linked su codecov.io

---

## ✅ Checklist Finale

Prima del primo release pubblico:

- [ ] Push tutti i file .github
- [ ] Configura Codecov (opzionale ma consigliato)
- [ ] Configura PyPI trusted publisher
- [ ] Abilita branch protection su main
- [ ] Testa CI con una PR
- [ ] Aggiungi CONTRIBUTING.md (prossimo step!)
- [ ] Aggiungi SECURITY.md
- [ ] Aggiungi CHANGELOG.md
- [ ] Verifica badge nel README

---

**Tutto pronto!** 🎉 Hai ora un setup CI/CD di livello enterprise, completamente gratuito con GitHub Actions.

Vuoi che proceda con il prossimo punto (es. CONTRIBUTING.md, test più completi, pre-commit hooks)?
