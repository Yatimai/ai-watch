# AI Watch Agent — Spécification complète

## Le projet

Agent autonome de veille IA. Tourne chaque matin via GitHub Actions. Produit un briefing markdown quotidien à partir de 3 sources. Déployé sur GitHub Pages.

Projet portfolio pour AI engineer.

---

## Pipeline

```
[Fetch 3 sources] → [Filtre GitHub: est-ce IA ?] → [Top 3 par source] → [Agent enrichissement + briefing] → [Push markdown]
```

**Étapes déterministes (script)** : fetch, filtre, tri, top 3.
**Étape agentique (LLM + tools)** : enrichissement et rédaction du briefing.

---

## Sources

### 1. HuggingFace Daily Papers
- **Accès** : `GET https://huggingface.co/api/daily_papers`
- **Données** : titre, abstract, upvotes, arXiv ID, auteurs
- **Tri** : par upvotes décroissants
- **Top** : 3
- **Pas de filtre nécessaire** : tout est IA

### 2. GitHub Trending
- **Accès** : scraping `https://github.com/trending?since=daily` (tous langages)
- **Données** : nom, description, stars/jour, langage
- **Filtre** : appel LLM pour déterminer si chaque repo est lié à l'IA (voir prompt filtre)
- **Tri** : par stars/jour décroissants parmi les repos IA
- **Top** : 3

### 3. Simon Willison
- **Accès** : RSS `https://simonwillison.net/atom/everything/`
- **Données** : titre, lien, tags, extrait
- **Pas de filtre nécessaire** : Simon poste rarement plus de 3 articles/jour, quasi tout est IA
- **Top** : 3 (ou moins si peu de posts)

---

## Prompts

### Prompt filtre GitHub

Reçoit la liste des repos trending. Répond `true/false` pour chaque : lié à l'IA ou non.

- **Input** : JSON `[{"repo": "owner/name", "description": "...", "language": "..."}]`
- **Output** : JSON `[{"repo": "owner/name", "is_ai": true}]`
- **Critères IA** : modèles, frameworks ML, agents, prompts, RAG, fine-tuning, UI pour outils IA, datasets
- **Critères non-IA** : réseau, VPN, jeux, streaming, infra générale, devops généraliste

Fichier : `prompt-filter-github.md`

### Prompt enrichissement + briefing

Reçoit les 9 items bruts. Utilise des tools pour enrichir, puis rédige le briefing markdown.

- **Tools disponibles** :
  - `fetch_url(url)` — lire une page web
  - `search_hf_models(query)` — chercher un modèle sur le Hub
  - `get_github_repo(owner, repo)` — README, stars, activité

- **Règle d'appel des tools** : seulement si les données brutes ne suffisent pas pour un résumé utile
  - Abstract clair → pas de tool
  - Modèle mentionné sans détails → `search_hf_models`
  - Description GitHub vague → `get_github_repo`
  - Post Simon avec titre seul → `fetch_url`

- **Format de sortie** : voir ci-dessous

- **Règles de rédaction** :
  - 2-4 phrases par item
  - "Pourquoi c'est notable" > "ce que c'est"
  - Pas de phrases creuses
  - Chiffres concrets bienvenus
  - On garde les 9 items, on ne triche pas avec le tri
  - Français

Fichier : `prompt-briefing.md`

---

## Format du briefing

```markdown
# Veille IA — {date}

## 🔬 Recherche

### {titre}
**{upvotes} upvotes** · {auteur/org} · [Papier]({lien})

{2-4 phrases}

(×3 items)

---

## 🛠 Outils

### {owner/repo} — {description courte}
**{stars}/jour** · [Repo]({lien})

{2-4 phrases}

(×3 items)

---

## 📡 Analyse

### {titre}
{date} · [Post]({lien}) · tags : {tags}

{2-4 phrases}

(×3 items)

---

*Sources : HuggingFace Papers API, GitHub Trending, simonwillison.net*
```

---

## Logs

L'agent logge chaque décision de tool use :
```
[HF-1] Qwen3-VL: abstract disponible, modèle mentionné → search_hf_models("Qwen3-VL") → 2M+ downloads, 6 variantes
[HF-2] Anti-Exploration: abstract clair → pas d'enrichissement
[GH-1] system-prompts: description explicite → pas d'enrichissement
[GH-2] huggingface/skills: description vague → get_github_repo → skills pour Claude Code/Codex/Gemini CLI
[SW-1] "Claws": titre seul → fetch_url → Karpathy nomme les agents persistants
```

Les logs sont sauvegardés avec chaque briefing. Ils montrent que l'agent prend des décisions différentes selon le contenu.

---

## Stack technique

- **Orchestration** : LangGraph
- **LLM** : Claude Sonnet (un seul modèle pour tout)
- **Langage** : Python
- **Déploiement** : GitHub Actions (cron quotidien ~8h)
- **Hébergement** : GitHub Pages (briefings markdown)
- **Stockage** : les briefings sont commit dans le repo

---

## Gestion des erreurs

- Si une source est indisponible : le briefing est produit avec les sources restantes + note "⚠️ Source indisponible : {nom}"
- Si le LLM échoue : retry 1 fois, puis erreur logguée

---

## Ce que le recruteur voit

1. **README** : description en 5 lignes
2. **Briefing du jour** : données réelles, pas un notebook one-shot
3. **Historique** : 2-3 semaines de commits quotidiens automatiques
4. **Code** : LangGraph propre, prompts explicites, tools définis
5. **Logs** : l'agent qui réfléchit, visible

---

## Plan de travail

### Phase 0 — Calibrage (fait)
✅ Données brutes des 3 sources récupérées
✅ Briefing cible écrit à la main
✅ Prompts rédigés (filtre + briefing)

### Phase 1 — Agent fonctionnel
- Fetch des 3 sources (API HF, scraping GitHub, RSS Simon)
- Filtre GitHub (appel LLM)
- Tri + top 3 par source
- Agent enrichissement + briefing (LLM + tools via LangGraph)
- Output markdown + logs
- Tester le prompt contre le briefing cible

### Phase 2 — Déploiement
- GitHub Actions cron quotidien
- Push auto des briefings + logs
- README portfolio

### Phase 3 — Polish (optionnel)
- Page web GitHub Pages pour afficher les briefings
- Optimisation des prompts selon les résultats réels
