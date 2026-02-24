Tu reçois une liste de repos GitHub trending. Pour chaque repo, réponds `true` ou `false` : est-ce que ce repo est lié à l'IA, au machine learning, ou aux LLMs ?

## Critères

Lié à l'IA (`true`) :
- Modèles, frameworks, outils ML/DL
- Agents IA, assistants, chatbots
- Prompts, prompt engineering, system prompts
- RAG, embeddings, bases vectorielles
- Fine-tuning, entraînement, évaluation de modèles
- Wrappers/UI pour des outils IA (ex: interface pour Claude Code)
- Données/datasets pour ML

Pas lié à l'IA (`false`) :
- Outils réseau, VPN, proxy
- Jeux, streaming, médias
- Infrastructure générale (sauf si spécifiquement pour ML)
- Listes awesome non-IA
- DevOps/CI/CD généraliste

## Format d'entrée

```json
[
  {"repo": "owner/name", "description": "...", "language": "..."},
  ...
]
```

## Format de sortie

```json
[
  {"repo": "owner/name", "is_ai": true},
  ...
]
```

Réponds uniquement avec le JSON, rien d'autre.
