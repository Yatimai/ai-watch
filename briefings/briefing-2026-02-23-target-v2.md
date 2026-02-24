# Veille IA — 23 février 2026

## 🔬 Recherche

### Qwen3-VL Technical Report
**84 upvotes** · Qwen · [Papier](https://huggingface.co/papers/2511.21631)

Rapport technique du modèle vision-langage de nouvelle génération de Qwen. L'écosystème Qwen s'élargit vite — la version API Qwen3.5 Plus (397B params, 17B activés par forward pass) supporte déjà 1M tokens de contexte. Les GGUF Unsloth sont disponibles de 94 Go à 462 Go.

### OneThinker : raisonnement unifié image + vidéo
**19 upvotes**, 43 interactions · [Papier](https://huggingface.co/papers/2512.03043)

Modèle de raisonnement all-in-one qui traite image et vidéo dans un même pipeline, plutôt que deux modèles séparés. 14 auteurs.

### Steering VLA Models as Anti-Exploration
**29 upvotes** · [Papier](https://huggingface.co/papers/2512.02834)

Approche de test-time scaling pour les modèles Vision-Language-Action en robotique. Utilise l'anti-exploration pour améliorer les performances sans ré-entraînement.

---

## 🛠 Outils

### x1xhlol/system-prompts-and-models-of-ai-tools
**2 447 ⭐/jour** · [Repo](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools)

Collection complète des system prompts extraits des principaux outils IA : Augment Code, Claude Code, Cursor, Devin, Kiro, Windsurf et d'autres. Explose en popularité — utile pour comprendre comment chaque outil instruite son LLM sous le capot.

### huggingface/skills
**1 470 ⭐/jour** · [Repo](https://github.com/huggingface/skills)

Repo officiel HuggingFace de "skills" — des modules de compétences réutilisables pour agents IA. HuggingFace structure son approche des agents autour de briques modulaires.

### VectifyAI/PageIndex — RAG sans vecteurs
**552 ⭐/jour** · [Repo](https://github.com/VectifyAI/PageIndex)

Index de documents pour du RAG "vectorless" basé sur le raisonnement. Pas d'embeddings, pas de base vectorielle — le LLM interroge directement un index structuré. Si ça scale, ça élimine tout le pipeline chunk → embed → similarity search.

---

## 📡 Analyse

### Karpathy parle des "Claws"
21 fév · [Post](https://simonwillison.net/2026/Feb/21/claws/) · tags : ai-agents, openclaw

Andrej Karpathy utilise le terme "Claws" pour décrire les agents autonomes persistants qui tournent en continu. Le concept gagne en légitimité.

### Raspberry Pi +42% en bourse grâce à OpenClaw
22 fév · [Post](https://simonwillison.net/2026/Feb/22/raspberry-pi-openclaw/) · tags : raspberry-pi, ai-agents, openclaw

Des utilisateurs font tourner OpenClaw sur des Raspberry Pi. Le buzz a été vu des millions de fois. L'action Raspberry Pi a bondi de 42% en deux jours, entre le phénomène viral et un achat d'actions du CEO.

### Paul Ford dans le NYT : la disruption est arrivée
18 fév · [Post](https://simonwillison.net/2026/Feb/18/the-ai-disruption/) · tags : careers, coding-agents, claude-code

Paul Ford, ex-CEO de Postlight, chiffre l'impact : un site perso qu'il aurait facturé 25 000$ refait en un week-end, une conversion de dataset qu'il aurait facturée 350 000$ faite avec un abonnement Claude Pro. Il décrit le "moment novembre" où Claude Code est devenu vraiment capable.

---

*Sources : HuggingFace Papers API, GitHub Trending, simonwillison.net*
