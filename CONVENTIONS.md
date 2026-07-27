# CONVENTIONS — Projet TSO

## Règles générales

1. **TDD** — Toujours écrire le test qui échoue d'abord, puis implémenter, puis refactor.
2. **Git** — `main` est sacred. Toujours créer une branche feature avec `kickoff-branch` (via worktree). Jamais de commit direct sur `main`.
3. **Commits** — Conventional Commits : `feat:`, `fix:`, `refactor:`, `docs:`, `test:`. Un commit par changement atomique.
4. **Vérification avant commit** — `cargo check --bins && cargo clippy --all-targets && cargo test`
5. **Review** — Tout code nouveau passe par `audit-code` avant merge. Les PRs solo peuvent utiliser `land-branch.sh`.

## Code

- **Rust** — Utiliser `edition = "2024"`. Pas de `unsafe` sauf justification explicite.
- **Nommage** — Structs en PascalCase, fonctions/méthodes en snake_case, variables en snake_case.
- **Modules** — Chaque module expose son API publique dans `lib.rs`. Les binaires sont dans `src/bin/`.
- **Erreurs** — Préférer `Result<T, E>` à `panic!` / `unwrap()`. Les `unwrap()` sont tolérés dans les binaires de test/debug.
- **Logging** — Utiliser `eprintln!` pour le debug, pas `println!`.
- **Docs** — Les modules principaux ont des doc comments (`///`) avec au moins un exemple.
- **Perfs** — Éviter les `clone()` inutiles. Utiliser `ArrayView` de ndarray plutôt que `Array` aux interfaces.

## Projet

- `paper.md` — Document de recherche principal (55 pages). Les modifications architecturales majeures y sont reflétées.
- `specs/` — Fichiers de planification et suivi. `state.yaml` pour l'état de session, `UBIQUITOUS_LANGUAGE_LATEST.md` pour le glossaire.
- `tso-engine/src/bin/` — Binaires expérimentaux. Chaque expérience a son propre fichier.
- `target/` — Build artifacts. Ignoré par git.

## Workflow recommandé

1. `survey-context` — Où j'en suis ?
2. `kickoff-branch` — Je crée une branche propre
3. `develop-tdd` — J'implémente en TDD
4. `audit-code` — Je vérifie la qualité
5. `verify-work` — Je valide le résultat
6. `release-branch` — Je merge proprement
