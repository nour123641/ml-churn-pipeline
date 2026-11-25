"""Point d'entrée : exécute les étapes du pipeline via la ligne de commande."""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument("--all", action="store_true", help="Exécuter toutes les étapes")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle")
    parser.add_argument(
        "--load", action="store_true", help="Charger le modèle sauvegardé et l'évaluer"
    )
    args = parser.parse_args()

    # --all active toutes les étapes
    if args.all:
        args.prepare = args.train = args.evaluate = args.save = args.load = True

    if not (args.prepare or args.train or args.evaluate or args.save or args.load):
        parser.print_help()
        return

    try:
        # 1. Préparation des données
        print("=== 1. PREPARE ===")
        x_train, x_test, y_train, y_test = prepare_data()
        print(f"Données préparées : train {x_train.shape}, test {x_test.shape}")
        print("✅ Préparation des données réussie\n")

        # 2. Entraînement (nécessaire avant d'évaluer ou de sauvegarder)
        model = None
        if args.train or args.evaluate or args.save:
            print("=== 2. TRAIN ===")
            model = train_model(x_train, y_train)
            print("✅ Entraînement du modèle réussi\n")

        # 3. Évaluation
        if args.evaluate:
            print("=== 3. EVALUATE ===")
            evaluate_model(model, x_test, y_test)
            print("✅ Évaluation du modèle réussie\n")

        # 4. Sauvegarde
        if args.save:
            print("=== 4. SAVE ===")
            save_model(model)
            print("✅ Sauvegarde du modèle réussie\n")

        # 5. Chargement (puis évaluation pour prouver qu'il fonctionne)
        if args.load:
            print("=== 5. LOAD ===")
            model = load_model()
            evaluate_model(model, x_test, y_test)
            print("✅ Chargement du modèle réussi\n")

        print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS !")

    except Exception as erreur:
        print(f"❌ ÉCHEC du pipeline : {erreur}")


if __name__ == "__main__":
    main()
