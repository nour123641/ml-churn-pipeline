"""Déploiement Prefect : planifie les flows une fois par jour."""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_code, flow_evaluate, flow_train

if __name__ == "__main__":
    # Tous les jours à 02:00 (heure de Tunis)
    daily = CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis")

    deploy_all = flow_all.to_deployment(
        name="ml-pipeline-all",
        schedules=[daily],
        tags=["mlops", "full-pipeline"],
    )
    deploy_train = flow_train.to_deployment(
        name="ml-pipeline-train",
        tags=["mlops", "training"],
    )
    deploy_evaluate = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["mlops", "evaluation"],
    )
    deploy_code = flow_code.to_deployment(
        name="ml-pipeline-code",
        tags=["mlops", "code-quality"],
    )

    serve(deploy_all, deploy_train, deploy_evaluate, deploy_code)
