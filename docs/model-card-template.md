# Model card template

The template now lives with the code that fills it, so the trainer image can
ship it: [`ml/evaluation/model_card_template.md`](../ml/evaluation/model_card_template.md).

`ml.evaluation.card.render_model_card` fills it from the evaluation report,
the run's data and split metadata, feature importances and (after
`make promote`) the gate result. Every registered version has its rendered
card as the `model_card.md` artifact of its MLflow run.
