from model.troika import Troika


def get_model(
    config,
    dataset,
    attributes,
    classes,
    offset,
    **kwargs,
):
    if config.model_name != "troika":
        raise NotImplementedError(
            f"Unrecognized model name: {config.model_name}"
        )

    return Troika(
        config,
        dataset=dataset,
        attributes=attributes,
        classes=classes,
        offset=offset,
    )
