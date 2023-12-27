"""MVP config file."""

def get_config() -> dict:

    CONFIG = {}
    model_params = {
        'eps': 0.3,
        'min_samples': 10,
    }

    openai_model = 'gpt-3.5-turbo'

    CONFIG['model_params'] = model_params
    CONFIG['openai_model'] = openai_model

    return CONFIG