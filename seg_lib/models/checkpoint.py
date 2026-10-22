def strip_module_prefix(state_dict: dict) -> dict:
    '''Removes the `module.` prefix added by `DataParallel` to the keys.'''
    prefix = 'module.'
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }
