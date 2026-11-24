"""Family-aware tag identities shared by configuration, detection, and localization."""

DETECTOR_FAMILIES = {'36h11': '36h11', '41h12': 'Standard41h12'}


def canonical_family(name):
    """Normalize detector spellings without merging different tag families."""
    name = name.removeprefix('tag') if isinstance(name, str) else ''
    name = '41h12' if name == 'Standard41h12' else name
    if name not in DETECTOR_FAMILIES:
        raise ValueError(f'Unsupported tag family: {name}')
    return name


def tag_key(item):
    """Identify a landmark by (canonical family, numeric ID)."""
    return canonical_family(item.get('family', '36h11')), item['id']


def tag_frame(key):
    """Return a unique TF frame even when numeric IDs overlap."""
    return f'tag_{key[0]}_{key[1]}'


def mapped_families(environment):
    """Return the canonical families that have surveyed landmarks."""
    return sorted({tag_key(tag)[0] for tag in environment['landmarks']})


def detector_parameters(environment, family):
    """Build detector parameters using the repository's exact family spelling."""
    # keep per-family identities and geometry together
    tags = [tag for tag in environment['landmarks'] if tag_key(tag)[0] == family]
    return {'family': DETECTOR_FAMILIES[family], 'pose_estimation_method': 'pnp',
            'tag.ids': [tag['id'] for tag in tags],
            'tag.frames': [tag_frame(tag_key(tag)) for tag in tags],
            'tag.sizes': [float(tag['tag_size_m']) for tag in tags]}
