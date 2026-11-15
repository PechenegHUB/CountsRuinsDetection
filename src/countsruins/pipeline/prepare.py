"""One-time preparation; never imported by the prediction entrypoint."""
import json
from importlib.metadata import version

from .cups import prepare_cup_head
from .items import prepare_item_heads
from .inputs import ASSETS, Inputs, digest, unpack
from .heads import export


def prepare(data, bundle, work, annotations, output):
    reference = json.loads((ASSETS / 'reference_compact.json').read_text(encoding='utf-8'))
    if digest(bundle) != reference['bundle_sha256']:
        raise ValueError('Unexpected input bundle')
    if not annotations.is_file():
        raise FileNotFoundError(annotations)
    if output.exists():
        raise FileExistsError('Choose a new heads output file')
    work.mkdir(parents=True, exist_ok=False)
    unpack(bundle, work / 'cache')
    inputs = Inputs(data, work / 'cache', training=True)
    models = {'cups': prepare_cup_head(inputs)}
    models.update(prepare_item_heads(inputs, annotations))
    export(models, output, {
        'bundle_sha256': digest(bundle), 'annotations_sha256': digest(annotations),
        'versions': {n: version(n) for n in ('numpy', 'scipy', 'pandas', 'scikit-learn')},
        'scope': 'Three proposal verifiers. Base counts and occupancy are upstream inputs.',
    })
    print('Saved models:', output, flush=True)
