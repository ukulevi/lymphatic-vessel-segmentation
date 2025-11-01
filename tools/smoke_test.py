import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

print('sys.path first entries:')
for p in sys.path[:5]:
    print(' -', p)

try:
    from src.config import ExperimentConfig
    from src.utils.augment import create_train_transform, create_val_transform
    from src.data.datasets import create_dataset, LabeledVesselDataset, VideoDataset, PseudoLabeledDataset

    print('ExperimentConfig default:')
    cfg = ExperimentConfig()
    print(cfg)

    print('\nTransforms:')
    tr = create_train_transform()
    vt = create_val_transform()
    print('train transform:', type(tr))
    print('val transform:', type(vt))

    print('\nDataset classes available:')
    print('LabeledVesselDataset ->', LabeledVesselDataset)
    print('VideoDataset ->', VideoDataset)
    print('PseudoLabeledDataset ->', PseudoLabeledDataset)

    # Try to create dataset factory without touching filesystem
    try:
        ds = create_dataset(cfg.paths, cfg.data, transform=tr, include_unlabeled=False, include_pseudo=False)
        print('\nCreated dataset via factory:', type(ds))
    except RuntimeError as e:
        print('\ncreate_dataset failed as expected (no valid data sources):', repr(e))
    except Exception as e:
        print('\nAn unexpected error occurred during dataset creation:', repr(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

except Exception as e:
    print('Error during import/setup:')
    import traceback
    traceback.print_exc()
    sys.exit(1)