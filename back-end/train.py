import torch
import warnings
from tqdm import tqdm
from system.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from architectures.autoencoder import *
from architectures.classifier import *
from architectures.decoder import *
from architectures.transformer import *

def train_specific(stage):
    if stage == "classifier":
        train_classifier()
    elif stage == "autoencoder":
        train_autoencoder()
    elif stage == "decoder":
        train_decoder()
    elif stage == "pretransformer":
        train_pre_transformer()
    elif stage == "rltransformer":
        train_rl_transformer()
    else:
        print("Invalid stage.")

def train_all():
    # Without dependencies
    train_classifier()
    train_autoencoder()
    train_decoder()
    train_pre_transformer()

    # With dependencies
    train_rl_transformer()

if __name__ == "__main__":
    # train_specific("autoencoder") Still need
    train_specific("rltransformer")  # Still need
    # extract_best_scientist(
    #     RESTORE_SCIENTISTS_PATH,
    #     NUM_SCIENTISTS,
    #     LATENT_DIM, EFFECTS_COUNT,
    #     MOLECULE_LENGTH,
    #     NUM_GENERATED_MOLECULES,
    #     BEST_SCIENTIST_PATH,
    #     EFFECTS_COUNT,
    #     CLASSES_COUNT
    # )
