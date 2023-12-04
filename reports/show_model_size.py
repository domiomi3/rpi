from torchinfo import summary
from src.models.model import ModelWrapper, RNAProteinInterAct

"""
Script outputs a model summary with torchinfo. 
Enables an easy overview on how big individual parts of the network are.
"""

def main():
    model_1 = RNAProteinInterAct(batch_first=True,
                                 embed_dim=640,
                                 d_model=160,
                                 num_encoder_layers=1,
                                 nhead=2,
                                 dim_feedforward=320,
                                 norm_first=True)
    model_2 = ModelWrapper(model_1)
    batch_size = 128
    summary(model_2, [(batch_size, 150, 640), (batch_size, 1024, 640)], depth=5,
            col_names=("input_size", "output_size", "num_params"))


if __name__ == '__main__':
    main()
