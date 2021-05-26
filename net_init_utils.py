class NetInit:
    #
    # This class contains all initialisation information that are passed as arguments.
    #
    # CNNFilters:                   Number of output filters in the CNN layer
    # CNNKernel:                    CNN filter size in the CNN layer
    # RNNUnits:                     Number of hidden states in the RNN(LSTM/GRU) layer
    # SkipRNNUnits:                 Number of hidden states in the SkipRNN layer
    # FeatDims:                     The number of networks outputs
    # skip:                         Number of timeseries to skip. 0 => do not add Skip RNN layer
    # dropout:                      Dropout frequency
    # highway_window:               Number of timeseries values to consider for the linear layer (AR layer)

    def __init__(self):
        self.CNNFilters             = 32
        self.RNNUnits               = 32
        self.CNNKernel              = 6
        self.time_steps             = 24*7
        self.FeatDims               = 1
        self.task                   = 'forecasting'

        # LSTNet
        self.SkipRNNUnits           = 20
        self.skip                   = 24
        self.highway_window         = 24
        self.dropout                = 0.1

        # TCN
        self.en_cnn_hidden_sizes    = [self.CNNFilters] * 6
        self.dilations              = [1, 2, 4, 8, 16, 32]

        # MTNet
        self.en_rnn_hidden_sizes    = [self.RNNUnits, self.CNNFilters]
        self.ltms_n                 = 6
        
        # ResNet
        self.ResLayers              = [3, 3, 3]
        self.ResKernels             = [6, 6, 6]

        # LSTM-FCN
        self.is_attention           = False

        # NBeatsNet
        self.stack_types            = ['trend_block','seasonality_block', 'general'] #
        self.forecast_length        = 5
        self.backcast_length        = 10
        self.nb_blocks_per_stack    = 3
        self.hidden_layer_units     = 256
        self.thetas_dims            = (2, 8, 3)
        self.share_weights_in_stack = False

        # Transformer
        self.heads                  = 8
        self.encoder_stack          = 2
        self.model_dim              = 64 
        self.ff_dim                 = 256