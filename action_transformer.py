import torch.nn as nn
import sys
import torch

# sys.path.insert(0, "../pytorchI3d")
from pytorchI3d.torchi3d import InceptionI3d

from transformer_encoder import TransformerEncoder


class ActionTransformer(nn.Module):
    """ Pytorch implementation of http://arxiv.org/abs/1812.02707
    - n_heads = number of attention heads for the transformer
    - model_dim = number of dimensions the model will use for
                  transformer section.
    - trunk_weights = Path to the .pth or .pt file containing
                      parameter weights for the trunk.
    - n_proposals =  max number of proposals for RPN
    - query_preproc = determines whether to do High Res or Low Res
                      preprocessing
    """

    def __init__(self, n_heads=2, model_dim=128, trunk_weights=None
                 , n_proposals=300, query_preproc='High', n_classes=400):
        super(ActionTransformer, self).__init__()
        self.in_channels = 832
        self.out_channels = 128
        self.encoding_dim = 50
        self.n_classes = n_classes
        # Model Parameters
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.trunk = self.load_trunk(trunk_weights)
        self.n_proposals = n_proposals
        self.query_preproc = query_preproc
        # Key, value projection layers
        self.key_proj = nn.Linear(self.in_channels + 2 * self.encoding_dim, self.model_dim, bias=True)
        self.val_proj = nn.Linear(self.in_channels + 2 * self.encoding_dim, self.model_dim, bias=True)
        self.roi_pool = torch.ones(2, self.in_channels + 2 * self.encoding_dim, 14, 14,
                                   device='cpu')  # TODO: real ROIPOOL
        self.max_pool_roi = nn.MaxPool2d(2, stride=2)
        # Query Preprocessing
        self.query_dim_reduction = nn.Conv2d(self.in_channels + 2 * self.encoding_dim, self.out_channels, kernel_size=1,
                                             stride=1, bias=True)
        # High res preprocessor
        self.high_res = nn.Linear(7 * 7 * self.out_channels, self.model_dim, bias=True)
        # Low res preprocessor
        self.low_res = nn.AvgPool2d(7)
        # Spatiotemporal Encodings
        self.temp_encoding = nn.Linear(1, self.encoding_dim)
        self.pos_encoding = nn.Sequential(nn.Linear(2, self.encoding_dim),
                                          nn.Linear(self.encoding_dim, self.encoding_dim))
        # Transformer Section
        self.transformer = TransformerEncoder(N=3, d_model=self.model_dim, h=self.n_heads)
        # Final Bbox and cls heads
        self.bbox_head = nn.Conv2d(self.model_dim, 4, kernel_size=1, stride=1, bias=True)
        self.cls_head = nn.Conv2d(self.model_dim, self.n_classes + 1, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        # Process features through trunk and add pos encoding
        input_size = x.size()

        batch_size = input_size[0]
        features = self.process_features(x)

        # Add spatiotemporal encodings
        temp_channels = self.temporal_encoding(features)
        pos_channels = self.positional_encoding(features)

        features = torch.cat((features, temp_channels, pos_channels), dim=1)
        # Extract temporally central frames from video
        central_frames = model.select_temporal_center(features)

        # Project features to (Q, K, V) spaces

        features = features.permute(0, 2, 3, 4, 1)

        keys = self.key_proj(features).view(batch_size, -1, self.model_dim)
        values = self.val_proj(features).view(batch_size, -1, self.model_dim)
        maxed = self.max_pool_roi(self.roi_pool)
        query = self.process_query(maxed).view(batch_size, -1, self.model_dim)

        output = self.transformer(query, keys, values).permute(0, 2, 1).view(batch_size, self.model_dim, 1, 1)

        # Classification
        bbox = self.bbox_head(output).squeeze()
        cls = self.cls_head(output).squeeze()
        return bbox, cls

    def load_trunk(self, trunk_weights=None):
        model = InceptionI3d()
        if trunk_weights:
            print("Loading weights from path: {}".format(trunk_weights))
            model.load_state_dict(torch.load(trunk_weights))

        else:
            print("Loading model from scratch...")

        return model

    """ 
    process_features
    Passes input video through trunk and adds spatiotemporal
    encoding
    - Input: a (batch_size, n_channels, frames, x, y) tensor
    - Output: a (batch_size, channels, frames, x, y) tensor 
    """

    def process_features(self, input):

        # Extract features from Mixed_4f layer from I3D
        features = self.trunk.extract_features_from_layer(input, 'Mixed_4f')
        return features

    """ 
    temporal_encoding
    - Input: a tensor of size (batch_size, channels, frames, x, y)
    - Output: a tensor of size (batch_size, encoding_dim, frames, x, y)
    """

    def temporal_encoding(self, features):
        # Temporal Embedding
        input_size = features.size()

        batch_size = input_size[0]

        n_frames, h, w = input_size[2:5]

        t = torch.tensor([a - input_size[2] / 2 for a in range(input_size[2])], dtype=torch.float, device='cpu').view(
            n_frames, 1)

        temporal_channels = torch.zeros(n_frames, self.encoding_dim, device='cpu')

        for i, frame in enumerate(t):
            temporal_channels[i] = self.temp_encoding(frame)

        # Rearranging output for concatenation with features
        temporal_channels = temporal_channels.transpose(1, 0)

        temporal_channels = temporal_channels.view(1, self.encoding_dim, n_frames, 1, 1).expand(batch_size, -1, -1, h,
                                                                                                w)

        return temporal_channels

    """ 
    positional_encoding
    - Input: a tensor of size (batch_size, channels, frames, x, y)
    - Output: a tensor of size (batch_size, encoding_dim, frames, x ,y)
    """

    def positional_encoding(self, features):

        input_size = features.size()

        batch_size = input_size[0]

        n_frames, h, w = input_size[2:5]

        x = torch.tensor([a - h / 2 for a in range(h)], dtype=torch.float, device='cpu')
        y = torch.tensor([a - w / 2 for a in range(w)], dtype=torch.float, device='cpu')

        C = (torch.tensor([i, j], device='cpu') for i in x for j in y)
        pos_channels = torch.zeros((h, w, self.encoding_dim), device='cpu')

        pos_channels = pos_channels.view(h * w, self.encoding_dim)

        for i, el in enumerate(C):
            pos_channels[i] = self.pos_encoding(el)

        # Rearranging output for concatenation with features
        pos_channels = pos_channels.view(h, w, self.encoding_dim)

        pos_channels = pos_channels.permute(2, 0, 1)

        pos_channels = pos_channels.view(1, self.encoding_dim, 1, h, w)

        pos_channels = pos_channels.expand(batch_size, -1, n_frames, -1, -1)

        return pos_channels

    """ 
    process_query
    - Input: a (batch_size, n_channels, 7, 7) tensor
    - Output: a (batch_size, model_dim) tensor representing a 
      projection of the proposal to query space.
    - self.query_preproc determines whether to do High Res or 
    Low Res preprocesing.
    """

    def process_query(self, query):

        query = self.query_dim_reduction(query)
        if self.query_preproc == 'High':

            query = self.high_res(query.view(-1, 7 * 7 * self.out_channels))

        else:

            query = self.low_res(query).view(-1, self.out_channels)

        return query

    """ 
    select_temporal_center
    - Input: a (batch_size, channels, frames, x, y) representing the processed
      features from the video after going through the trunk
    - Output: a (batch_size, channels, x, y) tensor representing the temporally
    central frame of the video.
    """

    def select_temporal_center(self, input):
        input_size = input.size()
        center = int(input_size[2] / 2)
        return input[:, :, center, :, :]


# Testing code
device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available()

model = ActionTransformer(trunk_weights='C:/Users/wlals/PycharmProjects/videoAnalyze-Transformer/pytorchI3d/models'
                                        '/rgb_imagenet.pt',
                          query_preproc='Low',
                          n_heads=2,
                          model_dim=128)
model.to(device)
input = torch.zeros(2, 3, 64, 400, 400, device=device)

print("Running inference on {}...".format(device))

# features, query, keys, values, maxed  = model(input)

# print("Features: ", features.size(), " Query: ", query.size() , " Keys: ", keys.size(), " Values: ", values.size(),
# "MaxPool", maxed.size())
bbox, cls = model(input)

print("Output: ", bbox.size(), cls.size())
# print("Output:", bbox, cls)
