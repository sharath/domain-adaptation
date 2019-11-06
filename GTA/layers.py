import torch.nn as nn

# ACGAN output layer
class ACModule(nn.Module):
    def __init__(self, filters, nclasses):
        super(ACModule, self).__init__()
        self.filters = filters
        self.classifier_c = nn.Sequential(nn.Linear(filters*2, nclasses))              
        self.classifier_s = nn.Sequential(
            nn.Linear(filters*2, 1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        real_prob = self.classifier_s(x.view(-1, self.filters*2)).view(-1)
        class_prob = self.classifier_c(x.view(-1, self.filters*2))
        return real_prob, class_prob